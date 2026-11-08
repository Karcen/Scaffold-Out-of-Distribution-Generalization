"""Fixed, resumable Chemprop D-MPNN fits for the 14 October revision."""

from __future__ import annotations
import os
import sys
from pathlib import Path

# A sibling statistics.py is a project analysis program, not Python's stdlib.
# Keep third-party imports from resolving that sibling as ``statistics``.
_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != _SCRIPT_DIRECTORY]

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from chemprop import data, featurizers, models, nn
import chemprop
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/revision/experiments"
OUT = ROOT / "results/review_2026_10_14/chemprop"
PROTOCOL = ROOT / "docs/review_2026_10_14/CHEMPROP_PROTOCOL.md"
sys.path.insert(0, str(ROOT / "codes/scaffold"))
from src.data import TASKS
from src.metrics import evaluate

RECIPE = "chemprop_2_3_1_cpu_fixed"
REGIMES = ("scaffold", "descriptor", "morgan")
CODE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PROTOCOL_HASH = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
# The pilot version differs only in process-specific repr/resume metadata.
COMPATIBLE_COMPLETED_CODE = {CODE_HASH, "ba5737eeca2c31e583e288cf1b8e2c872c5c10fb4ffdd8926d7bcad4f9134f56"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).relative_to(ROOT))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def copy_input(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert sha(source) == sha(destination), f"Changed input: {destination}"
    else:
        shutil.copyfile(source, destination)
    assert sha(source) == sha(destination)


def normalized_identity(config, completed=False):
    result = clean(config)
    result = json.loads(json.dumps(result))
    if completed:
        code_hash = result.pop("script_sha256")
        assert code_hash in COMPATIBLE_COMPLETED_CODE, "Unapproved source change"
        assert sha(OUT / "code_snapshots" / f"{code_hash}.py") == code_hash
    result["graph_metadata"].pop("cache_seconds", None)
    result["graph_metadata"]["featurizer"] = re.sub(
        r" at 0x[0-9a-fA-F]+", "", result["graph_metadata"]["featurizer"])
    return result


def predict(model, loader, scaler=None):
    arrays = []
    with torch.no_grad():
        for batch in loader:
            graph, atom_descriptors, molecule_descriptors, *_ = batch
            arrays.append(model(graph, atom_descriptors, molecule_descriptors).detach().cpu().numpy())
    prediction = np.concatenate(arrays).reshape(-1)
    return scaler.inverse_transform(prediction.reshape(-1, 1)).ravel() if scaler is not None else prediction


class ExactValidation(Callback):
    """Use project primary metrics and persist each epoch before checkpointing."""

    def __init__(self, spec, loader, original_targets, scaler, folder, fallback):
        self.spec, self.loader, self.y = spec, loader, original_targets
        self.scaler, self.folder, self.fallback = scaler, Path(folder), fallback
        history = self.folder / "history.csv"
        self.rows = pd.read_csv(history).to_dict("records") if history.exists() else []
        self.start = time.monotonic()

    def on_validation_epoch_end(self, trainer, module):
        if trainer.sanity_checking:
            return
        prediction = predict(module, self.loader, self.scaler)
        metrics = evaluate(self.spec.kind, self.y, prediction)
        if self.fallback:
            selected = float(log_loss(self.y, np.clip(prediction, 1e-7, 1 - 1e-7), labels=[0, 1]))
        else:
            selected = metrics[self.spec.primary_metric]
        if not np.isfinite(selected):
            raise ValueError("Nonfinite prespecified validation score")
        module.log("review_val_primary", torch.tensor(selected, dtype=torch.float32),
                   on_step=False, on_epoch=True, logger=True, prog_bar=False)
        row = {"epoch": int(trainer.current_epoch),
               "train_loss": float(module.criterion.compute().detach().cpu()),
               "val_loss": float(module.metrics[-1].compute().detach().cpu()),
               "val_primary_original_units": metrics[self.spec.primary_metric],
               "selection_score": selected, "selection_fallback_bce": self.fallback,
               "learning_rate": trainer.optimizers[0].param_groups[0]["lr"],
               "callback_elapsed_seconds": time.monotonic() - self.start}
        self.rows = [r for r in self.rows if int(r["epoch"]) != row["epoch"]] + [row]
        pd.DataFrame(self.rows).sort_values("epoch").to_csv(self.folder / "history.csv", index=False)


def prepare_endpoint(endpoint):
    folder = OUT / endpoint
    for filename in ("dataset.csv.gz", "dataset_config.json"):
        copy_input(SOURCE / endpoint / filename, folder / filename)
    frame = pd.read_csv(folder / "dataset.csv.gz")
    assert frame.row_index.tolist() == list(range(len(frame)))
    assert frame.connectivity_key.is_unique
    for regime in REGIMES:
        for seed in range(5):
            name = f"{regime}_test20_seed{seed}.csv.gz"
            copy_input(SOURCE / endpoint / "splits" / name, folder / "splits" / name)
    datapoints, invalid = [], []
    for row in frame.itertuples():
        try:
            datapoints.append(data.MoleculeDatapoint.from_smi(
                row.smiles, np.array([row.y], dtype=np.float32), name=str(row.row_index)))
        except Exception as error:
            invalid.append({"row_index": row.row_index, "connectivity_key": row.connectivity_key,
                            "smiles": row.smiles, "error": repr(error)})
    dump(folder / "parse_audit.json", {"n": len(frame), "invalid": invalid, "rdkit": rdBase.rdkitVersion})
    if invalid:
        raise ValueError(f"Graph parsing failed for {len(invalid)} rows; no rows removed")
    graph_featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    whole = data.MoleculeDataset(datapoints, graph_featurizer, n_workers=0)
    start = time.monotonic()
    whole.cache = True
    h = hashlib.sha256()
    graph_rows = []
    for i in range(len(whole)):
        graph = whole.mg_cache[i]
        for field in ("V", "E", "edge_index", "rev_edge_index"):
            array = np.asarray(getattr(graph, field))
            h.update(str(array.dtype).encode()); h.update(str(array.shape).encode()); h.update(array.tobytes())
        graph_rows.append({"row_index": i, "atom_n": len(graph.V), "directed_bond_n": len(graph.E)})
    pd.DataFrame(graph_rows).to_csv(folder / "graph_audit.csv", index=False)
    meta = {"endpoint": endpoint, "n": len(frame), "dataset_sha256": sha(folder / "dataset.csv.gz"),
            "rdkit": rdBase.rdkitVersion, "chemprop": chemprop.__version__,
            "featurizer": re.sub(r" at 0x[0-9a-fA-F]+", "", repr(graph_featurizer)),
            "graph_shapes": graph_featurizer.shape,
            "graph_arrays_sha256": h.hexdigest(), "cache_seconds": time.monotonic() - start,
            "extra_descriptors": None, "cuik_featurization": False,
            "recanonicalisation_or_deduplication": False}
    dump(folder / "graph_config.json", meta)
    return frame, whole, meta


def composition(kind, targets, parts):
    result = {}
    for name, index in parts.items():
        y = targets[index]
        row = {"n": len(y)}
        if kind == "binary":
            row.update({"positive_n": int((y == 1).sum()), "negative_n": int((y == 0).sum()),
                        "positive_fraction": float(y.mean())})
        else:
            row.update({"target_mean": y.mean(), "target_sd": y.std(ddof=1),
                        "target_median": np.median(y), "target_min": y.min(), "target_max": y.max()})
        result[name] = row
    return result


def run(spec, frame, whole, graph_meta, regime, seed, environment_hash):
    rid = f"{spec.name}__{regime}__test20__graph__chemprop_dmpnn__seed{seed}__{RECIPE}"
    folder = OUT / spec.name / "runs" / rid
    split = OUT / spec.name / "splits" / f"{regime}_test20_seed{seed}.csv.gz"
    manifest = pd.read_csv(split)
    assert manifest.row_index.tolist() == frame.row_index.tolist()
    assert manifest.connectivity_key.tolist() == frame.connectivity_key.tolist()
    assert np.array_equal(manifest.y.to_numpy(), frame.y.to_numpy())
    assert manifest.groupby("group").partition.nunique().max() == 1
    parts = {p: np.flatnonzero(manifest.partition.eq(p)) for p in ("train", "val", "test")}
    train, val, test = parts["train"], parts["val"], parts["test"]
    y = frame.y.to_numpy()
    if not len(val) or not len(test):
        raise ValueError("Empty validation/test partition")
    if spec.kind == "binary" and len(np.unique(y[train])) < 2:
        raise ValueError("Single-class training partition")
    fallback = spec.kind == "binary" and len(np.unique(y[val])) < 2
    scaler = StandardScaler().fit(y[train].reshape(-1, 1)) if spec.kind == "regression" else None
    whole.Y = scaler.transform(y.reshape(-1, 1)).astype(np.float32) if scaler is not None else y.reshape(-1, 1).astype(np.float32)
    target_metadata = {"kind": "unchanged_class_labels"} if scaler is None else {
        "kind": "training_fitted_StandardScaler", "mean": scaler.mean_, "scale": scaler.scale_,
        "validation_test_transform": "training_parameters_only", "prediction_units": "inverse_to_original"}
    config = {
        "run_id": rid, "endpoint": spec.name, "task_kind": spec.kind,
        "regime": regime, "seed": seed, "allocation_seed": seed, "fitting_seed": seed,
        "test_target": 0.2, "validation_target": 0.1, "representation": "graph",
        "model": "chemprop_dmpnn", "recipe": RECIPE, "primary_metric": spec.primary_metric,
        "source_dataset": relative(SOURCE / spec.name / "dataset.csv.gz"),
        "dataset": relative(OUT / spec.name / "dataset.csv.gz"),
        "dataset_sha256": sha(OUT / spec.name / "dataset.csv.gz"),
        "source_split_manifest": relative(SOURCE / spec.name / "splits" / split.name),
        "split_manifest": relative(split), "split_sha256": sha(split),
        "script_sha256": CODE_HASH, "protocol_sha256": PROTOCOL_HASH,
        "environment_sha256": environment_hash, "graph_metadata": graph_meta,
        "metrics_source_sha256": sha(ROOT / "codes/scaffold/src/metrics.py"),
        "target_transformation": target_metadata, "composition": composition(spec.kind, y, parts),
        "architecture": {"message_passing": "BondMessagePassing", "hidden_dim": 300,
                         "depth": 3, "bias": False, "undirected": False, "aggregation": "mean",
                         "ffn_hidden_layers": 2, "ffn_hidden_dim": 300, "dropout": 0,
                         "activation": "relu", "batch_norm": False, "additional_descriptors": False},
        "training": {"optimizer": "Adam", "weight_decay": 0, "gradient_clipping": None,
                     "warmup_epochs": 2, "init_lr": 1e-4, "max_lr": 1e-3, "final_lr": 1e-4,
                     "max_epochs": 50, "batch_size": 64, "drop_last": False,
                     "class_balance": False, "accelerator": "cpu", "devices": 1,
                     "threads": 2, "interop_threads": 1, "num_workers": 0, "deterministic": True,
                     "early_stopping_patience": 10, "min_delta": 0,
                     "selection": "validation_bce" if fallback else "validation_" + spec.primary_metric,
                     "selection_fallback": "single_class_validation" if fallback else None,
                     "selection_mode": "min" if fallback or spec.kind == "regression" else "max"},
    }
    complete = folder / "summary.json"
    if complete.exists():
        summary = json.loads(complete.read_text())
        if summary["status"] != "complete":
            raise ValueError(f"Incomplete existing summary: {rid}")
        stored = json.loads((folder / "config.json").read_text())
        if normalized_identity(stored, completed=True) != normalized_identity(config, completed=True):
            raise ValueError(f"Resume identity mismatch: {rid}")
        assert sha(folder / "predictions.csv.gz") == summary["predictions_sha256"]
        print("SKIP COMPLETE", rid, flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    existing = folder / "config.json"
    if existing.exists():
        stored = json.loads(existing.read_text())
        assert normalized_identity(stored) == normalized_identity(config), f"Cannot resume changed incomplete fit: {rid}"
    else:
        dump(existing, config)
    start = time.monotonic()
    try:
        pl.seed_everything(seed, workers=True, verbose=False)
        loaders = {p: data.build_dataloader(Subset(whole, index.tolist()), batch_size=64,
                                           num_workers=0, shuffle=(p == "train"),
                                           seed=seed if p == "train" else None,
                                           class_balance=False, drop_last=False)
                   for p, index in parts.items()}
        mp = nn.BondMessagePassing(d_v=whole.featurizer.shape[0], d_e=whole.featurizer.shape[1],
                                  d_h=300, depth=3, dropout=0, bias=False, undirected=False)
        head_class = nn.BinaryClassificationFFN if spec.kind == "binary" else nn.RegressionFFN
        head = head_class(n_tasks=1, input_dim=300, hidden_dim=300, n_layers=2, dropout=0)
        model = models.MPNN(mp, nn.MeanAggregation(), head, batch_norm=False,
                            warmup_epochs=2, init_lr=1e-4, max_lr=1e-3, final_lr=1e-4)
        n_params = sum(p.numel() for p in model.parameters())
        callback = ExactValidation(spec, loaders["val"], y[val], scaler, folder, fallback)
        check = ModelCheckpoint(dirpath=folder / "checkpoints", filename="best",
                                monitor="review_val_primary", mode=config["training"]["selection_mode"],
                                save_top_k=1, save_last=True, auto_insert_metric_name=False)
        stop = EarlyStopping(monitor="review_val_primary", mode=config["training"]["selection_mode"],
                             patience=10, min_delta=0, check_finite=True)
        logger = CSVLogger(save_dir=str(folder), name="training_log", version=0)
        trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=50,
                             callbacks=[callback, check, stop], logger=logger,
                             deterministic=True, enable_progress_bar=False,
                             enable_model_summary=False, num_sanity_val_steps=0,
                             log_every_n_steps=1)
        last = folder / "checkpoints/last.ckpt"
        resumed = last.exists()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trainer.fit(model, loaders["train"], loaders["val"], ckpt_path=str(last) if resumed else None)
            best = Path(check.best_model_path)
            if not best.is_file():
                raise ValueError("No valid selected checkpoint")
            checkpoint = torch.load(best, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            prediction = predict(model, loaders["test"], scaler)
            metrics = evaluate(spec.kind, y[test], prediction)
        assert len(prediction) == len(test) and np.isfinite(prediction).all()
        result = frame.iloc[test][["row_index", "connectivity_key", "smiles"]].copy()
        result["group"] = manifest.iloc[test].group.to_numpy()
        result["y_true"] = y[test]
        result["prediction"] = prediction
        result.to_csv(folder / "predictions.csv.gz", index=False,
                      compression={"method": "gzip", "mtime": 0})
        selected_epoch = int(checkpoint["epoch"])
        history = pd.read_csv(folder / "history.csv")
        row = history.loc[history.epoch.eq(selected_epoch)].iloc[0]
        summary = {
            **{k: config[k] for k in ("run_id", "endpoint", "task_kind", "regime", "seed", "test_target", "representation", "model", "recipe", "primary_metric")},
            "status": "complete", **metrics, "train_n": len(train), "val_n": len(val), "test_n": len(test),
            "test_positive_fraction": y[test].mean() if spec.kind == "binary" else None,
            "selected_epoch": selected_epoch, "epochs": len(history), "n_params": n_params,
            "best_validation_primary": row.val_primary_original_units,
            "best_selection_score": row.selection_score, "selection_fallback_bce": fallback,
            "seconds": time.monotonic() - start, "seconds_this_invocation": time.monotonic() - start,
            "resumed_from_last_checkpoint": resumed,
            "mean_epoch_callback_seconds": history.callback_elapsed_seconds.max() / len(history),
            "best_checkpoint": relative(best), "best_checkpoint_sha256": sha(best),
            "config_sha256": sha(folder / "config.json"),
            "predictions_sha256": sha(folder / "predictions.csv.gz"),
            "history_sha256": sha(folder / "history.csv"),
            "warnings": [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught],
        }
        dump(complete, summary)
        print(f"COMPLETE {rid} epochs={len(history)} seconds={summary['seconds']:.2f} {spec.primary_metric}={metrics[spec.primary_metric]:.6f}", flush=True)
    except Exception as error:
        dump(folder / "failure.json", {"run_id": rid, "error": repr(error), "traceback": traceback.format_exc(),
                                      "seconds": time.monotonic() - start,
                                      "script_sha256": CODE_HASH, "protocol_sha256": PROTOCOL_HASH})
        raise


def aggregate():
    records = [json.loads(p.read_text()) for p in sorted(OUT.glob("*/runs/*/summary.json")) if not p.name.startswith("._")]
    if records:
        pd.DataFrame(records).to_csv(OUT / "runs.csv", index=False)
    files = [p for p in OUT.rglob("*") if p.is_file() and not p.name.startswith("._")
             and p.name not in ("sha256_manifest.json", "sha256_manifest.json.tmp")]
    dump(OUT / "sha256_manifest.json", {relative(p): sha(p) for p in sorted(files)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["DILI", "BBB", "HIA", "PPBR", "CYP2D6", "Solubility"])
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    assert SOURCE.is_dir(), "Original mounted project input registry is required"
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    copy_input(Path(__file__), OUT / "code_snapshots" / f"{CODE_HASH}.py")
    copy_input(PROTOCOL, OUT / "protocol_snapshots" / f"{PROTOCOL_HASH}.md")
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    environment = {"python": sys.version, "platform": platform.platform(), "packages": versions,
                   "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                   "accelerator": "cpu", "code_sha256": CODE_HASH, "protocol_sha256": PROTOCOL_HASH}
    envpath = OUT / "environment.json"
    if envpath.exists():
        stored_environment = json.loads(envpath.read_text())
        old_code = stored_environment.pop("code_sha256")
        assert old_code in COMPATIBLE_COMPLETED_CODE
        assert sha(OUT / "code_snapshots" / f"{old_code}.py") == old_code
        runtime_environment = dict(environment)
        runtime_environment.pop("code_sha256")
        assert stored_environment == runtime_environment, "Environment changed since protocol launch"
    else:
        dump(envpath, environment)
        (OUT / "requirements-lock.txt").write_text("\n".join(f"{n}=={v}" for n, v in sorted(versions.items(), key=lambda item: item[0].lower())) + "\n")
    environment_hash = sha(envpath)
    names = ["DILI"] if args.pilot_only else args.tasks
    for endpoint in names:
        spec = next(t for t in TASKS if t.name == endpoint)
        frame, whole, graph_meta = prepare_endpoint(endpoint)
        if endpoint == "DILI":
            run(spec, frame, whole, graph_meta, "descriptor", 0, environment_hash)
            aggregate()
            if args.pilot_only:
                print("FORMAL PILOT COMPLETE", flush=True)
                return
        for regime in REGIMES:
            for seed in range(5):
                run(spec, frame, whole, graph_meta, regime, seed, environment_hash)
                aggregate()
        print("ENDPOINT COMPLETE", endpoint, flush=True)
    print("CHEMPROP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
