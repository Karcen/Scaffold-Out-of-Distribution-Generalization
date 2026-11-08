"""Read-only validation of the complete fixed Chemprop prediction registry."""
from pathlib import Path
import os, sys

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != SCRIPT_DIRECTORY]

import hashlib, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/review_2026_10_14/chemprop"
EXP = ROOT / "results/revision/experiments"
sys.path.insert(0, str(ROOT / "codes/scaffold"))
from src.data import TASKS
from src.metrics import evaluate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    registry = pd.read_csv(OUT / "runs.csv")
    assert len(registry) == 90 and registry.run_id.is_unique
    expected = {(e, r, s) for e in ("BBB", "HIA", "DILI", "PPBR", "CYP2D6", "Solubility")
                for r in ("scaffold", "descriptor", "morgan") for s in range(5)}
    assert set(zip(registry.endpoint, registry.regime, registry.seed)) == expected
    assert registry.status.eq("complete").all()
    assert registry.recipe.eq("chemprop_2_3_1_cpu_fixed").all()
    metrics_delta, primary_delta, source_hashes, validation_rows = [], [], set(), []
    for run in registry.itertuples():
        folder = OUT / run.endpoint / "runs" / run.run_id
        config = json.loads((folder / "config.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        assert sha(folder / "config.json") == summary["config_sha256"]
        assert sha(folder / "predictions.csv.gz") == summary["predictions_sha256"]
        assert sha(folder / "history.csv") == summary["history_sha256"]
        assert sha(ROOT / summary["best_checkpoint"]) == summary["best_checkpoint_sha256"]
        source_hashes.add(config["script_sha256"])
        assert sha(OUT / "code_snapshots" / f"{config['script_sha256']}.py") == config["script_sha256"]
        assert sha(OUT / "protocol_snapshots" / f"{config['protocol_sha256']}.md") == config["protocol_sha256"]
        assert sha(OUT / "environment.json") == config["environment_sha256"]
        assert sha(ROOT / config["dataset"]) == sha(ROOT / config["source_dataset"]) == config["dataset_sha256"]
        assert sha(ROOT / config["split_manifest"]) == sha(ROOT / config["source_split_manifest"]) == config["split_sha256"]
        assert sha(ROOT / "codes/scaffold/src/metrics.py") == config["metrics_source_sha256"]
        dataset = pd.read_csv(ROOT / config["dataset"])
        split = pd.read_csv(ROOT / config["split_manifest"])
        prediction = pd.read_csv(folder / "predictions.csv.gz")
        history = pd.read_csv(folder / "history.csv")
        assert np.array_equal(split.row_index, dataset.row_index)
        assert np.array_equal(split.connectivity_key, dataset.connectivity_key)
        assert np.array_equal(split.y, dataset.y)
        assert split.groupby("group").partition.nunique().max() == 1
        test = split[split.partition.eq("test")]
        train = split[split.partition.eq("train")]
        val = split[split.partition.eq("val")]
        assert np.array_equal(prediction.row_index, test.row_index)
        assert np.array_equal(prediction.connectivity_key, test.connectivity_key)
        assert np.array_equal(prediction.smiles, dataset.set_index("row_index").loc[test.row_index].smiles)
        assert np.array_equal(prediction.group.astype(str), test.group.astype(str))
        assert np.array_equal(prediction.y_true, test.y)
        assert not prediction.row_index.duplicated().any()
        assert np.isfinite(prediction.prediction).all()
        assert len(prediction) == summary["test_n"] == config["composition"]["test"]["n"]
        assert len(train) == summary["train_n"] and len(val) == summary["val_n"]
        spec = next(t for t in TASKS if t.name == run.endpoint)
        scores = evaluate(spec.kind, prediction.y_true.to_numpy(), prediction.prediction.to_numpy())
        for name, score in scores.items():
            saved = summary[name]
            if not np.isfinite(score):
                assert saved is None
                continue
            difference = abs(score - saved)
            assert difference < 1e-6, (run.run_id, name, difference)
            metrics_delta.append(difference)
        primary = (mean_absolute_error(prediction.y_true, prediction.prediction) if spec.kind == "regression"
                   else average_precision_score(prediction.y_true, prediction.prediction) if spec.primary_metric == "auprc"
                   else roc_auc_score(prediction.y_true, prediction.prediction))
        primary_delta.append(abs(primary - summary[spec.primary_metric]))
        assert abs(primary - getattr(run, spec.primary_metric)) < 1e-6
        assert np.array_equal(history.epoch.to_numpy(), np.arange(len(history)))
        assert len(history) == summary["epochs"] <= 50
        assert summary["n_params"] == 408601
        selected = history.loc[history.epoch.eq(summary["selected_epoch"])].iloc[0]
        assert abs(selected.selection_score - summary["best_selection_score"]) < 1e-6
        mode = config["training"]["selection_mode"]
        optimum = history.selection_score.min() if mode == "min" else history.selection_score.max()
        assert abs(selected.selection_score - optimum) < 2e-7, run.run_id
        expected_fallback = spec.kind == "binary" and val.y.nunique() < 2
        assert config["training"]["selection_fallback"] == ("single_class_validation" if expected_fallback else None)
        assert summary["selection_fallback_bce"] == expected_fallback
        if spec.kind == "regression":
            transform = config["target_transformation"]
            assert abs(transform["mean"][0] - train.y.mean()) < 1e-10
            assert abs(transform["scale"][0] - train.y.std(ddof=0)) < 1e-10
            assert transform["prediction_units"] == "inverse_to_original"
        else:
            assert prediction.prediction.between(0, 1).all()
            assert config["target_transformation"]["kind"] == "unchanged_class_labels"
        parse = json.loads((OUT / run.endpoint / "parse_audit.json").read_text())
        assert not parse["invalid"] and parse["n"] == len(dataset)
        assert config["graph_metadata"]["graph_shapes"] == [72, 14]
        assert not config["graph_metadata"]["recanonicalisation_or_deduplication"]
        validation_rows.append({"run_id": run.run_id, "endpoint": run.endpoint,
                                "regime": run.regime, "seed": int(run.seed),
                                "metric": spec.primary_metric, "recomputed_primary": primary,
                                "test_n": len(test), "test_groups": test.group.nunique(),
                                "selection_epoch": summary["selected_epoch"], "epochs": len(history),
                                "status": "pass"})
    manifest = json.loads((OUT / "sha256_manifest.json").read_text())
    for path, digest in manifest.items():
        assert sha(ROOT / path) == digest, path
    assert not list(OUT.glob("*/runs/*/failure.json"))
    report = {"status": "pass", "expected_records": 90, "validated_records": len(registry),
              "per_run": validation_rows, "sha256_entries_checked": len(manifest),
              "source_sha256": sorted(source_hashes),
              "maximum_metric_recalculation_difference": max(metrics_delta),
              "maximum_primary_metric_difference": max(primary_delta),
              "selected_checkpoint_rule": "validation-only optimum within float32 monitor tolerance 2e-7",
              "scope": "Exact canonical row/label/group identity, source/split copies, per-run hashes, probability/target units, training-fitted scaler, complete epoch sequence and selected checkpoint; no refitting or tuning.",
              "validator_sha256": sha(__file__), "registry_sha256": sha(OUT / "runs.csv"),
              "manifest_sha256": sha(OUT / "sha256_manifest.json")}
    destination = OUT.parent / "chemprop_validation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_run"}, indent=2))


if __name__ == "__main__":
    main()
