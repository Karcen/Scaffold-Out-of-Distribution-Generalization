"""Frozen, resumable classical baselines for the 14 October reviewer revision."""

from __future__ import annotations

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "2"

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import RobustScaler, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "results/revision/experiments"
OUT = ROOT / "results/review_2026_10_14/baselines"
PROTOCOL = ROOT / "docs/review_2026_10_14/BASELINE_PROTOCOL.md"
sys.path.insert(0, str(ROOT / "codes/scaffold"))
sys.path.insert(0, str(ROOT / "codes/repro"))
from src.data import TASKS
from src.metrics import evaluate
from grid_common import physchem, maccs

SEEDS = range(5)
REGIMES = ("scaffold", "descriptor", "morgan")
RECIPE = "common_baselines_2026_10_14"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).relative_to(ROOT))


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def copy_verified(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(source) != sha(destination):
            raise ValueError(f"Changed copied input: {destination}")
    else:
        shutil.copyfile(source, destination)
    assert sha(source) == sha(destination)


def prepare(endpoint):
    src = OLD / endpoint
    dst = OUT / endpoint
    for name in ("dataset.csv.gz", "dataset_config.json"):
        copy_verified(src / name, dst / name)
    data = pd.read_csv(dst / "dataset.csv.gz")
    assert data.row_index.tolist() == list(range(len(data)))
    assert data.connectivity_key.is_unique
    old = np.load(src / "features.npz", allow_pickle=False)
    feature_path = dst / "features.npz"
    if feature_path.exists():
        features = {key: arr for key, arr in np.load(feature_path, allow_pickle=False).items()}
    else:
        features = {"morgan": old["morgan"], "groups": old["groups"]}
        for rep, fn in (("physchem", physchem), ("maccs", maccs)):
            features[rep] = old[rep] if rep in old.files else np.stack([fn(s) for s in data.smiles])
        np.savez_compressed(feature_path, **features)
    assert features["physchem"].shape == (len(data), 9)
    assert features["maccs"].shape == (len(data), 167)
    assert features["morgan"].shape == (len(data), 1024)
    assert np.array_equal(features["morgan"], old["morgan"])
    assert np.array_equal(features["groups"], old["groups"])
    for regime in REGIMES:
        for seed in SEEDS:
            name = f"{regime}_test20_seed{seed}.csv.gz"
            copy_verified(src / "splits" / name, dst / "splits" / name)
    dump(dst / "feature_config.json", {
        "endpoint": endpoint, "n": len(data), "old_cache": relative(src / "features.npz"),
        "old_cache_sha256": sha(src / "features.npz"), "feature_cache_sha256": sha(feature_path),
        "dataset_sha256": sha(dst / "dataset.csv.gz"),
        "feature_source_sha256": sha(ROOT / "codes/repro/grid_common.py"),
        "physchem": "nine grid_common descriptors", "maccs_bits": 167,
        "morgan_radius": 2, "morgan_bits": 1024,
        "physchem_and_maccs_source": "prior cache" if endpoint in ("DILI", "BBB") else "same featuriser, recomputed",
    })
    return data, features


def transform(features, representation, train):
    raw = features[representation]
    if representation != "physchem":
        return raw.astype(np.float32), {"binary_fingerprints": "unchanged"}
    raw = np.asarray(raw, dtype=float)
    clean = np.where(np.isfinite(raw[train]), raw[train], np.nan)
    median = np.nan_to_num(np.nanmedian(clean, axis=0))
    raw = np.where(np.isfinite(raw), raw, median)
    scaler = RobustScaler(quantile_range=(10, 90)).fit(raw[train])
    x = np.clip(scaler.transform(raw), -20, 20).astype(np.float32)
    return x, {"kind": "training_fitted_RobustScaler", "quantile_range": [10, 90],
               "clip": [-20, 20], "median_imputation": median,
               "center": scaler.center_, "scale": scaler.scale_}


def composition(kind, y, parts):
    result = {}
    for partition, idx in parts.items():
        z = y[idx]
        row = {"n": len(z)}
        if kind == "binary":
            row.update({"negative_n": int((z == 0).sum()), "positive_n": int((z == 1).sum()),
                        "positive_fraction": float(z.mean())})
        else:
            row.update({"target_mean": z.mean(), "target_sd": z.std(ddof=1),
                        "target_median": np.median(z), "target_min": z.min(), "target_max": z.max()})
        result[partition] = row
    return result


def estimator(kind, model, seed):
    if model == "linear":
        if kind == "binary":
            return LogisticRegression(C=1, penalty="l2", solver="lbfgs", max_iter=2000,
                                      tol=1e-4, random_state=seed, class_weight=None)
        return Ridge(alpha=1, fit_intercept=True, solver="lsqr", tol=1e-8,
                     random_state=seed)
    cls = RandomForestClassifier if kind == "binary" else RandomForestRegressor
    return cls(n_estimators=200, max_features="sqrt", min_samples_leaf=2,
               max_depth=None, bootstrap=True, n_jobs=2, random_state=seed)


def target_transform(kind, family, y, train):
    """Return training targets, inverse scaler, and recorded target-processing rule."""
    if kind == "regression" and family == "linear":
        scaler = StandardScaler().fit(np.asarray(y[train], dtype=float).reshape(-1, 1))
        target = scaler.transform(np.asarray(y[train], dtype=float).reshape(-1, 1)).ravel()
        return target, scaler, {"kind": "training_fitted_StandardScaler", "mean": scaler.mean_,
                                "scale": scaler.scale_, "predictions": "inverse_transformed_to_original_units"}
    return y[train], None, {"kind": "unchanged", "units": "original_endpoint_units"}


def fit_prediction(kind, clf, x, training_target, train, test, target_scaler=None):
    """Fixed-fit helper shared with allocation experiments; does not save files."""
    clf.fit(x[train], training_target)
    if kind == "binary":
        return clf.predict_proba(x[test])[:, list(clf.classes_).index(1)]
    prediction = clf.predict(x[test])
    return target_scaler.inverse_transform(prediction.reshape(-1, 1)).ravel() if target_scaler is not None else prediction


def run(spec, data, features, regime, seed, representation, family):
    model = "constant_median" if family == "constant" else (
        "logistic" if family == "linear" and spec.kind == "binary" else
        "ridge" if family == "linear" else "rf")
    rid = f"{spec.name}__{regime}__test20__{representation}__{model}__seed{seed}__{RECIPE}"
    folder = OUT / spec.name / "runs" / rid
    split = OUT / spec.name / "splits" / f"{regime}_test20_seed{seed}.csv.gz"
    manifest = pd.read_csv(split)
    assert manifest.row_index.tolist() == data.row_index.tolist()
    assert manifest.connectivity_key.tolist() == data.connectivity_key.tolist()
    assert np.array_equal(manifest.y.to_numpy(), data.y.to_numpy())
    assert manifest.groupby("group").partition.nunique().max() == 1
    parts = {p: np.flatnonzero(manifest.partition.to_numpy() == p) for p in ("train", "val", "test")}
    train, test = parts["train"], parts["test"]
    y = data.y.to_numpy()
    clf = None if family == "constant" else estimator(spec.kind, family, seed)
    x, scaling = (None, {"training_median": float(np.median(y[train]))}) if clf is None else transform(features, representation, train)
    training_target, target_scaler, target_scaling = target_transform(spec.kind, family, y, train)
    config = {
        "run_id": rid, "endpoint": spec.name, "task_kind": spec.kind,
        "regime": regime, "test_target": 0.2, "validation_target": 0.1,
        "seed": seed, "allocation_seed": seed, "fitting_seed": seed,
        "representation": representation, "model": model, "recipe": RECIPE,
        "primary_metric": spec.primary_metric, "split_manifest": relative(split),
        "split_sha256": sha(split), "dataset_sha256": sha(OUT / spec.name / "dataset.csv.gz"),
        "feature_cache_sha256": sha(OUT / spec.name / "features.npz"),
        "script_sha256": sha(__file__), "protocol_sha256": sha(PROTOCOL),
        "metrics_source_sha256": sha(ROOT / "codes/scaffold/src/metrics.py"),
        "validation_used": False, "hyperparameter_selection": "fixed before fitting",
        "input_transformation": scaling, "target_transformation": target_scaling,
        "model_recipe": "fixed_random_forest" if family == "rf" else "common_linear" if family == "linear" else "training_median_control",
        "composition": composition(spec.kind, y, parts),
        "estimator": "training_label_median" if clf is None else type(clf).__name__,
        "training": {} if clf is None else clf.get_params(), "thread_limit": 2,
    }
    previous = folder / "summary.json"
    if previous.exists():
        if json.loads(previous.read_text())["status"] != "complete":
            raise ValueError(f"Previous incomplete summary: {rid}")
        stored = json.loads((folder / "config.json").read_text())
        if stored != clean_json(config):
            raise ValueError(f"Resume config/input/code mismatch: {rid}")
        return
    folder.mkdir(parents=True, exist_ok=True)
    dump(folder / "config.json", config)
    start = time.monotonic()
    try:
        if spec.kind == "binary" and len(np.unique(y[train])) < 2:
            raise ValueError("Single-class training partition")
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
            warnings.simplefilter("always")
            if clf is None:
                pred = np.full(len(test), np.median(y[train]))
            else:
                pred = fit_prediction(spec.kind, clf, x, training_target, train, test, target_scaler)
            metrics = evaluate(spec.kind, y[test], pred)
        assert np.isfinite(pred).all()
        result = data.iloc[test][["row_index", "connectivity_key", "smiles"]].copy()
        result["group"] = manifest.iloc[test].group.to_numpy()
        result["y_true"] = y[test]
        result["prediction"] = pred
        result.to_csv(folder / "predictions.csv.gz", index=False,
                      compression={"method": "gzip", "mtime": 0})
        if clf is not None:
            joblib.dump(clf, folder / "estimator.joblib", compress=3)
        baseline = float(np.abs(y[test] - np.median(y[train])).mean()) if spec.kind == "regression" else None
        legacy = None
        if spec.name == "DILI" and model == "logistic":
            oldrun = OLD / spec.name / "runs" / f"DILI__{regime}__test20__{representation}__logistic__seed{seed}__common" / "predictions.csv.gz"
            if oldrun.exists():
                oldpred = pd.read_csv(oldrun).set_index("row_index").loc[result.row_index].prediction.to_numpy()
                discrepancy = float(np.max(np.abs(pred - oldpred)))
                legacy = {"path": relative(oldrun), "sha256": sha(oldrun), "maximum_prediction_difference": discrepancy,
                          "passed_at_1e-6": discrepancy <= 1e-6}
                if discrepancy > 1e-6:
                    raise ValueError(f"Prior common-pipeline LR prediction mismatch: {discrepancy}")
        summary = {
            **{k: config[k] for k in ("run_id", "endpoint", "task_kind", "regime", "test_target", "seed", "representation", "model", "recipe", "model_recipe", "primary_metric")},
            "status": "complete", **metrics,
            "train_n": len(train), "val_n": len(parts["val"]), "test_n": len(test),
            "test_positive_fraction": float(y[test].mean()) if spec.kind == "binary" else None,
            "train_median_test_mae": baseline,
            "mae_skill_vs_train_median": 1 - metrics["mae"] / baseline if baseline and spec.kind == "regression" else None,
            "zero_baseline_mae": baseline == 0 if spec.kind == "regression" else None,
            "warnings": [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught],
            "n_iter": getattr(clf, "n_iter_", None), "seconds": time.monotonic() - start,
            "config_sha256": sha(folder / "config.json"),
            "predictions_sha256": sha(folder / "predictions.csv.gz"),
            "legacy_common_prediction_check": legacy,
        }
        dump(previous, summary)
        print(f"COMPLETE {rid} {spec.primary_metric}={metrics[spec.primary_metric]:.6f} seconds={summary['seconds']:.2f}", flush=True)
    except Exception as error:
        dump(folder / "failure.json", {"run_id": rid, "error": repr(error), "seconds": time.monotonic() - start})
        raise


def aggregate():
    records = [json.loads(p.read_text()) for p in sorted(OUT.glob("*/runs/*/summary.json")) if not p.name.startswith("._")]
    if records:
        pd.DataFrame(records).to_csv(OUT / "runs.csv", index=False)
    files = [p for p in OUT.rglob("*") if p.is_file() and not p.name.startswith("._") and p.name not in ("sha256_manifest.json", "sha256_manifest.json.tmp")]
    dump(OUT / "sha256_manifest.json", {relative(p): sha(p) for p in sorted(files)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["DILI", "BBB", "HIA", "PPBR", "CYP2D6", "Solubility"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    code_hash = sha(__file__)
    copy_verified(Path(__file__), OUT / "code_snapshots" / f"{code_hash}.py")
    copy_verified(PROTOCOL, OUT / "protocol.md")
    env = {"python": sys.version, "platform": platform.platform(),
           "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "rdkit", "joblib", "threadpoolctl")},
           "threads": 2, "command": sys.argv, "script_sha256": code_hash, "protocol_sha256": sha(PROTOCOL)}
    dump(OUT / "environment.json", env)
    for endpoint in args.tasks:
        spec = next(s for s in TASKS if s.name == endpoint)
        data, features = prepare(endpoint)
        for regime in REGIMES:
            for seed in SEEDS:
                if spec.kind == "regression":
                    run(spec, data, features, regime, seed, "constant", "constant")
                for representation in ("physchem", "maccs"):
                    run(spec, data, features, regime, seed, representation, "linear")
                run(spec, data, features, regime, seed, "morgan", "rf")
        aggregate()
        print(f"ENDPOINT COMPLETE {endpoint}", flush=True)
    aggregate()
    print("BASELINES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
