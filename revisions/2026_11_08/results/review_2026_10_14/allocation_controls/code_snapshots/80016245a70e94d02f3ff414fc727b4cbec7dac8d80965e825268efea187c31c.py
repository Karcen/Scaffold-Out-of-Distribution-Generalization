"""Label-blind exact-count group controls and reversed frontiers.

Reuse harmonised endpoint rows, grouping and geometry scores. Deterministic LR
and Ridge are fitted once per distinct partition. No labels enter allocation.
The protocol must exist before running; all completed runs are resumable.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
from pathlib import Path
import argparse
import concurrent.futures
import datetime
import hashlib
import importlib.util
import json
import platform
import shutil
import sys
import time
import warnings
import numpy as np
import pandas as pd
import scipy
import sklearn
import rdkit

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/revision/experiments"
OUT = ROOT / "results/review_2026_10_14/allocation_controls"
PROTOCOL = ROOT / "docs/review_2026_10_14/ALLOCATION_CONTROLS_PROTOCOL.md"
sys.path.insert(0, str(ROOT / "codes/scaffold"))
sys.path.insert(0, str(ROOT / "codes/repro"))
sys.path.insert(0, str(ROOT / "codes/review_2026_10_14"))
from src.data import TASKS
from src.metrics import evaluate
from grid_common import physchem

# Resume from the helper actually used by the original experiment, even if the
# baseline runner subsequently gains unrelated CLI options. The snapshot's
# module-root globals are corrected because it lives under results/.
HELPER_SNAPSHOT = OUT / "baseline_helper_snapshot.py"
LOADED_HELPER_PATH = HELPER_SNAPSHOT if HELPER_SNAPSHOT.exists() else ROOT / "codes/review_2026_10_14/baselines.py"
HELPER_SHA256 = hashlib.sha256(LOADED_HELPER_PATH.read_bytes()).hexdigest()
_helper_spec = importlib.util.spec_from_file_location("allocation_frozen_baselines", LOADED_HELPER_PATH)
_helper_module = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_helper_module)
_helper_module.ROOT = ROOT
common_transform = _helper_module.transform
estimator = _helper_module.estimator
target_transform = _helper_module.target_transform

ENDPOINTS = ("BBB", "HIA", "CYP2D6", "PPBR", "Solubility", "DILI")
REPRESENTATIONS = ("physchem", "morgan")
GEOMETRIES = ("descriptor", "morgan")
RANDOM_BASE_SEED = 2026101400


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True,
                               default=lambda x: x.tolist() if isinstance(x, np.ndarray)
                               else x.item() if hasattr(x, "item") else str(x)) + "\n")


def relative(path):
    return str(Path(path).relative_to(ROOT))


def assignment_sha(assignment):
    return hashlib.sha256("\n".join(assignment).encode()).hexdigest()


def exact_subset(group_ids, sizes, target, rng):
    """Random priority, first-reachable prefix, exact bitset subset-sum.

    This is not a uniform distribution over subsets. Backtracking skips later
    priority groups when earlier groups alone can satisfy the remaining sum.
    No label or chemistry coordinate enters this constrained selection.
    """
    if target == 0:
        return []
    order = rng.permutation(np.asarray(group_ids, dtype=np.int64))
    bits = 1
    states = [bits]
    used = []
    limit = (1 << (target + 1)) - 1
    for group_id in order:
        weight = int(sizes[group_id])
        if weight > target:
            continue
        bits = (bits | (bits << weight)) & limit
        used.append(int(group_id))
        states.append(bits)
        if (bits >> target) & 1:
            break
    if not ((bits >> target) & 1):
        raise ValueError(f"No exact subset sum {target} in remaining groups")
    remaining = target
    chosen = []
    for j in range(len(used) - 1, -1, -1):
        if (states[j] >> remaining) & 1:
            continue
        group_id = used[j]
        chosen.append(group_id)
        remaining -= int(sizes[group_id])
    assert remaining == 0
    assert sum(int(sizes[g]) for g in chosen) == target
    return chosen


def exact_assignment(group_to_rows, sizes, n, test_target, val_target, seed,
                     max_attempts=500):
    rng = np.random.default_rng(seed)
    ids = np.arange(len(sizes), dtype=np.int64)
    failure_messages = []
    for attempt in range(1, max_attempts + 1):
        try:
            test = exact_subset(ids, sizes, test_target, rng)
            test_set = set(test)
            available = [g for g in ids if g not in test_set]
            val = exact_subset(available, sizes, val_target, rng)
            assignment = np.full(n, "train", dtype="<U5")
            for g in test:
                assignment[group_to_rows[g]] = "test"
            for g in val:
                assignment[group_to_rows[g]] = "val"
            assert int(np.sum(assignment == "test")) == test_target
            assert int(np.sum(assignment == "val")) == val_target
            return assignment, {"attempts": attempt, "sampler": "random_priority_exact_subset_sum",
                                "uniform_over_feasible_partitions": False,
                                "size_matching": "exact_test_and_validation_molecule_counts",
                                "label_blind": True, "failed_attempts": failure_messages}
        except ValueError as exc:
            failure_messages.append(str(exc))
    raise RuntimeError(f"Exact joint allocation not found after {max_attempts} label-blind attempts; "
                       "reference counts are feasible, but this sequential sampler failed.")


def reverse_assignment(group_to_rows, scores, n, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(scores + rng.uniform(0, 1e-9, len(scores)), kind="mergesort")
    assignment = np.full(n, "train", dtype="<U5")
    counts = {"test": 0, "val": 0}
    current = "test"
    for g in order:
        if current == "test" and counts["test"] >= 0.2 * n:
            current = "val"
        if current == "val" and counts["val"] >= 0.1 * n:
            current = "train"
        rows = group_to_rows[g]
        assignment[rows] = current
        if current in counts:
            counts[current] += len(rows)
    return assignment, {"sampler": "ascending_frontier_score_whole_groups",
                        "allocation_seed": seed, "score_jitter": 1e-9,
                        "targets": {"test": 0.2, "val": 0.1, "train": 0.7},
                        "size_matching": "nominal_fractions_whole_group_overshoot_allowed",
                        "label_blind": True}


def prepare_endpoint(name):
    df = pd.read_csv(BASE / name / "dataset.csv.gz")
    with np.load(BASE / name / "features.npz", allow_pickle=False) as cache:
        groups = cache["groups"].astype(str)
        arrays = {"morgan": cache["morgan"]}
        arrays["physchem"] = cache["physchem"] if "physchem" in cache.files else None
    assert len(groups) == len(df) == len(arrays["morgan"])
    if arrays["physchem"] is None:
        arrays["physchem"] = np.stack([physchem(smiles) for smiles in df.smiles])
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    feature_path = folder / "physchem_features.npz"
    if not feature_path.exists():
        np.savez_compressed(feature_path, physchem=arrays["physchem"])
    else:
        with np.load(feature_path) as saved:
            assert np.allclose(saved["physchem"], arrays["physchem"], equal_nan=True)
    keys = list(dict.fromkeys(groups))
    rows = [np.flatnonzero(groups == group) for group in keys]
    sizes = np.array([len(r) for r in rows], dtype=int)
    input_config = {
        "endpoint": name, "n_molecules": len(df), "n_groups": len(keys),
        "dataset": relative(BASE / name / "dataset.csv.gz"),
        "dataset_sha256": sha(BASE / name / "dataset.csv.gz"),
        "features": relative(BASE / name / "features.npz"),
        "features_sha256": sha(BASE / name / "features.npz"),
        "derived_physchem": relative(feature_path), "derived_physchem_sha256": sha(feature_path),
        "physchem_function_file": "codes/repro/grid_common.py",
        "physchem_function_file_sha256": sha(ROOT / "codes/repro/grid_common.py"),
        "raw_feature_definition": "existing radius2 1024-bit Morgan; nine existing physchem descriptors",
        "groups": "unchanged cached groups; original descriptor-derived acyclic cells retained",
        "script_sha256": sha(__file__), "protocol_sha256": sha(PROTOCOL),
    }
    input_config_path = folder / "input_config.json"
    if input_config_path.exists():
        old = json.loads(input_config_path.read_text())
        for field in ["dataset_sha256", "features_sha256", "derived_physchem_sha256"]:
            assert old[field] == input_config[field], "Inputs changed on resume"
    else:
        dump(input_config_path, input_config)
    return df, arrays, groups, keys, rows, sizes


def save_split(name, df, groups, assignment, scores, split_id, metadata):
    folder = OUT / name / "splits"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / f"{split_id}.csv.gz"
    assert set(assignment) <= {"train", "val", "test"}
    check = pd.DataFrame({"group": groups, "partition": assignment}).groupby("group").partition.nunique()
    assert int(check.max()) == 1
    table = df[["row_index", "connectivity_key", "y"]].copy()
    table["group"] = groups
    table["partition"] = assignment
    table["frontier_score"] = scores
    if not manifest.exists():
        table.to_csv(manifest, index=False, compression={"method": "gzip", "mtime": 0})
    else:
        saved = pd.read_csv(manifest)
        assert saved.partition.tolist() == assignment.tolist()
        assert saved.connectivity_key.tolist() == df.connectivity_key.tolist()
    split_metadata = {
        "split_id": split_id, **metadata,
        "counts": {p: int(np.sum(assignment == p)) for p in ["train", "val", "test"]},
        "assignment_sha256": assignment_sha(assignment),
        "manifest_sha256": sha(manifest), "script_sha256": sha(__file__),
        "protocol_sha256": sha(PROTOCOL),
    }
    metadata_path = folder / f"{split_id}.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        assert old["assignment_sha256"] == split_metadata["assignment_sha256"]
        assert old["manifest_sha256"] == split_metadata["manifest_sha256"]
    else:
        dump(metadata_path, split_metadata)
    return manifest


def transform_inputs(name, representation, x, train):
    return common_transform({representation: x}, representation, train)


def fit_run(name, df, arrays, manifest, split_id, mode, geometry, allocation_seed,
            representation, canonical_run=None):
    spec = next(t for t in TASKS if t.name == name)
    model_kind = "logistic" if spec.kind == "binary" else "ridge"
    run_id = f"{split_id}__{representation}__{model_kind}"
    folder = OUT / name / "runs" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    summary_file = folder / "summary.json"
    if summary_file.exists():
        old = json.loads(summary_file.read_text())
        old_hash = old["script_sha256"]
        recorded_source = OUT / "code_snapshots" / f"{old_hash}.py"
        legacy_snapshot = OUT / "code_snapshot.py"
        assert ((recorded_source.exists() and sha(recorded_source) == old_hash)
                or (legacy_snapshot.exists() and sha(legacy_snapshot) == old_hash)), "Missing executed code snapshot"
        old_config = json.loads((folder / "config.json").read_text())
        assert old_config["dependency_hashes"]["codes/review_2026_10_14/baselines.py"] == HELPER_SHA256
        assert old_config["split_manifest_sha256"] == sha(manifest)
        assert old_config["dataset_sha256"] == sha(BASE / name / "dataset.csv.gz")
        return old
    man = pd.read_csv(manifest)
    assignment = man.partition.to_numpy()
    train = np.flatnonzero(assignment == "train")
    test = np.flatnonzero(assignment == "test")
    y = df.y.to_numpy()
    config = {
        "run_id": run_id, "endpoint": name, "mode": mode, "geometry": geometry,
        "allocation_seed": allocation_seed, "fitting_seed": 0,
        "deterministic_fit": True, "representation": representation, "model": model_kind,
        "primary_metric": spec.primary_metric, "task_kind": spec.kind,
        "dataset_sha256": sha(BASE / name / "dataset.csv.gz"),
        "split_manifest": relative(manifest), "split_manifest_sha256": sha(manifest),
        "script_sha256": sha(__file__), "protocol_sha256": sha(PROTOCOL),
        "dependency_hashes": {
            "codes/review_2026_10_14/baselines.py": HELPER_SHA256,
            "codes/repro/grid_common.py": sha(ROOT / "codes/repro/grid_common.py"),
            "codes/scaffold/src/metrics.py": sha(ROOT / "codes/scaffold/src/metrics.py")},
        "validation_use": "allocation counts only; no hyperparameter tuning or early stopping",
        "model_settings": {"C": 1.0, "solver": "lbfgs", "max_iter": 2000,
                           "class_weight": None, "random_state": 0, "penalty": "l2"}
                          if spec.kind == "binary" else
                          {"alpha": 1.0, "solver": "lsqr", "tol": 1e-8,
                           "max_iter": None, "fit_intercept": True},
    }
    start = time.time()
    if canonical_run is not None:
        source_folder = OUT / name / "runs" / canonical_run
        source = json.loads((source_folder / "summary.json").read_text())
        config["reuses_deterministic_run"] = relative(source_folder)
        dump(folder / "config.json", config)
        if source.get("status") == "complete":
            shutil.copy2(source_folder / "predictions.csv.gz", folder / "predictions.csv.gz")
        summary = {**source, **{k: config[k] for k in ["run_id", "mode", "geometry", "allocation_seed"]},
                   "fitted_again": False, "canonical_run": canonical_run,
                   "config_sha256": sha(folder / "config.json")}
        dump(summary_file, summary)
        return summary
    warnings_seen = []
    try:
        if spec.kind == "binary" and len(np.unique(y[train])) < 2:
            raise ValueError("Single-class training set; label-blind allocation is retained and fit fails.")
        x, transformation = transform_inputs(name, representation, arrays[representation], train)
        config["input_transformation"] = transformation
        training_y, inverse_scaler, target_metadata = target_transform(spec.kind, "linear", y, train)
        config["target_transformation"] = target_metadata
        dump(folder / "config.json", config)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = estimator(spec.kind, "linear", 0)
            model.fit(x[train], training_y)
            if spec.kind == "binary":
                prediction = model.predict_proba(x[test])[:, 1]
            else:
                prediction = model.predict(x[test])
                if inverse_scaler is not None:
                    prediction = inverse_scaler.inverse_transform(prediction.reshape(-1, 1)).ravel()
            warnings_seen = [str(w.message) for w in caught]
        metrics = evaluate(spec.kind, y[test], prediction)
        output = df.iloc[test][["row_index", "connectivity_key"]].copy()
        output["y_true"] = y[test]
        output["prediction"] = prediction
        output.to_csv(folder / "predictions.csv.gz", index=False,
                      compression={"method": "gzip", "mtime": 0})
        np.savez_compressed(folder / "fitted_linear_parameters.npz",
                            coefficient=model.coef_, intercept=np.asarray(model.intercept_))
        summary = {**{k: config[k] for k in ["run_id", "endpoint", "mode", "geometry", "allocation_seed",
                                           "representation", "model", "primary_metric"]},
                   "status": "complete", **metrics, "fitted_again": True,
                   "train_n": len(train), "val_n": int(np.sum(assignment == "val")),
                   "test_n": len(test), "n_iter": np.asarray(getattr(model, "n_iter_", None)).tolist(),
                   "test_positive_fraction": float(np.mean(y[test])) if spec.kind == "binary" else None,
                   "training_positive_fraction": float(np.mean(y[train])) if spec.kind == "binary" else None,
                   "warnings": warnings_seen, "seconds": time.time() - start,
                   "script_sha256": sha(__file__), "config_sha256": sha(folder / "config.json"),
                   "prediction_sha256": sha(folder / "predictions.csv.gz")}
    except Exception as exc:
        dump(folder / "config.json", config)
        summary = {**{k: config[k] for k in ["run_id", "endpoint", "mode", "geometry", "allocation_seed",
                                           "representation", "model", "primary_metric"]},
                   "status": "failed", "error": repr(exc), "fitted_again": True,
                   "script_sha256": sha(__file__), "config_sha256": sha(folder / "config.json"),
                   "seconds": time.time() - start}
    dump(summary_file, summary)
    return summary


def run_endpoint(name, draws, reverse_seeds, splits_only):
    df, arrays, groups, keys, rows, sizes = prepare_endpoint(name)
    all_summaries = []
    split_records = []
    fitted_partitions = {}
    endpoint_index = ENDPOINTS.index(name)
    for geometry_index, geometry in enumerate(GEOMETRIES):
        score_file = BASE / name / f"{geometry}_group_scores.csv"
        score_table = pd.read_csv(score_file)
        score_map = score_table.set_index("group").score.to_dict()
        score = np.asarray([score_map[g] for g in keys])
        row_score = np.asarray([score_map[g] for g in groups])
        reference = BASE / name / "splits" / f"{geometry}_test20_seed0.csv.gz"
        ref = pd.read_csv(reference)
        assert ref.connectivity_key.tolist() == df.connectivity_key.tolist()
        assert ref.group.tolist() == groups.tolist()
        assert np.allclose(ref.frontier_score, row_score)
        target_test = int(np.sum(ref.partition == "test"))
        target_val = int(np.sum(ref.partition == "val"))
        cases = [("frontier_reference", 0, ref.partition.to_numpy(),
                  {"sampler": "original_high_frontier_seed0", "reference_manifest": relative(reference)})]
        for draw in range(draws):
            allocation_seed = RANDOM_BASE_SEED + endpoint_index * 10000 + geometry_index * 1000 + draw
            try:
                assignment, meta = exact_assignment(rows, sizes, len(df), target_test, target_val, allocation_seed)
                cases.append(("exact_random", allocation_seed, assignment, meta))
            except RuntimeError as exc:
                failure = {"endpoint": name, "geometry": geometry, "draw": draw,
                           "allocation_seed": allocation_seed, "status": "allocation_not_found",
                           "reference_test_count": target_test, "reference_val_count": target_val,
                           "error": str(exc)}
                dump(OUT / name / "allocation_failures" / f"{geometry}_{draw:02d}.json", failure)
                split_records.append(failure)
        for seed in range(reverse_seeds):
            assignment, meta = reverse_assignment(rows, score, len(df), seed)
            cases.append(("reverse_frontier", seed, assignment, meta))
        for case_index, (mode, allocation_seed, assignment, meta) in enumerate(cases):
            split_id = f"{name}__{geometry}__{mode}__alloc{allocation_seed}"
            metadata = {**meta, "endpoint": name, "geometry": geometry, "mode": mode,
                        "allocation_seed": allocation_seed,
                        "reference_manifest": relative(reference), "reference_sha256": sha(reference),
                        "reference_test_count": target_test, "reference_val_count": target_val,
                        "group_scores_file": relative(score_file), "group_scores_sha256": sha(score_file),
                        "no_label_stratification": True}
            manifest = save_split(name, df, groups, assignment, row_score, split_id, metadata)
            split_records.append({"split_id": split_id, **metadata,
                                  "test_n": int(np.sum(assignment == "test")),
                                  "val_n": int(np.sum(assignment == "val")),
                                  "assignment_sha256": assignment_sha(assignment)})
            if not splits_only:
                for representation in REPRESENTATIONS:
                    partition_key = (assignment_sha(assignment), representation)
                    canonical = fitted_partitions.get(partition_key)
                    summary = fit_run(name, df, arrays, manifest, split_id, mode,
                                      geometry, allocation_seed, representation, canonical)
                    fitted_partitions.setdefault(partition_key, summary["run_id"])
                    all_summaries.append(summary)
            if case_index % 10 == 0 or case_index == len(cases) - 1:
                print(f"{name} {geometry}: {case_index + 1}/{len(cases)} partitions saved"
                      + (" and scored" if not splits_only else ""), flush=True)
    pd.DataFrame(split_records).to_csv(OUT / name / "allocation_manifest.csv", index=False)
    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(OUT / name / "runs.csv", index=False)
    return {"endpoint": name, "split_records": len(split_records), "run_records": len(all_summaries),
            "completed": sum(r["status"] == "complete" for r in all_summaries),
            "failed": sum(r["status"] != "complete" for r in all_summaries)}


def manifest():
    records = [{"path": relative(p), "sha256": sha(p), "bytes": p.stat().st_size}
               for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "sha256_manifest.json"
               and not p.name.endswith(".log") and not any(part.startswith(".") for part in p.relative_to(OUT).parts)]
    dump(OUT / "sha256_manifest.json", {"algorithm": "SHA-256", "files": records,
                                        "exclusions": "volatile .log files and hidden operating-system metadata"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", nargs="+", default=list(ENDPOINTS), choices=ENDPOINTS)
    parser.add_argument("--draws", type=int, default=50)
    parser.add_argument("--reverse-seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--splits-only", action="store_true")
    args = parser.parse_args()
    assert PROTOCOL.exists(), "Write and freeze protocol before run"
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_code = OUT / "code_snapshots" / f"{sha(__file__)}.py"
    frozen_code.parent.mkdir(parents=True, exist_ok=True)
    if not frozen_code.exists():
        shutil.copy2(__file__, frozen_code)
    assert sha(frozen_code) == sha(__file__)
    if not HELPER_SNAPSHOT.exists():
        shutil.copy2(LOADED_HELPER_PATH, HELPER_SNAPSHOT)
    assert sha(HELPER_SNAPSHOT) == HELPER_SHA256
    protocol_snapshot = OUT / "protocol_snapshot.md"
    if protocol_snapshot.exists():
        assert sha(protocol_snapshot) == sha(PROTOCOL), "Frozen allocation protocol changed"
    else:
        shutil.copy2(PROTOCOL, protocol_snapshot)
    environment = {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "rdkit": rdkit.__version__, "executable": sys.executable,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "script_sha256": sha(__file__), "protocol_sha256": sha(PROTOCOL),
        "common_helper_sha256": HELPER_SHA256, "common_helper_loaded_from": relative(LOADED_HELPER_PATH),
        "arguments": vars(args), "random_base_seed": RANDOM_BASE_SEED,
        "reverse_allocation_seeds": list(range(args.reverse_seeds)),
        "fitting_seed": 0, "deterministic_fit_once_per_distinct_partition": True,
    }
    session_tag = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    dump(OUT / "sessions" / f"environment_{session_tag}.json", environment)
    if not (OUT / "environment.json").exists():
        dump(OUT / "environment.json", environment)
    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_endpoint, name, args.draws, args.reverse_seeds, args.splits_only)
                       for name in args.endpoints]
            summaries = [future.result() for future in futures]
    else:
        summaries = [run_endpoint(name, args.draws, args.reverse_seeds, args.splits_only)
                     for name in args.endpoints]
    completion = {"endpoint_summaries": summaries, "arguments": vars(args),
                  "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    dump(OUT / "sessions" / f"completion_{session_tag}.json", completion)
    if not (OUT / "completion.json").exists():
        dump(OUT / "completion.json", completion)
    run_paths = sorted(OUT.glob("*/runs.csv"))
    if run_paths:
        pd.concat([pd.read_csv(p) for p in run_paths], ignore_index=True).to_csv(OUT / "runs.csv", index=False)
    manifest()
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
