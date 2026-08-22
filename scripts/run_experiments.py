#!/usr/bin/env python3
"""Resumable, multi-seed scaffold/OOD experiment runner."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

if "." not in sys.path:
    sys.path.insert(0, ".")

from src.chem import featurize_many
from src.data import TASKS, TaskSpec, load_tdc_task
from src.environments import VIEW_NAMES, build_novelty_environments
from src.models import TrainConfig, train_model
from src.objectives import ObjectiveConfig
from src.splits import assert_split_integrity, frontier_split, scaffold_split


RESULTS = Path("results")
CACHE = Path("cache")
RUNS = RESULTS / "runs"
SPLITS = RESULTS / "splits"

MAIN_METHODS = {
    "ERM": ObjectiveConfig(method="erm"),
    "V-REx": ObjectiveConfig(method="vrex"),
    "GroupDRO": ObjectiveConfig(method="groupdro"),
    "CVaR": ObjectiveConfig(method="cvar"),
    "MV-FREX": ObjectiveConfig(method="mvfrex"),
}
ABLATIONS = {
    "MV-FREX-no-slope": ObjectiveConfig(method="mvfrex_no_slope"),
    "FREX-single-view": ObjectiveConfig(method="frex_single_view"),
}


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def build_feature_cache(spec: TaskSpec) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{spec.name.lower()}_features.npz"
    frame = load_tdc_task(spec, Path("data/external"))
    if path.exists():
        try:
            loaded = np.load(path, allow_pickle=False)
            cached_smiles = loaded["smiles"].astype(str)
            if np.array_equal(cached_smiles, frame["smiles"].astype(str).to_numpy()):
                return frame, {view: loaded[view] for view in VIEW_NAMES}
        except (ValueError, OSError, KeyError):
            print(f"[{spec.name}] rebuilding stale feature cache", flush=True)
    print(f"[{spec.name}] featurizing {len(frame):,} molecules", flush=True)
    views = {
        "ecfp": featurize_many(frame["smiles"], "ecfp", n_bits=1024),
        "maccs": featurize_many(frame["smiles"], "maccs"),
        "atompair": featurize_many(frame["smiles"], "atompair", n_bits=1024),
        "physchem": featurize_many(frame["smiles"], "physchem"),
    }
    np.savez_compressed(
        path, smiles=np.asarray(frame["smiles"].astype(str).tolist(), dtype=str), **views
    )
    return frame, views


def split_for(kind: str, smiles: np.ndarray, seed: int):
    if kind == "scaffold":
        return scaffold_split(smiles, seed=seed, fractions=(0.8, 0.1, 0.1))
    if kind == "scaffold_matched":
        return scaffold_split(smiles, seed=seed, fractions=(0.7, 0.1, 0.2))
    if kind == "scaffold_clustered_matched":
        return scaffold_split(
            smiles,
            seed=seed,
            fractions=(0.7, 0.1, 0.2),
            cluster_acyclic=True,
        )
    if kind == "frontier":
        return frontier_split(smiles, seed=seed, fractions=(0.7, 0.1, 0.2))
    raise ValueError(kind)


def split_distribution(y: np.ndarray, assignment: np.ndarray, kind: str) -> dict:
    out = {}
    for partition in ["train", "val", "test"]:
        values = y[assignment == partition]
        if kind == "binary":
            out[partition] = {
                "n": len(values),
                "positive_n": int(values.sum()),
                "positive_fraction": float(values.mean()),
            }
        else:
            out[partition] = {
                "n": len(values),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
    return out


def save_split_manifest(
    spec: TaskSpec,
    frame: pd.DataFrame,
    split_kind: str,
    seed: int,
    result,
    environments,
) -> Path:
    SPLITS.mkdir(parents=True, exist_ok=True)
    path = SPLITS / f"{spec.name}__{split_kind}__seed{seed}.csv.gz"
    manifest = pd.DataFrame(
        {
            "row_index": np.arange(len(frame)),
            "source_id": frame["source_id"],
            "smiles": frame["smiles"],
            "connectivity_key": frame["connectivity_key"],
            "scaffold": result.scaffold,
            "partition": result.assignment,
            "frontier_score": result.frontier_score,
        }
    )
    for view_index, view in enumerate(environments.view_names):
        manifest[f"novelty_{view}"] = environments.coordinates[view_index]
        manifest[f"environment_{view}"] = environments.groups[view_index]
    manifest.to_csv(path, index=False, compression="gzip")
    return path


def run_one(
    spec: TaskSpec,
    frame: pd.DataFrame,
    feature_views: dict[str, np.ndarray],
    split_kind: str,
    seed: int,
    method_name: str,
    objective: ObjectiveConfig,
    architecture: str,
    split_result,
    environments,
    train_config: TrainConfig,
    force: bool,
) -> None:
    run_id = f"{spec.name}__{split_kind}__seed{seed}__{method_name}__{architecture}".replace("/", "-")
    run_dir = RUNS / run_id
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not force:
        try:
            if json.loads(summary_path.read_text())["status"] == "complete":
                print(f"skip {run_id}", flush=True)
                return
        except Exception:
            pass
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    y = frame["y"].to_numpy(dtype=np.float32)
    assignment = split_result.assignment
    train_idx = np.where(assignment == "train")[0]
    val_idx = np.where(assignment == "val")[0]
    test_idx = np.where(assignment == "test")[0]
    local_config = replace(train_config, seed=seed, hidden_dim=0 if architecture == "linear" else train_config.hidden_dim)
    config_payload = {
        "run_id": run_id,
        "task": asdict(spec),
        "split": split_result.metadata,
        "seed": seed,
        "method": method_name,
        "architecture": architecture,
        "objective": asdict(objective),
        "training": asdict(local_config),
        "views": list(environments.view_names),
        "environment_thresholds": environments.thresholds,
        "environment_centers": environments.group_centers,
        "environment_counts": environments.train_group_counts,
        "label_distribution": split_distribution(y, assignment, spec.kind),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, default=json_default), encoding="utf-8"
    )
    try:
        result = train_model(
            x=feature_views["ecfp"].astype(np.float32),
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            kind=spec.kind,
            primary_metric=spec.primary_metric,
            view_groups=environments.groups,
            group_centers=environments.group_centers,
            train_group_counts=environments.train_group_counts,
            objective_config=objective,
            train_config=local_config,
        )
        torch.save(
            {
                "state_dict": result.state_dict,
                "input_dim": feature_views["ecfp"].shape[1],
                "train_config": asdict(local_config),
                "target_mean": result.target_mean,
                "target_scale": result.target_scale,
                "kind": spec.kind,
            },
            run_dir / "model.pt",
        )
        pd.DataFrame(result.history).to_csv(run_dir / "history.csv", index=False)
        prediction_rows = []
        for partition, indices, prediction in [
            ("val", val_idx, result.val_prediction),
            ("test", test_idx, result.test_prediction),
        ]:
            block = pd.DataFrame(
                {
                    "row_index": indices,
                    "source_id": frame.iloc[indices]["source_id"].to_numpy(),
                    "smiles": frame.iloc[indices]["smiles"].to_numpy(),
                    "partition": partition,
                    "y_true": y[indices],
                    "prediction": prediction,
                    "frontier_score": split_result.frontier_score[indices],
                }
            )
            for view_index, view in enumerate(environments.view_names):
                block[f"novelty_{view}"] = environments.coordinates[view_index, indices]
                block[f"environment_{view}"] = environments.groups[view_index, indices]
            prediction_rows.append(block)
        pd.concat(prediction_rows, ignore_index=True).to_csv(
            run_dir / "predictions.csv.gz", index=False, compression="gzip"
        )
        summary = {
            "status": "complete",
            "run_id": run_id,
            "task": spec.name,
            "tdc_name": spec.tdc_task,
            "kind": spec.kind,
            "primary_metric": spec.primary_metric,
            "split": split_kind,
            "seed": seed,
            "method": method_name,
            "architecture": architecture,
            "best_epoch": result.best_epoch,
            "train_n": len(train_idx),
            "val_n": len(val_idx),
            "test_n": len(test_idx),
            "val_metrics": result.val_metrics,
            "test_metrics": result.test_metrics,
            "elapsed_seconds": time.time() - started,
        }
        summary_path.write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
        print(
            f"done {run_id} {spec.primary_metric}={result.test_metrics.get(spec.primary_metric, float('nan')):.4f} "
            f"epoch={result.best_epoch} time={summary['elapsed_seconds']:.1f}s",
            flush=True,
        )
    except Exception as error:
        failure = {
            "status": "failed",
            "run_id": run_id,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
        summary_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(f"FAILED {run_id}: {error}", file=sys.stderr, flush=True)
        raise


def rebuild_metrics() -> pd.DataFrame:
    rows = []
    for path in sorted(RUNS.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        if summary.get("status") != "complete":
            continue
        for partition in ["val", "test"]:
            for metric, value in summary[f"{partition}_metrics"].items():
                rows.append(
                    {
                        "run_id": summary["run_id"],
                        "task": summary["task"],
                        "tdc_name": summary["tdc_name"],
                        "kind": summary["kind"],
                        "primary_metric": summary["primary_metric"],
                        "split": summary["split"],
                        "seed": summary["seed"],
                        "method": summary["method"],
                        "architecture": summary["architecture"],
                        "partition": partition,
                        "metric": metric,
                        "value": value,
                        "best_epoch": summary["best_epoch"],
                        "train_n": summary["train_n"],
                        "val_n": summary["val_n"],
                        "test_n": summary["test_n"],
                        "elapsed_seconds": summary["elapsed_seconds"],
                    }
                )
    metrics = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=[spec.name for spec in TASKS])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--splits", nargs="*", default=["scaffold", "frontier"])
    parser.add_argument("--methods", nargs="*", default=list(MAIN_METHODS))
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--include-linear-control", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    train_config = TrainConfig()
    if args.quick:
        train_config = replace(train_config, max_epochs=4, patience=2, batch_size=2048)
    task_map = {spec.name: spec for spec in TASKS}
    methods = {name: MAIN_METHODS[name] for name in args.methods}
    for task_name in args.tasks:
        spec = task_map[task_name]
        frame, feature_views = build_feature_cache(spec)
        smiles = frame["smiles"].astype(str).to_numpy()
        for split_kind in args.splits:
            for seed in args.seeds:
                split_result = split_for(split_kind, smiles, seed)
                assert_split_integrity(split_result)
                train_idx = np.where(split_result.assignment == "train")[0]
                environments = build_novelty_environments(feature_views, train_idx, seed=seed)
                save_split_manifest(spec, frame, split_kind, seed, split_result, environments)
                run_methods = dict(methods)
                if split_kind == "frontier" and args.include_ablations and seed < 3:
                    run_methods.update(ABLATIONS)
                for method_name, objective in run_methods.items():
                    run_one(
                        spec,
                        frame,
                        feature_views,
                        split_kind,
                        seed,
                        method_name,
                        objective,
                        "mlp",
                        split_result,
                        environments,
                        train_config,
                        args.force,
                    )
                if split_kind in {"frontier", "scaffold_matched", "scaffold_clustered_matched"} and args.include_linear_control and seed < 3:
                    control_methods = ["ERM", "MV-FREX"] if split_kind == "frontier" else ["ERM"]
                    for method_name in control_methods:
                        run_one(
                            spec,
                            frame,
                            feature_views,
                            split_kind,
                            seed,
                            method_name,
                            MAIN_METHODS[method_name],
                            "linear",
                            split_result,
                            environments,
                            train_config,
                            args.force,
                        )
                rebuild_metrics()
    metrics = rebuild_metrics()
    print(f"wrote {RESULTS / 'metrics.csv'} with {len(metrics):,} metric rows", flush=True)


if __name__ == "__main__":
    main()
