#!/usr/bin/env python3
"""Run the major-revision message-passing capacity control.

The control mirrors the strict 70/10/20 matched scaffold and structural-frontier
allocations.  ERM is fitted on both; the falsified MV-FREX probe is fitted on the
frontier.  Outputs are isolated under ``results/gnn_runs`` so the original 468-run
corpus remains immutable and auditable.
"""

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
# PyG is an optional review-control dependency installed outside the core
# environment.  Appending (rather than prepending) keeps the pinned core NumPy.
deps = Path(".deps")
if deps.is_dir() and str(deps) not in sys.path:
    sys.path.append(str(deps))

from scripts.run_experiments import (  # noqa: E402
    build_feature_cache,
    json_default,
    split_distribution,
    split_for,
)
from src.data import TASKS, TaskSpec  # noqa: E402
from src.environments import build_novelty_environments  # noqa: E402
from src.graph_models import (  # noqa: E402
    GraphTrainConfig,
    build_graphs,
    config_dict,
    train_graph_model,
)
from src.objectives import ObjectiveConfig  # noqa: E402
from src.splits import assert_split_integrity  # noqa: E402


RESULTS = Path("results")
RUNS = RESULTS / "gnn_runs"
CACHE = Path("cache")
METHODS = {
    "ERM": ObjectiveConfig(method="erm"),
    "MV-FREX": ObjectiveConfig(method="mvfrex"),
}
HIGHER_IS_BETTER = {"auroc", "auprc", "balanced_accuracy", "f1", "mcc", "spearman", "r2"}


def _graph_cache(spec: TaskSpec, frame: pd.DataFrame, y: np.ndarray) -> list:
    path = CACHE / f"{spec.name.lower()}_pyg_graphs.pt"
    smiles = frame["smiles"].astype(str).to_numpy()
    if path.exists():
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if np.array_equal(np.asarray(payload["smiles"], dtype=str), smiles):
                return payload["graphs"]
        except (OSError, RuntimeError, KeyError, ValueError):
            print(f"[{spec.name}] rebuilding stale graph cache", flush=True)
    print(f"[{spec.name}] parsing {len(frame):,} molecular graphs", flush=True)
    placeholder_groups = np.zeros((4, len(frame)), dtype=np.int64)
    graphs = build_graphs(smiles, y, placeholder_groups)
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({"smiles": smiles.tolist(), "graphs": graphs}, path)
    return graphs


def _set_environment_groups(graphs: list, groups: np.ndarray) -> None:
    for index, graph in enumerate(graphs):
        graph.env_groups = torch.tensor(groups[:, index][None, :], dtype=torch.long)


def run_one(
    spec: TaskSpec,
    frame: pd.DataFrame,
    graphs: list,
    split_kind: str,
    seed: int,
    method_name: str,
    objective: ObjectiveConfig,
    split_result,
    environments,
    train_config: GraphTrainConfig,
    force: bool,
) -> None:
    run_id = f"{spec.name}__{split_kind}__seed{seed}__{method_name}__gine3x128"
    run_dir = RUNS / run_id
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not force:
        try:
            if json.loads(summary_path.read_text(encoding="utf-8"))["status"] == "complete":
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
    local_config = replace(train_config, seed=seed)
    config_payload = {
        "run_id": run_id,
        "task": asdict(spec),
        "split": split_result.metadata,
        "seed": seed,
        "method": method_name,
        "architecture": "gine3x128",
        "objective": asdict(objective),
        "training": config_dict(local_config),
        "views": list(environments.view_names),
        "environment_thresholds": environments.thresholds,
        "environment_centers": environments.group_centers,
        "environment_counts": environments.train_group_counts,
        "label_distribution": split_distribution(y, assignment, spec.kind),
        "representation": "learned categorical molecular graph; no pretrained weights",
    }
    (run_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, default=json_default), encoding="utf-8"
    )

    try:
        result = train_graph_model(
            graphs=graphs,
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            kind=spec.kind,
            primary_metric=spec.primary_metric,
            group_centers=environments.group_centers,
            train_group_counts=environments.train_group_counts,
            objective_config=objective,
            train_config=local_config,
        )
        torch.save(
            {
                "state_dict": result.state_dict,
                "train_config": config_dict(local_config),
                "target_mean": result.target_mean,
                "target_scale": result.target_scale,
                "kind": spec.kind,
                "parameter_count": result.parameter_count,
            },
            run_dir / "model.pt",
        )
        pd.DataFrame(result.history).to_csv(run_dir / "history.csv", index=False)
        rows = []
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
            rows.append(block)
        pd.concat(rows, ignore_index=True).to_csv(
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
            "architecture": "gine3x128",
            "parameter_count": result.parameter_count,
            "best_epoch": result.best_epoch,
            "train_n": len(train_idx),
            "val_n": len(val_idx),
            "test_n": len(test_idx),
            "val_metrics": result.val_metrics,
            "test_metrics": result.test_metrics,
            "elapsed_seconds": time.time() - started,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, default=json_default), encoding="utf-8"
        )
        print(
            f"done {run_id} {spec.primary_metric}="
            f"{result.test_metrics.get(spec.primary_metric, float('nan')):.4f} "
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


def _rebuild_metrics() -> pd.DataFrame:
    rows = []
    summaries = []
    failed = 0
    for path in sorted(RUNS.glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            failed += 1
            continue
        summaries.append(summary)
        for partition in ["val", "test"]:
            for metric, value in summary[f"{partition}_metrics"].items():
                rows.append(
                    {
                        "run_id": summary["run_id"],
                        "task": summary["task"],
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
                        "parameter_count": summary["parameter_count"],
                        "elapsed_seconds": summary["elapsed_seconds"],
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "gnn_metrics.csv", index=False)
    elapsed = np.asarray([record["elapsed_seconds"] for record in summaries], dtype=float)
    completion = {
        "complete_runs": len(summaries),
        "failed_runs": failed,
        "metric_rows": len(frame),
        "elapsed_seconds_sum": float(elapsed.sum()) if len(elapsed) else 0.0,
        "elapsed_seconds_mean": float(elapsed.mean()) if len(elapsed) else 0.0,
        "elapsed_seconds_max": float(elapsed.max()) if len(elapsed) else 0.0,
    }
    (RESULTS / "gnn_completion_summary.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    return frame


def _test_prevalence(task: str, split: str, seed: int) -> float:
    path = RUNS / f"{task}__{split}__seed{seed}__ERM__gine3x128" / "config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["label_distribution"]["test"].get("positive_fraction", np.nan))


def _bootstrap_mean(frame: pd.DataFrame, column: str, seed: int = 20260712) -> dict:
    rng = np.random.default_rng(seed)
    tasks = frame["task"].unique()
    draws = []
    for _ in range(10000):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        task_means = []
        for task in sampled_tasks:
            values = frame.loc[frame.task == task, column].dropna().to_numpy()
            task_means.append(rng.choice(values, size=len(values), replace=True).mean())
        draws.append(np.mean(task_means))
    task_points = frame.groupby("task")[column].mean()
    return {
        "mean": float(task_points.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "task_n": int(len(task_points)),
        "pair_n": int(len(frame)),
    }


def analyze(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    primary = frame.loc[
        (frame.partition == "test") & (frame.metric == frame.primary_metric)
    ].copy()
    summary = (
        primary.groupby(["task", "kind", "primary_metric", "split", "method"], as_index=False)
        .agg(mean=("value", "mean"), sd=("value", "std"), n=("value", "count"),
             parameter_count=("parameter_count", "first"))
    )
    summary.to_csv(RESULTS / "gnn_primary_summary.csv", index=False)

    erm = primary.loc[primary.method == "ERM"]
    baseline = erm.loc[
        erm.split == "scaffold_clustered_matched",
        ["task", "seed", "primary_metric", "value"],
    ].rename(columns={"value": "baseline"})
    frontier = erm.loc[
        erm.split == "frontier", ["task", "seed", "primary_metric", "value"]
    ].rename(columns={"value": "frontier"})
    paired = baseline.merge(frontier, on=["task", "seed", "primary_metric"], how="inner")
    split_rows = []
    for row in paired.itertuples(index=False):
        if row.primary_metric in HIGHER_IS_BETTER:
            baseline_error, frontier_error = 1 - row.baseline, 1 - row.frontier
            adjusted_baseline_error, adjusted_frontier_error = baseline_error, frontier_error
            absolute = row.baseline - row.frontier
            if row.primary_metric == "auprc":
                adjusted_baseline_error /= max(
                    1 - _test_prevalence(row.task, "scaffold_clustered_matched", row.seed),
                    1e-8,
                )
                adjusted_frontier_error /= max(
                    1 - _test_prevalence(row.task, "frontier", row.seed), 1e-8
                )
        else:
            baseline_error, frontier_error = abs(row.baseline), abs(row.frontier)
            adjusted_baseline_error, adjusted_frontier_error = baseline_error, frontier_error
            absolute = row.frontier - row.baseline
        split_rows.append(
            {
                "task": row.task,
                "seed": row.seed,
                "metric": row.primary_metric,
                "baseline": row.baseline,
                "frontier": row.frontier,
                "absolute_degradation": absolute,
                "relative_error_inflation": frontier_error / max(baseline_error, 1e-8) - 1,
                "chance_adjusted_relative_error_inflation": (
                    adjusted_frontier_error / max(adjusted_baseline_error, 1e-8) - 1
                ),
            }
        )
    split_effect = pd.DataFrame(split_rows)
    split_effect.to_csv(RESULTS / "gnn_split_effects.csv", index=False)

    front = primary.loc[
        (primary.split == "frontier") & primary.method.isin(["ERM", "MV-FREX"])
    ]
    method = front.pivot_table(
        index=["task", "seed", "primary_metric"], columns="method", values="value"
    ).reset_index()
    if {"ERM", "MV-FREX"}.issubset(method.columns):
        method = method.dropna(subset=["ERM", "MV-FREX"]).copy()
        higher = method.primary_metric.isin(HIGHER_IS_BETTER)
        method["erm_error"] = np.where(higher, 1 - method["ERM"], np.abs(method["ERM"]))
        method["method_error"] = np.where(
            higher, 1 - method["MV-FREX"], np.abs(method["MV-FREX"])
        )
        method["normalized_error_reduction"] = (
            method["erm_error"] - method["method_error"]
        ) / method["erm_error"].clip(lower=1e-8)
    else:
        method = pd.DataFrame(
            columns=["task", "seed", "primary_metric", "ERM", "MV-FREX",
                     "erm_error", "method_error", "normalized_error_reduction"]
        )
    method.to_csv(RESULTS / "gnn_method_effects.csv", index=False)

    payload = {}
    if not split_effect.empty:
        payload["frontier_vs_scaffold_clustered_matched"] = _bootstrap_mean(
            split_effect, "relative_error_inflation"
        )
        payload["frontier_vs_scaffold_clustered_matched_chance_adjusted"] = _bootstrap_mean(
            split_effect, "chance_adjusted_relative_error_inflation", seed=20260714
        )
    if not method.empty:
        payload["mvfrex_vs_erm_frontier"] = _bootstrap_mean(
            method, "normalized_error_reduction", seed=20260713
        )
    (RESULTS / "gnn_statistical_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=[spec.name for spec in TASKS])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    train_config = GraphTrainConfig()
    if args.quick:
        train_config = replace(train_config, max_epochs=3, patience=2, batch_size=1024)
    task_map = {spec.name: spec for spec in TASKS}
    for task_name in args.tasks:
        spec = task_map[task_name]
        frame, feature_views = build_feature_cache(spec)
        y = frame["y"].to_numpy(dtype=np.float32)
        graphs = _graph_cache(spec, frame, y)
        smiles = frame["smiles"].astype(str).to_numpy()
        for split_kind in ["scaffold_clustered_matched", "frontier"]:
            for seed in args.seeds:
                split_result = split_for(split_kind, smiles, seed)
                assert_split_integrity(split_result)
                train_idx = np.where(split_result.assignment == "train")[0]
                environments = build_novelty_environments(feature_views, train_idx, seed=seed)
                _set_environment_groups(graphs, environments.groups)
                run_methods = ["ERM"] if split_kind == "scaffold_clustered_matched" else ["ERM", "MV-FREX"]
                for method_name in run_methods:
                    run_one(
                        spec,
                        frame,
                        graphs,
                        split_kind,
                        seed,
                        method_name,
                        METHODS[method_name],
                        split_result,
                        environments,
                        train_config,
                        args.force,
                    )
                analyze(_rebuild_metrics())
    metrics = _rebuild_metrics()
    analyze(metrics)
    print(f"wrote {RESULTS / 'gnn_metrics.csv'} with {len(metrics):,} rows", flush=True)


if __name__ == "__main__":
    main()
