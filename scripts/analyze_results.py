#!/usr/bin/env python3
"""Aggregate raw runs without modifying or filtering unfavorable outcomes."""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if "." not in sys.path:
    sys.path.insert(0, ".")

from src.chem import descriptor_vector
from src.metrics import evaluate


RESULTS = Path("results")
MAIN_METHODS = ["ERM", "V-REx", "GroupDRO", "CVaR", "MV-FREX"]
HIGHER_IS_BETTER = {"auroc", "auprc", "balanced_accuracy", "f1", "mcc", "spearman", "r2"}


def primary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics.loc[
        (metrics["partition"] == "test")
        & (metrics["architecture"] == "mlp")
        & (metrics["metric"] == metrics["primary_metric"])
    ].copy()
    grouped = (
        primary.groupby(["task", "kind", "primary_metric", "split", "method"], as_index=False)
        .agg(mean=("value", "mean"), sd=("value", "std"), n=("value", "count"))
    )
    grouped.to_csv(RESULTS / "primary_summary.csv", index=False)
    return primary


def _test_prevalence(task: str, split: str, seed: int) -> float:
    path = RESULTS / "runs" / f"{task}__{split}__seed{seed}__ERM__mlp" / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    return float(config["label_distribution"]["test"].get("positive_fraction", np.nan))


def paired_split_effects(
    primary: pd.DataFrame,
    baseline_split: str,
    output_name: str,
) -> pd.DataFrame:
    """Compare frontier ERM with one scaffold control using paired task/seed runs."""

    erm = primary.loc[
        (primary["method"] == "ERM")
        & primary["split"].isin([baseline_split, "frontier"])
    ]
    baseline = erm.loc[
        erm["split"] == baseline_split,
        ["task", "seed", "primary_metric", "value"],
    ].rename(columns={"value": "baseline"})
    frontier = erm.loc[
        erm["split"] == "frontier",
        ["task", "seed", "primary_metric", "value"],
    ].rename(columns={"value": "frontier"})
    pivot = baseline.merge(frontier, on=["task", "seed", "primary_metric"], how="inner")
    rows = []
    for row in pivot.itertuples(index=False):
        if not np.isfinite(row.baseline) or not np.isfinite(row.frontier):
            continue
        if row.primary_metric in HIGHER_IS_BETTER:
            absolute = row.baseline - row.frontier
            baseline_error = 1.0 - row.baseline
            frontier_error = 1.0 - row.frontier
            if row.primary_metric == "auprc":
                baseline_prev = _test_prevalence(row.task, baseline_split, row.seed)
                frontier_prev = _test_prevalence(row.task, "frontier", row.seed)
                baseline_adjusted = baseline_error / max(1.0 - baseline_prev, 1e-8)
                frontier_adjusted = frontier_error / max(1.0 - frontier_prev, 1e-8)
            else:
                baseline_prev = frontier_prev = np.nan
                baseline_adjusted = baseline_error
                frontier_adjusted = frontier_error
        else:
            absolute = row.frontier - row.baseline
            baseline_error = abs(row.baseline)
            frontier_error = abs(row.frontier)
            baseline_prev = frontier_prev = np.nan
            baseline_adjusted = baseline_error
            frontier_adjusted = frontier_error
        error_inflation = frontier_error / max(baseline_error, 1e-8) - 1.0
        adjusted_inflation = frontier_adjusted / max(baseline_adjusted, 1e-8) - 1.0
        rows.append(
            {
                "task": row.task,
                "seed": row.seed,
                "metric": row.primary_metric,
                "baseline_split": baseline_split,
                "baseline": row.baseline,
                "frontier": row.frontier,
                "absolute_degradation": absolute,
                "relative_error_inflation": error_inflation,
                "baseline_test_prevalence": baseline_prev,
                "frontier_test_prevalence": frontier_prev,
                "chance_adjusted_relative_error_inflation": adjusted_inflation,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / output_name, index=False)
    return out


def split_effects(primary: pd.DataFrame) -> pd.DataFrame:
    return paired_split_effects(primary, "scaffold", "split_effects.csv")


def method_effects(primary: pd.DataFrame) -> pd.DataFrame:
    frontier = primary.loc[primary["split"] == "frontier"]
    pivot = frontier.pivot_table(
        index=["task", "seed", "primary_metric"], columns="method", values="value"
    ).reset_index()
    rows = []
    for row in pivot.itertuples(index=False):
        baseline = getattr(row, "ERM")
        for method in [name for name in MAIN_METHODS if name != "ERM"]:
            value = getattr(row, method.replace("-", "_"), np.nan)
            # Namedtuple sanitization can be inconsistent for punctuation; fall back to dataframe below.
            if not np.isfinite(value):
                match = frontier.loc[
                    (frontier["task"] == row.task)
                    & (frontier["seed"] == row.seed)
                    & (frontier["method"] == method),
                    "value",
                ]
                value = float(match.iloc[0]) if len(match) else np.nan
            if not np.isfinite(value):
                continue
            if row.primary_metric in HIGHER_IS_BETTER:
                gain = value - baseline
                normalized = gain / max(1.0 - baseline, 1e-8)
            else:
                gain = baseline - value
                normalized = gain / max(abs(baseline), 1e-8)
            rows.append(
                {
                    "task": row.task,
                    "seed": row.seed,
                    "metric": row.primary_metric,
                    "method": method,
                    "erm": baseline,
                    "value": value,
                    "absolute_gain": gain,
                    "normalized_error_reduction": normalized,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "method_effects.csv", index=False)
    return out


def robustness_metrics() -> pd.DataFrame:
    rows = []
    for run_dir in sorted((RESULTS / "runs").iterdir()):
        summary_path = run_dir / "summary.json"
        predictions_path = run_dir / "predictions.csv.gz"
        if not summary_path.exists() or not predictions_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete" or summary.get("architecture") != "mlp":
            continue
        predictions = pd.read_csv(predictions_path)
        predictions = predictions.loc[predictions["partition"] == "test"].copy()
        if summary["kind"] == "binary":
            predictions["loss"] = (predictions["prediction"] - predictions["y_true"]) ** 2
        else:
            config = json.loads((run_dir / "config.json").read_text())
            scale = max(float(config["label_distribution"]["train"]["std"]), 1e-8)
            predictions["loss"] = (predictions["prediction"] - predictions["y_true"]).abs() / scale
        view_group_errors = []
        view_slopes = []
        for view in ["ecfp", "maccs", "atompair", "physchem"]:
            group_column = f"environment_{view}"
            coordinate_column = f"novelty_{view}"
            grouped = predictions.groupby(group_column)["loss"].agg(["mean", "size"])
            eligible = grouped.loc[grouped["size"] >= 10, "mean"]
            if len(eligible):
                view_group_errors.append(float(eligible.max()))
            rho = spearmanr(predictions[coordinate_column], predictions["loss"]).statistic
            if np.isfinite(rho):
                view_slopes.append(float(rho))
        tail_k = max(1, int(np.ceil(0.2 * len(predictions))))
        rows.append(
            {
                "run_id": summary["run_id"],
                "task": summary["task"],
                "split": summary["split"],
                "seed": summary["seed"],
                "method": summary["method"],
                "mean_normalized_error": float(predictions["loss"].mean()),
                "sample_tail_error": float(predictions["loss"].nlargest(tail_k).mean()),
                "worst_view_bin_error": max(view_group_errors) if view_group_errors else np.nan,
                "worst_view_novelty_error_spearman": max(view_slopes) if view_slopes else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "robustness_metrics.csv", index=False)
    return out


def view_correlations() -> pd.DataFrame:
    rows = []
    for path in sorted((RESULTS / "splits").glob("*__frontier__seed0.csv.gz")):
        task = path.name.split("__")[0]
        frame = pd.read_csv(path)
        test = frame.loc[frame["partition"] == "test"]
        for left, right in combinations(["ecfp", "maccs", "atompair", "physchem"], 2):
            rho = spearmanr(test[f"novelty_{left}"], test[f"novelty_{right}"]).statistic
            rows.append({"task": task, "view_a": left, "view_b": right, "spearman": rho, "n": len(test)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "view_novelty_correlations.csv", index=False)
    return out


def frontier_correlations() -> pd.DataFrame:
    names = ["mw", "logp", "tpsa", "hbd", "hba", "rotatable", "rings", "aromatic_rings", "fraction_csp3", "heavy_atoms", "formal_charge", "metal"]
    rows = []
    for path in sorted((RESULTS / "splits").glob("*__frontier__seed0.csv.gz")):
        task = path.name.split("__")[0]
        frame = pd.read_csv(path)
        descriptors = np.stack([descriptor_vector(smiles) for smiles in frame["smiles"]])
        for index, name in enumerate(names):
            if np.std(descriptors[:, index]) == 0:
                rho = np.nan
            else:
                rho = spearmanr(frame["frontier_score"], descriptors[:, index]).statistic
            rows.append({"task": task, "descriptor": name, "spearman": rho, "n": len(frame)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "frontier_correlations.csv", index=False)
    return out


def auxiliary_summaries(metrics: pd.DataFrame) -> None:
    test = metrics.loc[(metrics["partition"] == "test") & (metrics["metric"] == metrics["primary_metric"])].copy()
    ablation = test.loc[
        (test["architecture"] == "mlp")
        & (test["split"] == "frontier")
        & (test["seed"] < 3)
        & test["method"].isin(["ERM", "MV-FREX", "MV-FREX-no-slope", "FREX-single-view"])
    ]
    (
        ablation.groupby(["task", "primary_metric", "method"], as_index=False)
        .agg(mean=("value", "mean"), sd=("value", "std"), n=("value", "count"))
        .to_csv(RESULTS / "ablation_summary.csv", index=False)
    )
    linear = test.loc[
        test["split"].isin(["frontier", "scaffold_matched", "scaffold_clustered_matched"])
        & (test["seed"] < 3)
        & test["method"].isin(["ERM", "MV-FREX"])
    ]
    (
        linear.groupby(["task", "primary_metric", "split", "architecture", "method"], as_index=False)
        .agg(mean=("value", "mean"), sd=("value", "std"), n=("value", "count"))
        .to_csv(RESULTS / "linear_control_summary.csv", index=False)
    )
    calibration = metrics.loc[
        (metrics["partition"] == "test")
        & (metrics["architecture"] == "mlp")
        & metrics["metric"].isin(["brier", "ece"])
        & metrics["method"].isin(MAIN_METHODS)
    ]
    (
        calibration.groupby(["task", "split", "method", "metric"], as_index=False)
        .agg(mean=("value", "mean"), sd=("value", "std"), n=("value", "count"))
        .to_csv(RESULTS / "calibration_summary.csv", index=False)
    )


def split_label_summary() -> pd.DataFrame:
    rows = []
    for path in sorted((RESULTS / "runs").glob("*__ERM__mlp/config.json")):
        config = json.loads(path.read_text())
        split_name = config["run_id"].split("__")[1]
        for partition, values in config["label_distribution"].items():
            row = {
                "task": config["task"]["name"],
                "kind": config["task"]["kind"],
                "split": split_name,
                "seed": config["seed"],
                "partition": partition,
                **values,
            }
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "split_label_summary.csv", index=False)
    return out


def split_partition_summary() -> pd.DataFrame:
    rows = []
    for path in sorted((RESULTS / "splits").glob("*.csv.gz")):
        stem = path.name.removesuffix(".csv.gz")
        task, split, seed_text = stem.split("__")
        frame = pd.read_csv(path, usecols=["partition", "scaffold"])
        for partition, block in frame.groupby("partition"):
            rows.append(
                {
                    "task": task,
                    "split": split,
                    "seed": int(seed_text.removeprefix("seed")),
                    "partition": partition,
                    "molecules": int(len(block)),
                    "groups": int(block["scaffold"].nunique()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "split_partition_summary.csv", index=False)
    return out


def completion_summary(metrics: pd.DataFrame) -> dict:
    records = []
    failed = 0
    for path in sorted((RESULTS / "runs").glob("*/summary.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "complete":
            records.append(record)
        else:
            failed += 1
    elapsed = np.asarray([float(record["elapsed_seconds"]) for record in records])
    payload = {
        "complete_runs": int(len(records)),
        "failed_runs": int(failed),
        "metric_rows": int(len(metrics)),
        "split_manifests": int(len(list((RESULTS / "splits").glob("*.csv.gz")))),
        "elapsed_seconds_sum": float(elapsed.sum()),
        "elapsed_seconds_mean": float(elapsed.mean()),
        "elapsed_seconds_max": float(elapsed.max()),
    }
    (RESULTS / "completion_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def mw_sensitivity_metrics() -> pd.DataFrame:
    rows = []
    mw_cache: dict[str, float] = {}
    for run_dir in sorted((RESULTS / "runs").iterdir()):
        summary_path = run_dir / "summary.json"
        predictions_path = run_dir / "predictions.csv.gz"
        if not summary_path.exists() or not predictions_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete" or summary.get("architecture") != "mlp":
            continue
        predictions = pd.read_csv(predictions_path)
        predictions = predictions.loc[predictions["partition"] == "test"].copy()
        molecular_weight = []
        for smiles in predictions["smiles"]:
            if smiles not in mw_cache:
                mw_cache[smiles] = float(descriptor_vector(smiles)[0])
            molecular_weight.append(mw_cache[smiles])
        predictions["mw"] = molecular_weight
        filtered = predictions.loc[predictions["mw"] <= 1000]
        if len(filtered) == 0:
            continue
        evaluated = evaluate(
            summary["kind"], filtered["y_true"].to_numpy(), filtered["prediction"].to_numpy()
        )
        for metric, value in evaluated.items():
            rows.append(
                {
                    "run_id": summary["run_id"],
                    "task": summary["task"],
                    "split": summary["split"],
                    "seed": summary["seed"],
                    "method": summary["method"],
                    "metric": metric,
                    "value": value,
                    "retained_n": len(filtered),
                    "original_n": len(predictions),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "mw_le_1000_sensitivity_metrics.csv", index=False)
    return out


def certificate_summary() -> pd.DataFrame:
    rows = []
    for path in sorted((RESULTS / "runs").glob("*MV-FREX__mlp/config.json")):
        config = json.loads(path.read_text())
        counts = np.asarray(config["environment_counts"], dtype=float)
        groups = counts.size
        correction = np.sqrt(np.log(2 * groups / config["objective"]["confidence_delta"]) / (2 * counts))
        rows.append(
            {
                "run_id": config["run_id"],
                "task": config["task"]["name"],
                "split": config["split"]["kind"],
                "seed": config["seed"],
                "minimum_group_n": float(counts.min()),
                "maximum_group_n": float(counts.max()),
                "maximum_hoeffding_correction": float(correction.max()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "certificate_summary.csv", index=False)
    return out


def bootstrap_mean(values: pd.DataFrame, column: str, rng: np.random.Generator, reps: int = 10000) -> dict:
    tasks = values["task"].unique()
    draws = []
    for _ in range(reps):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        task_means = []
        for task in sampled_tasks:
            task_values = values.loc[values["task"] == task, column].dropna().to_numpy()
            if len(task_values):
                sampled = rng.choice(task_values, size=len(task_values), replace=True)
                task_means.append(sampled.mean())
        if task_means:
            draws.append(np.mean(task_means))
    draws = np.asarray(draws)
    return {
        "mean": float(values.groupby("task")[column].mean().mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "task_n": int(values["task"].nunique()),
        "pair_n": int(len(values)),
    }


def bootstrap_median(values: pd.DataFrame, column: str, rng: np.random.Generator, reps: int = 10000) -> dict:
    tasks = values["task"].unique()
    draws = []
    for _ in range(reps):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        task_means = []
        for task in sampled_tasks:
            task_values = values.loc[values["task"] == task, column].dropna().to_numpy()
            if len(task_values):
                sampled = rng.choice(task_values, size=len(task_values), replace=True)
                task_means.append(sampled.mean())
        if task_means:
            draws.append(np.median(task_means))
    draws = np.asarray(draws)
    task_points = values.groupby("task")[column].mean().to_numpy()
    return {
        "median": float(np.median(task_points)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "task_n": int(values["task"].nunique()),
        "pair_n": int(len(values)),
    }


def statistical_summary(
    split_effect: pd.DataFrame,
    matched_effect: pd.DataFrame,
    strict_effect: pd.DataFrame,
    strict_linear_effect: pd.DataFrame,
    cyp_auroc_effect: pd.DataFrame,
    method_effect: pd.DataFrame,
    robust: pd.DataFrame,
) -> dict:
    rng = np.random.default_rng(20260711)
    summary = {
        "frontier_vs_scaffold": bootstrap_mean(split_effect, "relative_error_inflation", rng),
        "frontier_vs_scaffold_matched": bootstrap_mean(
            matched_effect, "relative_error_inflation", rng
        ),
        "frontier_vs_scaffold_clustered_matched": bootstrap_mean(
            strict_effect, "relative_error_inflation", rng
        ),
        "frontier_vs_scaffold_clustered_matched_chance_adjusted": bootstrap_mean(
            strict_effect, "chance_adjusted_relative_error_inflation", rng
        ),
        "frontier_vs_scaffold_clustered_matched_task_median": bootstrap_median(
            strict_effect, "relative_error_inflation", rng
        ),
        "linear_frontier_vs_scaffold_clustered_matched": bootstrap_mean(
            strict_linear_effect, "relative_error_inflation", rng
        ),
        "cyp2d6_auroc_frontier_vs_scaffold_clustered_matched": bootstrap_mean(
            cyp_auroc_effect, "relative_error_inflation", rng
        ),
        "methods_vs_erm_frontier": {},
    }
    for method, block in method_effect.groupby("method"):
        record = bootstrap_mean(block, "normalized_error_reduction", rng)
        record["wins"] = int((block["absolute_gain"] > 0).sum())
        record["ties"] = int(np.isclose(block["absolute_gain"], 0.0, atol=1e-8).sum())
        record["losses"] = int((block["absolute_gain"] < 0).sum())
        summary["methods_vs_erm_frontier"][method] = record

    main = robust.loc[
        (robust["split"] == "frontier")
        & (robust["method"].isin(["ERM", "MV-FREX"]))
    ]
    pivot = main.pivot_table(index=["task", "seed"], columns="method", values="worst_view_bin_error").dropna().reset_index()
    pivot["reduction"] = (pivot["ERM"] - pivot["MV-FREX"]) / pivot["ERM"].clip(lower=1e-8)
    summary["mvfrex_worst_view_bin_error_reduction"] = bootstrap_mean(pivot, "reduction", rng)
    (RESULTS / "statistical_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    auxiliary_summaries(metrics)
    split_label_summary()
    split_partition_summary()
    completion_summary(metrics)
    primary = primary_table(metrics)
    split_effect = split_effects(primary)
    matched_effect = paired_split_effects(
        primary, "scaffold_matched", "matched_split_effects.csv"
    )
    strict_effect = paired_split_effects(
        primary,
        "scaffold_clustered_matched",
        "strict_matched_split_effects.csv",
    )
    strict_linear_primary = metrics.loc[
        (metrics["partition"] == "test")
        & (metrics["architecture"] == "linear")
        & (metrics["method"] == "ERM")
        & (metrics["metric"] == metrics["primary_metric"])
    ].copy()
    strict_linear_effect = paired_split_effects(
        strict_linear_primary,
        "scaffold_clustered_matched",
        "strict_matched_linear_split_effects.csv",
    )
    cyp_auroc_primary = metrics.loc[
        (metrics["partition"] == "test")
        & (metrics["architecture"] == "mlp")
        & (metrics["method"] == "ERM")
        & (metrics["task"] == "CYP2D6")
        & (metrics["metric"] == "auroc")
    ].copy()
    cyp_auroc_primary["primary_metric"] = "auroc"
    cyp_auroc_effect = paired_split_effects(
        cyp_auroc_primary,
        "scaffold_clustered_matched",
        "cyp2d6_auroc_split_sensitivity.csv",
    )
    method_effect = method_effects(primary)
    robust = robustness_metrics()
    view_correlations()
    frontier_correlations()
    certificate_summary()
    mw_sensitivity_metrics()
    summary = statistical_summary(
        split_effect,
        matched_effect,
        strict_effect,
        strict_linear_effect,
        cyp_auroc_effect,
        method_effect,
        robust,
    )
    print((RESULTS / "primary_summary.csv").read_text())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
