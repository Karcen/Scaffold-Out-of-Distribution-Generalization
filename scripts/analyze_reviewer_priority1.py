#!/usr/bin/env python3
"""Post-hoc diagnostics requested in reviewer Priority 1.

This script only reads completed ERM/MLP prediction records.  It does not train
models or modify the manuscript.  Outputs are written to
``results/reviewer_priority1``.

The decomposition is deliberately descriptive, not causal:

* Binary endpoints: reweight the frontier test observations so their effective
  class prevalence equals the paired matched-scaffold test prevalence.
* Regression endpoints: reweight the frontier test observations so their
  outcome-bin histogram equals the paired matched-scaffold histogram (10
  pooled-quantile bins; 5/20-bin sensitivity is also reported).

For a lower-is-better loss L, the observed gap is decomposed exactly as

    L_frontier - L_matched
      = (L_frontier - L_frontier,standardized)
      + (L_frontier,standardized - L_matched).

The first term is an outcome-mixture component under the chosen
standardization; the second is a residual within-outcome component that still
mixes support shift, model shift, and sampling variation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)


RESULTS = Path("results")
OUTPUT = RESULTS / "reviewer_priority1"
MATCHED_SPLIT = "scaffold_clustered_matched"
FRONTIER_SPLIT = "frontier"
SEEDS = tuple(range(5))
BOOTSTRAP_SEED = 20260711
CLASSIFICATION_TASKS = ("BBB", "HIA", "DILI", "CYP2D6")
REGRESSION_TASKS = ("PPBR", "Solubility")


def _run_dir(task: str, split: str, seed: int) -> Path:
    return RESULTS / "runs" / f"{task}__{split}__seed{seed}__ERM__mlp"


def _load(task: str, split: str, seed: int) -> tuple[pd.DataFrame, dict, dict]:
    run_dir = _run_dir(task, split, seed)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise RuntimeError(f"Incomplete run: {run_dir.name}")
    predictions = pd.read_csv(run_dir / "predictions.csv.gz")
    expected = {"row_index", "partition", "y_true", "prediction", "frontier_score"}
    missing = expected.difference(predictions.columns)
    if missing:
        raise ValueError(f"{run_dir.name}: missing prediction columns {sorted(missing)}")
    return predictions, summary, config


def _test(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[frame["partition"] == "test"].copy()
    if out.empty:
        raise ValueError("Prediction file contains no test observations")
    return out


def _prevalence_weights(y: np.ndarray, target_prevalence: float) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    observed = float(y.mean())
    if not 0.0 < observed < 1.0 or not 0.0 < target_prevalence < 1.0:
        raise ValueError("Prevalence standardization requires both classes")
    return np.where(
        y == 1,
        target_prevalence / observed,
        (1.0 - target_prevalence) / (1.0 - observed),
    )


def _outcome_bin_weights(
    reference_y: np.ndarray,
    frontier_y: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, float, int]:
    """Histogram-standardize frontier outcomes to the matched distribution."""

    reference_y = np.asarray(reference_y, dtype=float)
    frontier_y = np.asarray(frontier_y, dtype=float)
    pooled = np.concatenate([reference_y, frontier_y])
    edges = np.unique(np.quantile(pooled, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        raise ValueError("Too few distinct outcome-bin edges")
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_bin = np.digitize(reference_y, edges[1:-1], right=True)
    frontier_bin = np.digitize(frontier_y, edges[1:-1], right=True)
    bin_n = len(edges) - 1
    ref_mass = np.bincount(ref_bin, minlength=bin_n) / len(reference_y)
    frontier_mass = np.bincount(frontier_bin, minlength=bin_n) / len(frontier_y)
    overlap = float(ref_mass[frontier_mass > 0].sum())
    weights = np.asarray(
        [ref_mass[index] / frontier_mass[index] if frontier_mass[index] > 0 else 0.0
         for index in frontier_bin],
        dtype=float,
    )
    return weights, overlap, bin_n


def _stratified_auc(
    y: np.ndarray,
    probability: np.ndarray,
    coordinate: np.ndarray,
    bins: int = 5,
) -> dict[str, float]:
    """Split pairwise AUROC into within- and cross-coordinate-bin terms."""

    frame = pd.DataFrame({"y": y, "p": probability, "coordinate": coordinate})
    frame["bin"] = pd.qcut(frame["coordinate"], q=bins, labels=False, duplicates="drop")
    if frame["bin"].isna().all():
        return {
            "novelty_within_bin_auroc": np.nan,
            "novelty_cross_bin_auroc": np.nan,
            "novelty_within_pair_fraction": np.nan,
        }
    positive = frame.loc[frame["y"] == 1]
    negative = frame.loc[frame["y"] == 0]
    total_pairs = len(positive) * len(negative)
    total_wins = float(roc_auc_score(frame["y"], frame["p"]) * total_pairs)
    within_pairs = 0
    within_wins = 0.0
    for _, block in frame.groupby("bin", observed=True):
        pos = block.loc[block["y"] == 1, "p"].to_numpy()
        neg = block.loc[block["y"] == 0, "p"].to_numpy()
        if not len(pos) or not len(neg):
            continue
        differences = pos[:, None] - neg[None, :]
        within_pairs += differences.size
        within_wins += float((differences > 0).sum() + 0.5 * (differences == 0).sum())
    cross_pairs = total_pairs - within_pairs
    return {
        "novelty_within_bin_auroc": within_wins / within_pairs if within_pairs else np.nan,
        "novelty_cross_bin_auroc": (
            (total_wins - within_wins) / cross_pairs if cross_pairs else np.nan
        ),
        "novelty_within_pair_fraction": within_pairs / total_pairs if total_pairs else np.nan,
    }


def binary_diagnostics() -> pd.DataFrame:
    rows: list[dict] = []
    for task in CLASSIFICATION_TASKS:
        for seed in SEEDS:
            matched_all, _, matched_config = _load(task, MATCHED_SPLIT, seed)
            frontier_all, _, frontier_config = _load(task, FRONTIER_SPLIT, seed)
            matched = _test(matched_all)
            frontier = _test(frontier_all)
            matched_y = matched["y_true"].to_numpy(dtype=int)
            matched_p = matched["prediction"].to_numpy(dtype=float)
            frontier_y = frontier["y_true"].to_numpy(dtype=int)
            frontier_p = frontier["prediction"].to_numpy(dtype=float)
            matched_prevalence = float(matched_y.mean())
            frontier_prevalence = float(frontier_y.mean())
            weights = _prevalence_weights(frontier_y, matched_prevalence)

            matched_auc = float(roc_auc_score(matched_y, matched_p))
            frontier_auc = float(roc_auc_score(frontier_y, frontier_p))
            standardized_auc = float(
                roc_auc_score(frontier_y, frontier_p, sample_weight=weights)
            )
            matched_ap = float(average_precision_score(matched_y, matched_p))
            frontier_ap = float(average_precision_score(frontier_y, frontier_p))
            standardized_ap = float(
                average_precision_score(frontier_y, frontier_p, sample_weight=weights)
            )
            matched_brier = float(brier_score_loss(matched_y, matched_p))
            frontier_brier = float(brier_score_loss(frontier_y, frontier_p))
            standardized_brier = float(
                np.average((frontier_p - frontier_y) ** 2, weights=weights)
            )
            frontier_validation = frontier_all.loc[frontier_all["partition"] == "val"]
            stratified = _stratified_auc(
                frontier_y,
                frontier_p,
                frontier["frontier_score"].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "matched_n": len(matched),
                    "frontier_n": len(frontier),
                    "matched_train_prevalence": matched_config["label_distribution"]["train"]["positive_fraction"],
                    "frontier_train_prevalence": frontier_config["label_distribution"]["train"]["positive_fraction"],
                    "matched_test_prevalence": matched_prevalence,
                    "frontier_test_prevalence": frontier_prevalence,
                    "prevalence_shift_frontier_minus_matched": frontier_prevalence - matched_prevalence,
                    "matched_auroc": matched_auc,
                    "frontier_validation_auroc": float(
                        roc_auc_score(
                            frontier_validation["y_true"], frontier_validation["prediction"]
                        )
                    ),
                    "frontier_auroc": frontier_auc,
                    "frontier_auroc_prevalence_standardized": standardized_auc,
                    "auroc_degradation": matched_auc - frontier_auc,
                    "matched_auprc": matched_ap,
                    "frontier_auprc": frontier_ap,
                    "frontier_auprc_prevalence_standardized": standardized_ap,
                    "auprc_degradation_total": matched_ap - frontier_ap,
                    "auprc_prevalence_component": standardized_ap - frontier_ap,
                    "auprc_residual_component": matched_ap - standardized_ap,
                    "matched_balanced_accuracy": float(
                        balanced_accuracy_score(matched_y, matched_p >= 0.5)
                    ),
                    "frontier_balanced_accuracy": float(
                        balanced_accuracy_score(frontier_y, frontier_p >= 0.5)
                    ),
                    "matched_brier": matched_brier,
                    "frontier_brier": frontier_brier,
                    "frontier_brier_prevalence_standardized": standardized_brier,
                    "brier_gap_total": frontier_brier - matched_brier,
                    "brier_prevalence_component": frontier_brier - standardized_brier,
                    "brier_residual_component": standardized_brier - matched_brier,
                    "frontier_mean_score_y0": float(frontier.loc[frontier["y_true"] == 0, "prediction"].mean()),
                    "frontier_mean_score_y1": float(frontier.loc[frontier["y_true"] == 1, "prediction"].mean()),
                    "frontier_score_mean_difference_y1_minus_y0": float(
                        frontier.loc[frontier["y_true"] == 1, "prediction"].mean()
                        - frontier.loc[frontier["y_true"] == 0, "prediction"].mean()
                    ),
                    "frontier_prediction_novelty_spearman": float(
                        spearmanr(frontier_p, frontier["frontier_score"]).statistic
                    ),
                    "frontier_label_novelty_spearman": float(
                        spearmanr(frontier_y, frontier["frontier_score"]).statistic
                    ),
                    **stratified,
                }
            )
    out = pd.DataFrame(rows)
    # AUROC must be exactly invariant to class-constant prevalence weights up to FP noise.
    delta = (out["frontier_auroc"] - out["frontier_auroc_prevalence_standardized"]).abs()
    if float(delta.max()) > 1e-12:
        raise AssertionError(f"Prevalence weighting changed AUROC by {delta.max()}")
    return out


def regression_diagnostics(bins: int = 10) -> pd.DataFrame:
    rows: list[dict] = []
    for task in REGRESSION_TASKS:
        for seed in SEEDS:
            matched_all, _, _ = _load(task, MATCHED_SPLIT, seed)
            frontier_all, _, _ = _load(task, FRONTIER_SPLIT, seed)
            matched = _test(matched_all)
            frontier = _test(frontier_all)
            matched_y = matched["y_true"].to_numpy(dtype=float)
            matched_p = matched["prediction"].to_numpy(dtype=float)
            frontier_y = frontier["y_true"].to_numpy(dtype=float)
            frontier_p = frontier["prediction"].to_numpy(dtype=float)
            weights, overlap, realized_bins = _outcome_bin_weights(
                matched_y, frontier_y, bins=bins
            )
            matched_mae = float(mean_absolute_error(matched_y, matched_p))
            frontier_mae = float(mean_absolute_error(frontier_y, frontier_p))
            standardized_mae = float(
                np.average(np.abs(frontier_p - frontier_y), weights=weights)
            )
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "requested_outcome_bins": bins,
                    "realized_outcome_bins": realized_bins,
                    "matched_n": len(matched),
                    "frontier_n": len(frontier),
                    "matched_test_mean": float(matched_y.mean()),
                    "frontier_test_mean": float(frontier_y.mean()),
                    "mean_shift_frontier_minus_matched": float(frontier_y.mean() - matched_y.mean()),
                    "matched_test_std": float(matched_y.std(ddof=0)),
                    "frontier_test_std": float(frontier_y.std(ddof=0)),
                    "std_shift_frontier_minus_matched": float(frontier_y.std(ddof=0) - matched_y.std(ddof=0)),
                    "matched_mae": matched_mae,
                    "frontier_mae": frontier_mae,
                    "frontier_mae_outcome_standardized": standardized_mae,
                    "mae_gap_total": frontier_mae - matched_mae,
                    "mae_outcome_mixture_component": frontier_mae - standardized_mae,
                    "mae_residual_component": standardized_mae - matched_mae,
                    "matched_spearman": float(spearmanr(matched_y, matched_p).statistic),
                    "frontier_spearman": float(spearmanr(frontier_y, frontier_p).statistic),
                    "reference_histogram_overlap_mass": overlap,
                }
            )
    return pd.DataFrame(rows)


def _hierarchical_bootstrap(
    frame: pd.DataFrame,
    omitted_task: str | None,
    reps: int,
) -> dict[str, float | int | str]:
    block = frame.loc[frame["task"] != omitted_task].copy() if omitted_task else frame.copy()
    tasks = sorted(block["task"].unique())
    task_points = block.groupby("task")["relative_error_inflation"].mean()
    offset = 0 if omitted_task is None else sorted(frame["task"].unique()).index(omitted_task) + 1
    rng = np.random.default_rng(BOOTSTRAP_SEED + offset)
    draws = np.empty(reps, dtype=float)
    for replicate in range(reps):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        sampled_means = []
        for task in sampled_tasks:
            values = block.loc[
                block["task"] == task, "relative_error_inflation"
            ].to_numpy(dtype=float)
            sampled_means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        draws[replicate] = float(np.mean(sampled_means))
    return {
        "omitted_task": omitted_task or "NONE",
        "remaining_task_n": len(tasks),
        "remaining_pair_n": len(block),
        "mean_relative_error_inflation": float(task_points.mean()),
        "mean_relative_error_inflation_percent": float(100.0 * task_points.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "ci95_low_percent": float(100.0 * np.quantile(draws, 0.025)),
        "ci95_high_percent": float(100.0 * np.quantile(draws, 0.975)),
        "taskwise_median": float(task_points.median()),
        "taskwise_median_percent": float(100.0 * task_points.median()),
        "taskwise_min": float(task_points.min()),
        "taskwise_max": float(task_points.max()),
        "taskwise_min_percent": float(100.0 * task_points.min()),
        "taskwise_max_percent": float(100.0 * task_points.max()),
    }


def leave_one_endpoint_out(reps: int) -> pd.DataFrame:
    effects = pd.read_csv(RESULTS / "strict_matched_split_effects.csv")
    rows = [_hierarchical_bootstrap(effects, None, reps)]
    rows.extend(
        _hierarchical_bootstrap(effects, task, reps)
        for task in sorted(effects["task"].unique())
    )
    out = pd.DataFrame(rows)
    all_mean = float(out.loc[out["omitted_task"] == "NONE", "mean_relative_error_inflation"].iloc[0])
    out["change_from_all_endpoint_mean"] = out["mean_relative_error_inflation"] - all_mean
    out["change_from_all_endpoint_mean_percent_points"] = 100.0 * out["change_from_all_endpoint_mean"]
    return out


def _aggregate_seed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    identity = [column for column in ["task", "requested_outcome_bins"] if column in frame]
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {"seed", "requested_outcome_bins"}
    ]
    mean = frame.groupby(identity, as_index=False)[numeric].mean()
    sd = frame.groupby(identity, as_index=False)[numeric].std(ddof=1)
    mean = mean.rename(columns={column: f"{column}_mean" for column in numeric})
    sd = sd.rename(columns={column: f"{column}_sd" for column in numeric})
    out = mean.merge(sd, on=identity, how="inner", validate="one_to_one")
    out["seed_n"] = frame.groupby(identity).size().to_numpy()
    return out


def compact_endpoint_summary(binary: pd.DataFrame, regression: pd.DataFrame) -> pd.DataFrame:
    """Create one manuscript-facing row per endpoint with common semantics."""

    rows: list[dict] = []
    primary_metric = {"BBB": "auroc", "HIA": "auroc", "DILI": "auroc", "CYP2D6": "auprc"}
    for task, block in binary.groupby("task", sort=False):
        total_loss_gap = float(block["brier_gap_total"].mean())
        if primary_metric[task] == "auroc":
            matched_primary = float(block["matched_auroc"].mean())
            frontier_primary = float(block["frontier_auroc"].mean())
            standardized_primary = float(
                block["frontier_auroc_prevalence_standardized"].mean()
            )
        else:
            matched_primary = float(block["matched_auprc"].mean())
            frontier_primary = float(block["frontier_auprc"].mean())
            standardized_primary = float(
                block["frontier_auprc_prevalence_standardized"].mean()
            )
        rows.append(
            {
                "task": task,
                "kind": "binary",
                "standardization": "exact class-prevalence reweighting",
                "label_location_metric": "positive_fraction",
                "matched_label_location": float(block["matched_test_prevalence"].mean()),
                "frontier_label_location": float(block["frontier_test_prevalence"].mean()),
                "label_location_shift_frontier_minus_matched": float(
                    block["prevalence_shift_frontier_minus_matched"].mean()
                ),
                "primary_metric": primary_metric[task],
                "matched_primary": matched_primary,
                "frontier_primary": frontier_primary,
                "standardized_frontier_primary": standardized_primary,
                "primary_degradation": matched_primary - frontier_primary,
                "threshold_free_ranking_metric": "auroc",
                "matched_ranking": float(block["matched_auroc"].mean()),
                "frontier_ranking": float(block["frontier_auroc"].mean()),
                "standardized_frontier_ranking": float(
                    block["frontier_auroc_prevalence_standardized"].mean()
                ),
                "loss_metric": "brier",
                "matched_loss": float(block["matched_brier"].mean()),
                "frontier_loss": float(block["frontier_brier"].mean()),
                "standardized_frontier_loss": float(
                    block["frontier_brier_prevalence_standardized"].mean()
                ),
                "total_loss_gap": total_loss_gap,
                "label_mixture_component": float(block["brier_prevalence_component"].mean()),
                "residual_component": float(block["brier_residual_component"].mean()),
                "label_mixture_fraction_of_positive_gap": (
                    float(block["brier_prevalence_component"].mean()) / total_loss_gap
                    if total_loss_gap > 0
                    else np.nan
                ),
                "matched_auprc": float(block["matched_auprc"].mean()),
                "frontier_auprc": float(block["frontier_auprc"].mean()),
                "standardized_frontier_auprc": float(
                    block["frontier_auprc_prevalence_standardized"].mean()
                ),
                "auprc_prevalence_component": float(
                    block["auprc_prevalence_component"].mean()
                ),
                "auprc_residual_component": float(block["auprc_residual_component"].mean()),
                "seed_n": len(block),
            }
        )
    for task, block in regression.groupby("task", sort=False):
        total_loss_gap = float(block["mae_gap_total"].mean())
        rows.append(
            {
                "task": task,
                "kind": "regression",
                "standardization": "10 pooled-quantile outcome bins",
                "label_location_metric": "mean",
                "matched_label_location": float(block["matched_test_mean"].mean()),
                "frontier_label_location": float(block["frontier_test_mean"].mean()),
                "label_location_shift_frontier_minus_matched": float(
                    block["mean_shift_frontier_minus_matched"].mean()
                ),
                "primary_metric": "mae",
                "matched_primary": float(block["matched_mae"].mean()),
                "frontier_primary": float(block["frontier_mae"].mean()),
                "standardized_frontier_primary": float(
                    block["frontier_mae_outcome_standardized"].mean()
                ),
                "primary_degradation": total_loss_gap,
                "threshold_free_ranking_metric": "spearman",
                "matched_ranking": float(block["matched_spearman"].mean()),
                "frontier_ranking": float(block["frontier_spearman"].mean()),
                "standardized_frontier_ranking": np.nan,
                "loss_metric": "mae",
                "matched_loss": float(block["matched_mae"].mean()),
                "frontier_loss": float(block["frontier_mae"].mean()),
                "standardized_frontier_loss": float(
                    block["frontier_mae_outcome_standardized"].mean()
                ),
                "total_loss_gap": total_loss_gap,
                "label_mixture_component": float(
                    block["mae_outcome_mixture_component"].mean()
                ),
                "residual_component": float(block["mae_residual_component"].mean()),
                "label_mixture_fraction_of_positive_gap": (
                    float(block["mae_outcome_mixture_component"].mean()) / total_loss_gap
                    if total_loss_gap > 0
                    else np.nan
                ),
                "matched_auprc": np.nan,
                "frontier_auprc": np.nan,
                "standardized_frontier_auprc": np.nan,
                "auprc_prevalence_component": np.nan,
                "auprc_residual_component": np.nan,
                "seed_n": len(block),
            }
        )
    return pd.DataFrame(rows)


def bbb_ensemble_diagnostics(binary: pd.DataFrame, reps: int) -> tuple[pd.DataFrame, dict]:
    seed_frames = []
    for seed in SEEDS:
        predictions, _, _ = _load("BBB", FRONTIER_SPLIT, seed)
        test = _test(predictions)[["row_index", "y_true", "prediction", "frontier_score"]].copy()
        test = test.rename(columns={"prediction": f"prediction_seed{seed}"})
        seed_frames.append(test)
    merged = seed_frames[0]
    for seed, frame in enumerate(seed_frames[1:], start=1):
        merged = merged.merge(
            frame[["row_index", f"prediction_seed{seed}"]],
            on="row_index",
            how="inner",
            validate="one_to_one",
        )
    membership_identical = len(merged) == len(seed_frames[0]) and all(
        set(frame["row_index"]) == set(seed_frames[0]["row_index"])
        for frame in seed_frames[1:]
    )
    prediction_columns = [f"prediction_seed{seed}" for seed in SEEDS]
    merged["ensemble_prediction"] = merged[prediction_columns].mean(axis=1)
    merged["frontier_quintile"] = pd.qcut(
        merged["frontier_score"], q=5, labels=False, duplicates="drop"
    )
    quintiles = (
        merged.groupby("frontier_quintile", as_index=False, observed=True)
        .agg(
            n=("y_true", "size"),
            positive_fraction=("y_true", "mean"),
            mean_prediction=("ensemble_prediction", "mean"),
            mean_frontier_score=("frontier_score", "mean"),
        )
    )
    aucs = []
    for quintile, block in merged.groupby("frontier_quintile", observed=True):
        value = (
            float(roc_auc_score(block["y_true"], block["ensemble_prediction"]))
            if block["y_true"].nunique() == 2
            else np.nan
        )
        aucs.append((quintile, value))
    auc_frame = pd.DataFrame(aucs, columns=["frontier_quintile", "within_quintile_auroc"])
    quintiles = quintiles.merge(auc_frame, on="frontier_quintile", how="left")

    y = merged["y_true"].to_numpy(dtype=int)
    prediction_matrix = merged[prediction_columns].to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(reps, dtype=float)
    for replicate in range(reps):
        seed_indices = rng.integers(0, len(SEEDS), size=len(SEEDS))
        compound_indices = rng.integers(0, len(merged), size=len(merged))
        ensemble = prediction_matrix[:, seed_indices].mean(axis=1)
        sampled_y = y[compound_indices]
        if np.unique(sampled_y).size < 2:
            draws[replicate] = np.nan
        else:
            draws[replicate] = roc_auc_score(sampled_y, ensemble[compound_indices])
    draws = draws[np.isfinite(draws)]
    ensemble_auc = float(roc_auc_score(y, merged["ensemble_prediction"]))
    stratified = _stratified_auc(
        y,
        merged["ensemble_prediction"].to_numpy(dtype=float),
        merged["frontier_score"].to_numpy(dtype=float),
    )
    bbb = binary.loc[binary["task"] == "BBB"]
    payload = {
        "test_membership_identical_across_five_seeds": bool(membership_identical),
        "test_n": int(len(merged)),
        "positive_n": int(y.sum()),
        "negative_n": int((1 - y).sum()),
        "frontier_test_auroc_seed_mean": float(bbb["frontier_auroc"].mean()),
        "frontier_test_auroc_seed_sd": float(bbb["frontier_auroc"].std(ddof=1)),
        "frontier_test_auroc_seed_min": float(bbb["frontier_auroc"].min()),
        "frontier_test_auroc_seed_max": float(bbb["frontier_auroc"].max()),
        "frontier_validation_auroc_seed_mean": float(bbb["frontier_validation_auroc"].mean()),
        "matched_test_auroc_seed_mean": float(bbb["matched_auroc"].mean()),
        "ensemble_auroc": ensemble_auc,
        "ensemble_inverted_auroc": 1.0 - ensemble_auc,
        "compound_and_seed_bootstrap_reps": int(len(draws)),
        "ensemble_auroc_bootstrap_ci95_low": float(np.quantile(draws, 0.025)),
        "ensemble_auroc_bootstrap_ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_fraction_at_or_above_chance": float(np.mean(draws >= 0.5)),
        "ensemble_mean_score_y0": float(merged.loc[merged["y_true"] == 0, "ensemble_prediction"].mean()),
        "ensemble_mean_score_y1": float(merged.loc[merged["y_true"] == 1, "ensemble_prediction"].mean()),
        "prediction_frontier_score_spearman": float(
            spearmanr(merged["ensemble_prediction"], merged["frontier_score"]).statistic
        ),
        "label_frontier_score_spearman": float(
            spearmanr(merged["y_true"], merged["frontier_score"]).statistic
        ),
        **stratified,
    }
    return quintiles, payload


def _write_readme(bootstrap_reps: int) -> None:
    text = f"""# Reviewer Priority 1 diagnostics

Generated by `python scripts/analyze_reviewer_priority1.py --bootstrap-reps {bootstrap_reps}`.
No model was retrained; all calculations use completed ERM/MLP prediction files.

## Files

- `leave_one_endpoint_out.csv`: strict matched-control headline sensitivity.
- `binary_prevalence_diagnostics_per_seed.csv` and `_summary.csv`: exact
  class-prevalence standardization of AUROC, AUPRC, and Brier score.
- `regression_outcome_diagnostics_per_seed.csv` and `_summary.csv`: continuous
  outcome-mixture standardization using 10 pooled-quantile bins.
- `regression_reweighting_sensitivity.csv`: 5/10/20-bin sensitivity.
- `endpoint_shift_decomposition_summary.csv`: compact six-endpoint table joining
  the binary and regression diagnostics under common additive-gap semantics.
- `bbb_rank_reversal.json`: seed consistency, ensemble compound bootstrap, and
  novelty-stratified pairwise ranking diagnostics.
- `bbb_frontier_quintiles.csv`: BBB frontier-test prevalence, score, and AUROC
  within label-free frontier-score quintiles.

## Interpretation limits

Class-constant reweighting cannot change AUROC.  Therefore any AUROC gap is a
ranking/conditional-support phenomenon, not an arithmetic consequence of class
prevalence.  AUPRC and Brier score do change with prevalence; their additive
decompositions quantify that dependence descriptively.

For regression, outcome-bin standardization is a histogram sensitivity analysis,
not an identified covariate-shift correction.  In every endpoint, the residual
component still combines different test molecules, different fitted models,
support shift, and finite-sample variation.  It must not be described as a
causal decomposition.
"""
    (OUTPUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-reps", type=int, default=20_000)
    args = parser.parse_args()
    if args.bootstrap_reps < 1_000:
        raise ValueError("Use at least 1,000 bootstrap replicates")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    binary = binary_diagnostics()
    binary.to_csv(OUTPUT / "binary_prevalence_diagnostics_per_seed.csv", index=False)
    _aggregate_seed_rows(binary).to_csv(
        OUTPUT / "binary_prevalence_diagnostics_summary.csv", index=False
    )

    regression = regression_diagnostics(bins=10)
    regression.to_csv(OUTPUT / "regression_outcome_diagnostics_per_seed.csv", index=False)
    _aggregate_seed_rows(regression).to_csv(
        OUTPUT / "regression_outcome_diagnostics_summary.csv", index=False
    )
    compact_endpoint_summary(binary, regression).to_csv(
        OUTPUT / "endpoint_shift_decomposition_summary.csv", index=False
    )
    sensitivity = pd.concat(
        [regression_diagnostics(bins=bins) for bins in (5, 10, 20)],
        ignore_index=True,
    )
    _aggregate_seed_rows(sensitivity).to_csv(
        OUTPUT / "regression_reweighting_sensitivity.csv", index=False
    )

    loeo = leave_one_endpoint_out(args.bootstrap_reps)
    loeo.to_csv(OUTPUT / "leave_one_endpoint_out.csv", index=False)

    quintiles, bbb_payload = bbb_ensemble_diagnostics(binary, args.bootstrap_reps)
    quintiles.to_csv(OUTPUT / "bbb_frontier_quintiles.csv", index=False)
    (OUTPUT / "bbb_rank_reversal.json").write_text(
        json.dumps(bbb_payload, indent=2), encoding="utf-8"
    )

    _write_readme(args.bootstrap_reps)
    print(f"Wrote Priority 1 diagnostics to {OUTPUT}")
    print(loeo.to_string(index=False))
    print(json.dumps(bbb_payload, indent=2))


if __name__ == "__main__":
    main()
