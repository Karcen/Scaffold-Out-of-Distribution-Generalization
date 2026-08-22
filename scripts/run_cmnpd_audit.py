#!/usr/bin/env python3
"""Audit experimental models on the CMNPD natural-product teacher domain.

Agreement with CMNPD's Pipeline Pilot outputs is explicitly not biological
accuracy.  The audit quantifies structural novelty and teacher/student divergence.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    roc_auc_score,
)
import torch

if "." not in sys.path:
    sys.path.insert(0, ".")

from src.chem import featurize_many
from src.data import TASKS, cmnpd_teacher_target, load_cmnpd, load_tdc_task
from src.environments import VIEW_NAMES, build_novelty_environments
from src.models import FingerprintMLP


RESULTS = Path("results")
CACHE = Path("cache")


def load_feature_cache(task_name: str) -> dict[str, np.ndarray]:
    loaded = np.load(CACHE / f"{task_name.lower()}_features.npz", allow_pickle=False)
    return {view: loaded[view] for view in VIEW_NAMES}


def cmnpd_feature_cache(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    path = CACHE / "cmnpd_features.npz"
    smiles = np.asarray(frame["SMILES"].astype(str).tolist(), dtype=str)
    if path.exists():
        loaded = np.load(path, allow_pickle=False)
        if np.array_equal(loaded["smiles"].astype(str), smiles):
            return {view: loaded[view] for view in VIEW_NAMES}
    print(f"featurizing CMNPD ({len(frame):,} molecules)", flush=True)
    views = {
        "ecfp": featurize_many(smiles, "ecfp", n_bits=1024),
        "maccs": featurize_many(smiles, "maccs"),
        "atompair": featurize_many(smiles, "atompair", n_bits=1024),
        "physchem": featurize_many(smiles, "physchem"),
    }
    np.savez_compressed(path, smiles=smiles, **views)
    return views


@torch.no_grad()
def predict_checkpoint(path: Path, x: np.ndarray) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    train_config = payload["train_config"]
    model = FingerprintMLP(payload["input_dim"], train_config["hidden_dim"], train_config["dropout"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    outputs = []
    for start in range(0, len(x), 4096):
        tensor = torch.from_numpy(x[start : start + 4096].astype(np.float32))
        outputs.append(model(tensor).numpy())
    output = np.concatenate(outputs)
    if payload["kind"] == "binary":
        return 1.0 / (1.0 + np.exp(-np.clip(output, -30.0, 30.0)))
    return output * float(payload["target_scale"]) + float(payload["target_mean"])


def agreement_metrics(task: str, teacher: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if task == "Solubility":
        robust = np.isfinite(teacher) & (np.abs(teacher) <= 10)
        return {
            "mae_full": float(mean_absolute_error(teacher, prediction)),
            "mae_abs_teacher_le_10": float(mean_absolute_error(teacher[robust], prediction[robust])),
            "spearman": float(spearmanr(teacher, prediction).statistic),
            "robust_n": int(robust.sum()),
        }
    threshold = 90.0 if task == "PPBR" else 0.5
    hard = (prediction >= threshold).astype(int)
    return {
        "auroc_agreement": float(roc_auc_score(teacher, prediction))
        if len(np.unique(teacher)) == 2
        else float("nan"),
        "auprc_agreement": float(average_precision_score(teacher, prediction))
        if np.any(teacher == 1)
        else float("nan"),
        "accuracy_agreement": float(accuracy_score(teacher, hard)),
        "balanced_accuracy_agreement": float(balanced_accuracy_score(teacher, hard)),
    }


def main() -> None:
    cmnpd = load_cmnpd(Path("CMNPD_1.0_calc_prop.csv"))
    cmnpd_views = cmnpd_feature_cache(cmnpd)
    prediction_blocks = []
    metric_rows = []
    stratum_rows = []
    seed_stratum_rows = []
    shift_rows = []
    coverage_rows = []
    bootstrap_rows = []

    for spec in TASKS:
        print(f"audit {spec.name}", flush=True)
        tdc_frame = load_tdc_task(spec, Path("data/external"))
        tdc_views = load_feature_cache(spec.name)
        split_path = RESULTS / "splits" / f"{spec.name}__frontier__seed0.csv.gz"
        manifest = pd.read_csv(split_path)
        train_idx = np.where(manifest["partition"].to_numpy() == "train")[0]
        combined_views = {
            view: np.concatenate([tdc_views[view], cmnpd_views[view]], axis=0)
            for view in VIEW_NAMES
        }
        environments = build_novelty_environments(combined_views, train_idx=train_idx, seed=0)
        offset = len(tdc_frame)
        cmnpd_coordinates = environments.coordinates[:, offset:]
        cmnpd_groups = environments.groups[:, offset:]
        worst_novelty = np.max(cmnpd_coordinates, axis=0)
        quintile_edges = np.quantile(worst_novelty, [0.2, 0.4, 0.6, 0.8])
        novelty_quintile = np.digitize(worst_novelty, quintile_edges).astype(int)

        for view_index, view in enumerate(VIEW_NAMES):
            train_coordinate = environments.coordinates[view_index, train_idx]
            target_coordinate = cmnpd_coordinates[view_index]
            shift_rows.append(
                {
                    "task": spec.name,
                    "view": view,
                    "train_median": float(np.median(train_coordinate)),
                    "cmnpd_median": float(np.median(target_coordinate)),
                    "cmnpd_fraction_beyond_train_max": float(np.mean(target_coordinate > np.max(train_coordinate))),
                    "wasserstein_distance": float(wasserstein_distance(train_coordinate, target_coordinate)),
                }
            )

        ensemble_predictions = {}
        all_seed_predictions = {}
        for method in ["ERM", "MV-FREX"]:
            seed_predictions = []
            for seed in range(5):
                model_path = (
                    RESULTS
                    / "runs"
                    / f"{spec.name}__frontier__seed{seed}__{method}__mlp"
                    / "model.pt"
                )
                seed_predictions.append(predict_checkpoint(model_path, cmnpd_views["ecfp"]))
            all_seed_predictions[method] = seed_predictions
            ensemble_predictions[method] = np.mean(seed_predictions, axis=0)

        teacher, mask = cmnpd_teacher_target(cmnpd, spec.name)
        for quintile in range(5):
            quintile_mask = novelty_quintile == quintile
            available_mask = mask & quintile_mask
            robust_mask = (
                available_mask & (np.abs(teacher) <= 10)
                if spec.name == "Solubility"
                else available_mask
            )
            coverage_rows.append(
                {
                    "task": spec.name,
                    "novelty_quintile": quintile,
                    "total_n": int(quintile_mask.sum()),
                    "teacher_available_n": int(available_mask.sum()),
                    "teacher_available_fraction": float(
                        available_mask.sum() / max(quintile_mask.sum(), 1)
                    ),
                    "analysis_retained_n": int(robust_mask.sum()),
                    "analysis_retained_fraction": float(
                        robust_mask.sum() / max(quintile_mask.sum(), 1)
                    ),
                }
            )
        for method, seed_predictions in all_seed_predictions.items():
            for seed, prediction in enumerate(seed_predictions):
                for quintile in range(5):
                    block_mask = mask & (novelty_quintile == quintile)
                    block_metrics = agreement_metrics(
                        spec.name, teacher[block_mask], prediction[block_mask]
                    )
                    for metric, value in block_metrics.items():
                        seed_stratum_rows.append(
                            {
                                "task": spec.name,
                                "method": method,
                                "seed": seed,
                                "novelty_quintile": quintile,
                                "metric": metric,
                                "value": value,
                                "n": int(block_mask.sum()),
                            }
                        )
        for method, prediction in ensemble_predictions.items():
            metrics = agreement_metrics(spec.name, teacher[mask], prediction[mask])
            for metric, value in metrics.items():
                metric_rows.append(
                    {
                        "task": spec.name,
                        "method": method,
                        "metric": metric,
                        "value": value,
                        "n": int(mask.sum()),
                        "interpretation": "legacy_teacher_agreement_not_biological_accuracy",
                    }
                )
            for quintile in range(5):
                block_mask = mask & (novelty_quintile == quintile)
                block_metrics = agreement_metrics(spec.name, teacher[block_mask], prediction[block_mask])
                for metric, value in block_metrics.items():
                    stratum_rows.append(
                        {
                            "task": spec.name,
                            "method": method,
                            "novelty_quintile": quintile,
                            "metric": metric,
                            "value": value,
                            "n": int(block_mask.sum()),
                        }
                    )
                if method == "ERM" and spec.kind == "binary":
                    y_block = teacher[block_mask]
                    p_block = prediction[block_mask]
                    positive = np.where(y_block == 1)[0]
                    negative = np.where(y_block == 0)[0]
                    if len(positive) and len(negative):
                        task_seed = sum((index + 1) * ord(char) for index, char in enumerate(spec.name))
                        rng = np.random.default_rng(20260711 + 101 * task_seed + quintile)
                        draws = []
                        for _ in range(5000):
                            sampled = np.concatenate(
                                [
                                    rng.choice(positive, size=len(positive), replace=True),
                                    rng.choice(negative, size=len(negative), replace=True),
                                ]
                            )
                            draws.append(roc_auc_score(y_block[sampled], p_block[sampled]))
                        bootstrap_rows.append(
                            {
                                "task": spec.name,
                                "novelty_quintile": quintile,
                                "metric": "auroc_agreement",
                                "value": float(roc_auc_score(y_block, p_block)),
                                "ci95_low": float(np.quantile(draws, 0.025)),
                                "ci95_high": float(np.quantile(draws, 0.975)),
                                "n": int(len(y_block)),
                                "positive_n": int(len(positive)),
                                "negative_n": int(len(negative)),
                                "bootstrap_replicates": 5000,
                            }
                        )

        block = pd.DataFrame(
            {
                "task": spec.name,
                "row_index": np.arange(len(cmnpd)),
                "compound_id": cmnpd["COMPOUND_ID"],
                "smiles": cmnpd["SMILES"],
                "teacher": teacher,
                "teacher_available": mask,
                "prediction_erm": ensemble_predictions["ERM"],
                "prediction_mvfrex": ensemble_predictions["MV-FREX"],
                "worst_novelty": worst_novelty,
                "novelty_quintile": novelty_quintile,
            }
        )
        for view_index, view in enumerate(VIEW_NAMES):
            block[f"novelty_{view}"] = cmnpd_coordinates[view_index]
            block[f"environment_{view}"] = cmnpd_groups[view_index]
        prediction_blocks.append(block)

    pd.DataFrame(metric_rows).to_csv(RESULTS / "cmnpd_teacher_agreement.csv", index=False)
    pd.DataFrame(stratum_rows).to_csv(RESULTS / "cmnpd_teacher_agreement_by_novelty.csv", index=False)
    pd.DataFrame(seed_stratum_rows).to_csv(
        RESULTS / "cmnpd_teacher_agreement_by_novelty_seed.csv", index=False
    )
    pd.DataFrame(shift_rows).to_csv(RESULTS / "cmnpd_shift_summary.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(
        RESULTS / "cmnpd_teacher_coverage_by_novelty.csv", index=False
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        RESULTS / "cmnpd_teacher_agreement_compound_bootstrap.csv", index=False
    )
    pd.concat(prediction_blocks, ignore_index=True).to_csv(
        RESULTS / "cmnpd_audit_predictions.csv.gz", index=False, compression="gzip"
    )
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(pd.DataFrame(shift_rows).to_string(index=False))


if __name__ == "__main__":
    main()
