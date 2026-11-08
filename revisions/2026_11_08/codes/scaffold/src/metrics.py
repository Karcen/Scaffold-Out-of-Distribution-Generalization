"""Task-aware metrics and calibration diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probability >= left) & (probability < right)
        if right == 1.0:
            mask |= probability == 1.0
        if not np.any(mask):
            continue
        confidence = float(probability[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += mask.mean() * abs(confidence - accuracy)
    return float(ece)


def classification_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= 0.5).astype(int)
    out = {
        "auroc": float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else float("nan"),
        "auprc": float(average_precision_score(y_true, probability)) if np.any(y_true == 1) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece": expected_calibration_error(y_true, probability),
    }
    return out


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    rho = spearmanr(y_true, prediction).statistic
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "spearman": float(rho),
        "r2": float(r2_score(y_true, prediction)),
    }


def evaluate(kind: str, y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if kind == "binary":
        return classification_metrics(y_true, prediction)
    if kind == "regression":
        return regression_metrics(y_true, prediction)
    raise ValueError(kind)

