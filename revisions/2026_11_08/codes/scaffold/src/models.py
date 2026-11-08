"""Conventional predictor and reproducible training loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import random

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import evaluate
from .objectives import ObjectiveConfig, compute_objective


@dataclass(frozen=True)
class TrainConfig:
    hidden_dim: int = 64
    dropout: float = 0.10
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 30
    patience: int = 6
    gradient_clip: float = 5.0
    seed: int = 0


class FingerprintMLP(nn.Module):
    """A deliberately ordinary two-layer MLP shared by all objectives."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        if hidden_dim <= 0:
            self.network = nn.Linear(input_dim, 1)
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x).squeeze(-1)


@dataclass
class TrainResult:
    state_dict: dict
    history: list[dict]
    best_epoch: int
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    val_prediction: np.ndarray
    test_prediction: np.ndarray
    target_mean: float
    target_scale: float


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _selection_score(kind: str, primary_metric: str, metrics: dict[str, float]) -> float:
    value = metrics.get(primary_metric, float("nan"))
    if not np.isfinite(value):
        fallback = metrics.get("brier", float("nan"))
        return -fallback if np.isfinite(fallback) else -float("inf")
    return -value if primary_metric in {"mae", "rmse", "brier", "ece"} else value


@torch.no_grad()
def _predict(model: nn.Module, x: np.ndarray, kind: str, mean: float, scale: float) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(x.astype(np.float32))
    output = model(tensor).cpu().numpy()
    if kind == "binary":
        return 1.0 / (1.0 + np.exp(-np.clip(output, -30.0, 30.0)))
    return output * scale + mean


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    kind: str,
    primary_metric: str,
    view_groups: np.ndarray,
    group_centers: np.ndarray,
    train_group_counts: np.ndarray,
    objective_config: ObjectiveConfig,
    train_config: TrainConfig,
) -> TrainResult:
    _seed_everything(train_config.seed)
    device = torch.device("cpu")
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    if kind == "regression":
        target_mean = float(np.mean(y[train_idx]))
        target_scale = float(np.std(y[train_idx]))
        target_scale = max(target_scale, 1e-6)
        y_train_space = (y - target_mean) / target_scale
    else:
        target_mean, target_scale = 0.0, 1.0
        y_train_space = y

    dataset = TensorDataset(
        torch.from_numpy(x[train_idx]),
        torch.from_numpy(y_train_space[train_idx]),
        torch.from_numpy(view_groups[:, train_idx].T.astype(np.int64)),
    )
    generator = torch.Generator().manual_seed(train_config.seed)
    batch_size = min(train_config.batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, drop_last=False)

    model = FingerprintMLP(x.shape[1], train_config.hidden_dim, train_config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )
    centers_tensor = torch.from_numpy(group_centers.astype(np.float32)).to(device)
    counts_tensor = torch.from_numpy(train_group_counts.astype(np.float32)).to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_score = -float("inf")
    best_epoch = -1
    epochs_without_gain = 0
    history: list[dict] = []

    for epoch in range(train_config.max_epochs):
        model.train()
        epoch_losses = []
        for batch_x, batch_y, batch_groups in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_groups = batch_groups.to(device).T
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x)
            if kind == "binary":
                per_sample = nn.functional.binary_cross_entropy_with_logits(output, batch_y, reduction="none")
            else:
                per_sample = nn.functional.smooth_l1_loss(output, batch_y, reduction="none", beta=1.0)
            objective, diagnostics = compute_objective(
                per_sample, batch_groups, centers_tensor, counts_tensor, objective_config
            )
            if not torch.isfinite(objective):
                raise FloatingPointError(f"Non-finite objective at epoch {epoch}: {diagnostics}")
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip)
            optimizer.step()
            epoch_losses.append(float(objective.detach()))

        val_prediction = _predict(model, x[val_idx], kind, target_mean, target_scale)
        val_metrics = evaluate(kind, y[val_idx], val_prediction)
        score = _selection_score(kind, primary_metric, val_metrics)
        history.append(
            {
                "epoch": epoch,
                "objective": float(np.mean(epoch_losses)),
                "selection_score": score,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= train_config.patience:
                break

    model.load_state_dict(best_state)
    val_prediction = _predict(model, x[val_idx], kind, target_mean, target_scale)
    test_prediction = _predict(model, x[test_idx], kind, target_mean, target_scale)
    val_metrics = evaluate(kind, y[val_idx], val_prediction)
    test_metrics = evaluate(kind, y[test_idx], test_prediction)
    return TrainResult(
        state_dict={key: value.cpu() for key, value in best_state.items()},
        history=history,
        best_epoch=best_epoch,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_prediction=val_prediction,
        test_prediction=test_prediction,
        target_mean=target_mean,
        target_scale=target_scale,
    )
