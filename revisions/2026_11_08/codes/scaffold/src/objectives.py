"""Architecture-agnostic objectives for ordered molecular environments.

MV-FREX (Multi-View Frontier Risk Extrapolation) protects the tail of group
risks across several pre-registered notions of molecular similarity and
penalizes outward growth of risk along each ordered novelty coordinate.  The
predictor itself is deliberately conventional.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class ObjectiveConfig:
    method: str = "erm"
    tail_alpha: float = 0.4
    lambda_tail: float = 0.5
    lambda_slope: float = 0.2
    lambda_vrex: float = 1.0
    temperature: float = 0.1
    confidence_delta: float = 0.05
    loss_cap: float = 5.0


def _smooth_max(values: Tensor, temperature: float) -> Tensor:
    if values.numel() == 1:
        return values.squeeze()
    t = max(float(temperature), 1e-6)
    return t * torch.logsumexp(values / t, dim=0)


def _group_risks(losses: Tensor, groups: Tensor, n_groups: int) -> tuple[Tensor, Tensor]:
    risks = []
    counts = []
    for group_id in range(n_groups):
        mask = groups == group_id
        count = mask.sum()
        counts.append(count)
        if int(count) == 0:
            risks.append(torch.full((), float("nan"), device=losses.device))
        else:
            risks.append(losses[mask].mean())
    return torch.stack(risks), torch.stack(counts)


def _cvar_of_groups(risks: Tensor, alpha: float) -> Tensor:
    finite = risks[torch.isfinite(risks)]
    if finite.numel() == 0:
        raise RuntimeError("No finite environment risks in batch")
    k = max(1, math.ceil(float(alpha) * finite.numel()))
    return torch.topk(finite, k=k, largest=True).values.mean()


def _outward_slope(risks: Tensor, centers: Tensor, temperature: float) -> Tensor:
    candidates = []
    for left in range(risks.numel()):
        if not torch.isfinite(risks[left]):
            continue
        for right in range(left + 1, risks.numel()):
            if not torch.isfinite(risks[right]):
                continue
            distance = torch.clamp(centers[right] - centers[left], min=1e-4)
            candidates.append(torch.relu(risks[right] - risks[left]) / distance)
    if not candidates:
        return torch.zeros((), device=risks.device)
    return _smooth_max(torch.stack(candidates), temperature)


def compute_objective(
    per_sample_loss: Tensor,
    view_groups: Tensor,
    group_centers: Tensor,
    full_group_counts: Tensor,
    config: ObjectiveConfig,
) -> tuple[Tensor, dict[str, float]]:
    """Return the scalar training objective and detached diagnostics.

    Parameters
    ----------
    per_sample_loss:
        Unreduced task loss, shape ``[batch]``.
    view_groups:
        Ordered novelty-bin identifiers, shape ``[n_views, batch]``.
    group_centers:
        Mean novelty coordinate per view/bin, shape ``[n_views, n_bins]``.
    full_group_counts:
        Training-set group sizes used for simultaneous Hoeffding corrections.
    """

    method = config.method.lower()
    erm = per_sample_loss.mean()
    if method == "erm":
        return erm, {"erm": float(erm.detach())}

    robust_losses = torch.clamp(per_sample_loss, max=config.loss_cap) / config.loss_cap
    n_views, n_bins = group_centers.shape
    risks_by_view = []
    for view in range(n_views):
        risks, _ = _group_risks(robust_losses, view_groups[view], n_bins)
        risks_by_view.append(risks)

    primary = risks_by_view[0]
    if method == "vrex":
        finite = primary[torch.isfinite(primary)]
        penalty = finite.var(unbiased=False)
        objective = erm + config.lambda_vrex * penalty
        return objective, {"erm": float(erm.detach()), "vrex": float(penalty.detach())}

    if method == "groupdro":
        penalty = _smooth_max(primary[torch.isfinite(primary)], config.temperature)
        objective = erm + config.lambda_tail * penalty
        return objective, {"erm": float(erm.detach()), "worst_group": float(penalty.detach())}

    if method == "cvar":
        k = max(1, math.ceil(config.tail_alpha * robust_losses.numel()))
        penalty = torch.topk(robust_losses, k=k, largest=True).values.mean()
        objective = erm + config.lambda_tail * penalty
        return objective, {"erm": float(erm.detach()), "sample_cvar": float(penalty.detach())}

    if method not in {"mvfrex", "mvfrex_no_slope", "frex_single_view"}:
        raise ValueError(f"Unknown objective method: {config.method}")

    selected_views = 1 if method == "frex_single_view" else n_views
    simultaneous = max(2 * selected_views * n_bins / config.confidence_delta, 1.0)
    tail_terms = []
    slope_terms = []
    for view in range(selected_views):
        counts = torch.clamp(full_group_counts[view].to(per_sample_loss.device), min=1.0)
        correction = torch.sqrt(torch.log(torch.tensor(simultaneous, device=per_sample_loss.device)) / (2.0 * counts))
        upper = risks_by_view[view] + correction
        tail_terms.append(_cvar_of_groups(upper, config.tail_alpha))
        slope_terms.append(_outward_slope(upper, group_centers[view], config.temperature))

    worst_tail = _smooth_max(torch.stack(tail_terms), config.temperature)
    worst_slope = _smooth_max(torch.stack(slope_terms), config.temperature)
    slope_weight = 0.0 if method == "mvfrex_no_slope" else config.lambda_slope
    objective = erm + config.lambda_tail * worst_tail + slope_weight * worst_slope
    return objective, {
        "erm": float(erm.detach()),
        "worst_view_cvar_ucb": float(worst_tail.detach()),
        "worst_view_slope": float(worst_slope.detach()),
    }

