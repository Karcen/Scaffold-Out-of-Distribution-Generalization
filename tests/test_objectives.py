from __future__ import annotations

import torch

from src.objectives import ObjectiveConfig, compute_objective


def test_all_objectives_are_finite_and_differentiable() -> None:
    logits = torch.linspace(-1.0, 1.0, 32, requires_grad=True)
    targets = (torch.arange(32) % 2).float()
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    groups = torch.stack([torch.arange(32) % 4, (torch.arange(32) // 2) % 4])
    centers = torch.tensor([[-1.0, -0.3, 0.4, 1.2], [-0.8, -0.1, 0.5, 1.5]])
    counts = torch.full((2, 4), 8.0)
    for method in ["erm", "vrex", "groupdro", "cvar", "mvfrex", "mvfrex_no_slope", "frex_single_view"]:
        objective, _ = compute_objective(
            losses, groups, centers, counts, ObjectiveConfig(method=method)
        )
        assert torch.isfinite(objective)
        gradient = torch.autograd.grad(objective, logits, retain_graph=True)[0]
        assert torch.isfinite(gradient).all()

