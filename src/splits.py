"""Leakage-controlled scaffold and label-free frontier splits."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from .chem import descriptor_vector, murcko_scaffold


SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SplitResult:
    assignment: np.ndarray
    scaffold: np.ndarray
    frontier_score: np.ndarray
    metadata: dict


def _group_indices(keys: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        groups[str(key)].append(i)
    return groups


def _deterministic_acyclic_clusters(values: np.ndarray, target_size: int = 40) -> np.ndarray:
    """Small deterministic Lloyd clustering without an OpenMP dependency."""

    n = len(values)
    if n <= 50:
        return np.zeros(n, dtype=int)
    k = min(64, int(np.ceil(n / target_size)))
    centers = []
    median = np.median(values, axis=0)
    first = int(np.argmax(np.sum((values - median) ** 2, axis=1)))
    centers.append(values[first])
    min_distance = np.sum((values - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        index = int(np.argmax(min_distance))
        centers.append(values[index])
        min_distance = np.minimum(min_distance, np.sum((values - values[index]) ** 2, axis=1))
    centers = np.stack(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(12):
        distance = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        updated = np.argmin(distance, axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for cluster in range(k):
            members = values[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)

    # Hard-cap unusually large Lloyd cells.  Sorting by distance to the cell
    # center makes each deterministic subcell locally coherent.
    capped = np.empty(n, dtype=int)
    next_label = 0
    for cluster in range(k):
        indices = np.where(labels == cluster)[0]
        if not len(indices):
            continue
        distance = np.sum((values[indices] - centers[cluster]) ** 2, axis=1)
        ordered = indices[np.argsort(distance, kind="mergesort")]
        for start in range(0, len(ordered), 50):
            capped[ordered[start : start + 50]] = next_label
            next_label += 1
    return capped


def _cluster_acyclic_scaffolds(scaffolds: np.ndarray, scaled: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace only the empty-scaffold supergroup by deterministic descriptor cells."""

    scaffolds = scaffolds.copy()
    acyclic_idx = np.where(scaffolds == "__ACYCLIC__")[0]
    n_clusters = 1
    if len(acyclic_idx) > 50:
        labels = _deterministic_acyclic_clusters(scaled[acyclic_idx])
        n_clusters = int(labels.max()) + 1
        for idx, label in zip(acyclic_idx, labels):
            scaffolds[idx] = f"__ACYCLIC_CLUSTER_{int(label):03d}"
    return scaffolds, n_clusters


def _allocate_groups(
    groups: list[tuple[str, list[int]]],
    n: int,
    fractions: tuple[float, float, float],
) -> np.ndarray:
    targets = np.asarray(fractions, dtype=float) * n
    counts = np.zeros(3, dtype=int)
    assignment = np.empty(n, dtype=object)
    for _, idx in groups:
        size = len(idx)
        remaining = targets - counts
        feasible = np.where(remaining >= size)[0]
        if len(feasible):
            split_idx = int(feasible[np.argmax(remaining[feasible])])
        else:
            split_idx = int(np.argmax(remaining))
        assignment[idx] = SPLIT_NAMES[split_idx]
        counts[split_idx] += size
    return assignment


def scaffold_split(
    smiles: Sequence[str],
    seed: int,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    cluster_acyclic: bool = False,
) -> SplitResult:
    """Size-aware Bemis-Murcko split with seeded tie breaking."""

    scaffolds = np.asarray([murcko_scaffold(s) for s in smiles], dtype=object)
    acyclic_clusters = 1
    if cluster_acyclic:
        descriptors = np.stack([descriptor_vector(s) for s in smiles])
        if not np.isfinite(descriptors).all():
            med = np.nanmedian(descriptors, axis=0)
            descriptors = np.where(np.isfinite(descriptors), descriptors, med)
        scaled = RobustScaler(quantile_range=(10, 90)).fit_transform(descriptors)
        scaled = np.clip(scaled, -20.0, 20.0)
        scaffolds, acyclic_clusters = _cluster_acyclic_scaffolds(scaffolds, scaled)
    grouped = _group_indices(scaffolds)
    rng = np.random.default_rng(seed)
    items = list(grouped.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)
    assignment = _allocate_groups(items, len(smiles), fractions)
    return SplitResult(
        assignment=assignment,
        scaffold=scaffolds,
        frontier_score=np.zeros(len(smiles), dtype=float),
        metadata={
            "kind": "scaffold",
            "seed": seed,
            "fractions": fractions,
            "acyclic_policy": (
                "physchem_deterministic_lloyd_max50" if cluster_acyclic else "literal_empty_scaffold"
            ),
            "acyclic_clusters": acyclic_clusters,
        },
    )


def frontier_split(
    smiles: Sequence[str],
    seed: int,
    fractions: tuple[float, float, float] = (0.7, 0.1, 0.2),
    neighbors: int = 10,
) -> SplitResult:
    """Label-free structural-frontier split at scaffold-group resolution.

    Each scaffold is represented by the median of robustly scaled RDKit
    descriptors.  Its frontier score averages two rank statistics: global
    robust radius and local k-nearest-neighbor sparsity.  Highest-scoring
    scaffolds form the test frontier, then validation, then training.  Labels
    are never inspected.
    """

    descriptors = np.stack([descriptor_vector(s) for s in smiles])
    if not np.isfinite(descriptors).all():
        med = np.nanmedian(descriptors, axis=0)
        descriptors = np.where(np.isfinite(descriptors), descriptors, med)
    scaled = RobustScaler(quantile_range=(10, 90)).fit_transform(descriptors)
    scaled = np.clip(scaled, -20.0, 20.0)
    scaffolds = np.asarray([murcko_scaffold(s) for s in smiles], dtype=object)
    # A literal empty Murcko scaffold can be a heterogeneous supergroup (29.7%
    # of AqSolDB after harmonization).  For the *alternative* frontier benchmark,
    # partition only that supergroup with label-free physicochemical clustering.
    # The standard scaffold benchmark intentionally retains the literal policy.
    scaffolds, acyclic_clusters = _cluster_acyclic_scaffolds(scaffolds, scaled)
    grouped = _group_indices(scaffolds)
    keys = list(grouped)
    centers = np.stack([np.median(scaled[grouped[k]], axis=0) for k in keys])
    center0 = np.median(centers, axis=0)
    radius = np.linalg.norm(centers - center0, axis=1)
    k = min(neighbors + 1, len(keys))
    neighbor_model = NearestNeighbors(
        n_neighbors=k, metric="euclidean", algorithm="ball_tree", n_jobs=1
    )
    neighbor_model.fit(centers)
    distances, _ = neighbor_model.kneighbors(centers)
    local = distances[:, 1:k].mean(axis=1)

    def rank01(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks / max(len(values) - 1, 1)

    score = 0.5 * rank01(radius) + 0.5 * rank01(local)
    rng = np.random.default_rng(seed)
    tie_noise = rng.uniform(0.0, 1e-9, len(keys))
    order = np.argsort(-(score + tie_noise), kind="mergesort")

    # Allocate from the outer frontier inward: test, validation, train.
    targets = {"test": fractions[2] * len(smiles), "val": fractions[1] * len(smiles)}
    assignment = np.full(len(smiles), "train", dtype=object)
    group_split: dict[str, str] = {}
    counts = {"test": 0, "val": 0}
    current = "test"
    for j in order:
        key = keys[j]
        if current == "test" and counts["test"] >= targets["test"]:
            current = "val"
        if current == "val" and counts["val"] >= targets["val"]:
            current = "train"
        group_split[key] = current
        assignment[grouped[key]] = current
        if current in counts:
            counts[current] += len(grouped[key])

    score_map = {key: float(score[j]) for j, key in enumerate(keys)}
    per_row_score = np.asarray([score_map[s] for s in scaffolds], dtype=float)
    return SplitResult(
        assignment=assignment,
        scaffold=scaffolds,
        frontier_score=per_row_score,
        metadata={
            "kind": "frontier",
            "seed": seed,
            "fractions": fractions,
            "neighbors": neighbors,
            "label_free": True,
            "acyclic_policy": "physchem_deterministic_lloyd_max50",
            "acyclic_clusters": acyclic_clusters,
        },
    )


def assert_split_integrity(result: SplitResult) -> None:
    present = set(result.assignment.tolist())
    if present != set(SPLIT_NAMES):
        raise AssertionError(f"Missing split(s): {set(SPLIT_NAMES) - present}")
    owner: dict[str, str] = {}
    for scaffold, split in zip(result.scaffold, result.assignment):
        if scaffold in owner and owner[scaffold] != split:
            raise AssertionError(f"Scaffold leakage for {scaffold}")
        owner[scaffold] = split
