"""Training-only construction of ordered novelty environments.

Four pre-registered molecular views are reduced independently.  Novelty is the
average of a global-radius term and distance to the local training support.  No
labels are used.  Bins are fitted on training molecules only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler


VIEW_NAMES = ("ecfp", "maccs", "atompair", "physchem")


@dataclass
class EnvironmentResult:
    groups: np.ndarray  # [n_views, n_samples]
    coordinates: np.ndarray  # [n_views, n_samples]
    group_centers: np.ndarray  # [n_views, n_bins]
    train_group_counts: np.ndarray  # [n_views, n_bins]
    thresholds: np.ndarray  # [n_views, n_bins - 1]
    view_names: tuple[str, ...]


def _safe_scale(raw: np.ndarray, reference: np.ndarray) -> np.ndarray:
    median = float(np.median(reference))
    mad = float(np.median(np.abs(reference - median)))
    scale = max(1.4826 * mad, 1e-6)
    return (raw - median) / scale


def _embed_view(features: np.ndarray, train_idx: np.ndarray, seed: int, view: str) -> np.ndarray:
    train = features[train_idx]
    if view == "physchem":
        scaler = RobustScaler(quantile_range=(10, 90)).fit(train)
        return np.clip(scaler.transform(features), -20.0, 20.0).astype(np.float32)

    max_components = min(32, train.shape[0] - 1, train.shape[1] - 1)
    n_components = max(2, max_components)
    svd = TruncatedSVD(n_components=n_components, random_state=seed, n_iter=5)
    train_embedding = svd.fit_transform(train)
    all_embedding = svd.transform(features)
    scaler = StandardScaler().fit(train_embedding)
    return np.clip(scaler.transform(all_embedding), -10.0, 10.0).astype(np.float32)


def build_novelty_environments(
    feature_views: dict[str, np.ndarray],
    train_idx: np.ndarray,
    seed: int,
    n_bins: int = 4,
    neighbors: int = 5,
) -> EnvironmentResult:
    """Create ordered environments without reading labels or held-out outcomes."""

    n_samples = next(iter(feature_views.values())).shape[0]
    all_groups = []
    all_coordinates = []
    all_centers = []
    all_counts = []
    all_thresholds = []

    for view in VIEW_NAMES:
        embedding = _embed_view(feature_views[view], train_idx, seed, view)
        train_embedding = embedding[train_idx]
        center = np.median(train_embedding, axis=0)
        radius = np.linalg.norm(embedding - center, axis=1)

        k_train = min(neighbors + 1, len(train_idx))
        nn = NearestNeighbors(
            n_neighbors=k_train, metric="euclidean", algorithm="ball_tree", n_jobs=1
        )
        nn.fit(train_embedding)
        distances, _ = nn.kneighbors(embedding)
        # Training rows contain themselves at distance zero.  Held-out rows do
        # not, so use mean distances 1:k for train and 0:k-1 for held-out.
        local = distances[:, : max(1, k_train - 1)].mean(axis=1)
        train_position = {int(idx): pos for pos, idx in enumerate(train_idx)}
        for idx in train_idx:
            local[idx] = distances[idx, 1:k_train].mean() if k_train > 1 else 0.0

        radius_z = _safe_scale(radius, radius[train_idx])
        local_z = _safe_scale(local, local[train_idx])
        coordinate = 0.5 * radius_z + 0.5 * local_z
        coordinate = np.clip(coordinate, -8.0, 20.0)

        quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        thresholds = np.quantile(coordinate[train_idx], quantiles)
        # Degenerate ties are made strictly ordered without changing practical bins.
        thresholds = np.maximum.accumulate(thresholds + np.arange(len(thresholds)) * 1e-7)
        groups = np.digitize(coordinate, thresholds, right=False).astype(np.int64)
        centers = np.asarray(
            [
                coordinate[train_idx][groups[train_idx] == group].mean()
                if np.any(groups[train_idx] == group)
                else float(group)
                for group in range(n_bins)
            ],
            dtype=np.float32,
        )
        counts = np.bincount(groups[train_idx], minlength=n_bins).astype(np.float32)

        all_groups.append(groups)
        all_coordinates.append(coordinate.astype(np.float32))
        all_centers.append(centers)
        all_counts.append(counts)
        all_thresholds.append(thresholds.astype(np.float32))

    return EnvironmentResult(
        groups=np.stack(all_groups),
        coordinates=np.stack(all_coordinates),
        group_centers=np.stack(all_centers),
        train_group_counts=np.stack(all_counts),
        thresholds=np.stack(all_thresholds),
        view_names=VIEW_NAMES,
    )
