"""Small message-passing graph control used for the major-revision audit.

The main paper deliberately uses a fixed Morgan fingerprint.  This module adds
an independently trained three-layer GINE encoder so that the null effect of the
frontier penalty is not assessed only through a fixed representation.  No
pretrained weights or test labels are used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import random
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from rdkit import Chem

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool

from .metrics import evaluate
from .objectives import ObjectiveConfig, compute_objective


ATOM_CARDINALITIES = (119, 8, 11, 9, 9, 2, 7)
BOND_CARDINALITIES = (5, 7, 2, 2)


def _bounded(value: int, upper: int) -> int:
    return max(0, min(int(value), upper - 1))


def _hybridization_index(atom: Chem.Atom) -> int:
    mapping = {
        Chem.rdchem.HybridizationType.UNSPECIFIED: 0,
        Chem.rdchem.HybridizationType.S: 1,
        Chem.rdchem.HybridizationType.SP: 2,
        Chem.rdchem.HybridizationType.SP2: 3,
        Chem.rdchem.HybridizationType.SP3: 4,
        Chem.rdchem.HybridizationType.SP3D: 5,
        Chem.rdchem.HybridizationType.SP3D2: 6,
        Chem.rdchem.HybridizationType.OTHER: 7,
    }
    return mapping.get(atom.GetHybridization(), 8)


def _bond_type_index(bond: Chem.Bond) -> int:
    mapping = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    return mapping.get(bond.GetBondType(), 4)


def smiles_to_graph(smiles: str, row_index: int, y: float, groups: np.ndarray) -> Data:
    """Convert one valid SMILES string into categorical atom/bond tensors."""

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES at row {row_index}: {smiles!r}")

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(
            [
                _bounded(atom.GetAtomicNum(), ATOM_CARDINALITIES[0]),
                _bounded(atom.GetDegree(), ATOM_CARDINALITIES[1]),
                _bounded(atom.GetFormalCharge() + 5, ATOM_CARDINALITIES[2]),
                _bounded(int(atom.GetChiralTag()), ATOM_CARDINALITIES[3]),
                _hybridization_index(atom),
                int(atom.GetIsAromatic()),
                _bounded(atom.GetTotalNumHs(), ATOM_CARDINALITIES[6]),
            ]
        )

    edges: list[tuple[int, int]] = []
    edge_features: list[list[int]] = []
    for bond in mol.GetBonds():
        feature = [
            _bond_type_index(bond),
            _bounded(int(bond.GetStereo()), BOND_CARDINALITIES[1]),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
        ]
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([(begin, end), (end, begin)])
        edge_features.extend([feature, feature])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(BOND_CARDINALITIES)), dtype=torch.long)

    return Data(
        x=torch.tensor(atom_features, dtype=torch.long),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([float(y)], dtype=torch.float32),
        env_groups=torch.tensor(np.asarray(groups, dtype=np.int64)[None, :], dtype=torch.long),
        row_index=torch.tensor([int(row_index)], dtype=torch.long),
    )


def build_graphs(smiles: Sequence[str], y: np.ndarray, view_groups: np.ndarray) -> list[Data]:
    """Build one PyG graph per harmonized molecule in source-row order."""

    return [
        smiles_to_graph(smile, index, float(y[index]), view_groups[:, index])
        for index, smile in enumerate(smiles)
    ]


@dataclass(frozen=True)
class GraphTrainConfig:
    hidden_dim: int = 128
    layers: int = 3
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 30
    patience: int = 6
    gradient_clip: float = 5.0
    seed: int = 0


class MolecularGINE(nn.Module):
    """Three-layer edge-aware message-passing encoder with graph pooling."""

    def __init__(self, config: GraphTrainConfig):
        super().__init__()
        h = config.hidden_dim
        self.dropout = config.dropout
        self.atom_embeddings = nn.ModuleList([nn.Embedding(size, h) for size in ATOM_CARDINALITIES])
        self.bond_embeddings = nn.ModuleList([nn.Embedding(size, h) for size in BOND_CARDINALITIES])
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(config.layers):
            update = nn.Sequential(nn.Linear(h, 2 * h), nn.ReLU(), nn.Linear(2 * h, h))
            self.convs.append(GINEConv(update, train_eps=True, edge_dim=h))
            self.norms.append(nn.LayerNorm(h))
        self.head = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, 1),
        )

    def forward(self, batch: Data) -> Tensor:
        h = sum(embedding(batch.x[:, field]) for field, embedding in enumerate(self.atom_embeddings))
        edge = sum(
            embedding(batch.edge_attr[:, field])
            for field, embedding in enumerate(self.bond_embeddings)
        )
        for conv, norm in zip(self.convs, self.norms):
            updated = conv(h, batch.edge_index, edge)
            h = norm(h + nn.functional.dropout(updated, p=self.dropout, training=self.training))
            h = nn.functional.relu(h)
        pooled = torch.cat(
            [global_mean_pool(h, batch.batch), global_max_pool(h, batch.batch)], dim=-1
        )
        return self.head(pooled).squeeze(-1)


@dataclass
class GraphTrainResult:
    state_dict: dict
    history: list[dict]
    best_epoch: int
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    val_prediction: np.ndarray
    test_prediction: np.ndarray
    target_mean: float
    target_scale: float
    parameter_count: int


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
def _predict(
    model: MolecularGINE,
    graphs: list[Data],
    indices: np.ndarray,
    config: GraphTrainConfig,
    kind: str,
    target_mean: float,
    target_scale: float,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        [graphs[int(index)] for index in indices],
        batch_size=min(config.batch_size, len(indices)),
        shuffle=False,
        num_workers=0,
    )
    output = []
    for batch in loader:
        output.append(model(batch).cpu().numpy())
    raw = np.concatenate(output) if output else np.empty(0, dtype=np.float32)
    if kind == "binary":
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))
    return raw * target_scale + target_mean


def train_graph_model(
    graphs: list[Data],
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    kind: str,
    primary_metric: str,
    group_centers: np.ndarray,
    train_group_counts: np.ndarray,
    objective_config: ObjectiveConfig,
    train_config: GraphTrainConfig,
) -> GraphTrainResult:
    """Train and early-stop a graph encoder under the same objectives as the MLP."""

    _seed_everything(train_config.seed)
    y = np.asarray(y, dtype=np.float32)
    if kind == "regression":
        target_mean = float(np.mean(y[train_idx]))
        target_scale = max(float(np.std(y[train_idx])), 1e-6)
    else:
        target_mean, target_scale = 0.0, 1.0

    # Clone only the scalar target into training space; graph topology and all
    # environment assignments remain fixed and label-free.
    train_graphs = []
    for index in train_idx:
        graph = graphs[int(index)].clone()
        value = (float(y[index]) - target_mean) / target_scale if kind == "regression" else float(y[index])
        graph.y = torch.tensor([value], dtype=torch.float32)
        train_graphs.append(graph)

    generator = torch.Generator().manual_seed(train_config.seed)
    loader = DataLoader(
        train_graphs,
        batch_size=min(train_config.batch_size, len(train_graphs)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = MolecularGINE(train_config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )
    centers = torch.from_numpy(np.asarray(group_centers, dtype=np.float32))
    counts = torch.from_numpy(np.asarray(train_group_counts, dtype=np.float32))

    best_state = copy.deepcopy(model.state_dict())
    best_score = -float("inf")
    best_epoch = -1
    epochs_without_gain = 0
    history: list[dict] = []

    for epoch in range(train_config.max_epochs):
        model.train()
        epoch_losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            target = batch.y.view(-1)
            if kind == "binary":
                per_sample = nn.functional.binary_cross_entropy_with_logits(
                    output, target, reduction="none"
                )
            else:
                per_sample = nn.functional.smooth_l1_loss(
                    output, target, reduction="none", beta=1.0
                )
            objective, diagnostics = compute_objective(
                per_sample,
                batch.env_groups.view(-1, centers.shape[0]).T,
                centers,
                counts,
                objective_config,
            )
            if not torch.isfinite(objective):
                raise FloatingPointError(
                    f"Non-finite graph objective at epoch {epoch}: {diagnostics}"
                )
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip)
            optimizer.step()
            epoch_losses.append(float(objective.detach()))

        val_prediction = _predict(
            model, graphs, val_idx, train_config, kind, target_mean, target_scale
        )
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
    val_prediction = _predict(
        model, graphs, val_idx, train_config, kind, target_mean, target_scale
    )
    test_prediction = _predict(
        model, graphs, test_idx, train_config, kind, target_mean, target_scale
    )
    return GraphTrainResult(
        state_dict={key: value.cpu() for key, value in best_state.items()},
        history=history,
        best_epoch=best_epoch,
        val_metrics=evaluate(kind, y[val_idx], val_prediction),
        test_metrics=evaluate(kind, y[test_idx], test_prediction),
        val_prediction=val_prediction,
        test_prediction=test_prediction,
        target_mean=target_mean,
        target_scale=target_scale,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def config_dict(config: GraphTrainConfig) -> dict:
    return asdict(config)
