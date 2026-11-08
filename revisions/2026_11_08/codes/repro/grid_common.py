"""Shared feature, split, model and metric primitives for grid reproduction.

Representation-grid primitives:

  representations : nine physicochemical descriptors, 167-bit MACCS,
                    1,024-bit radius-2 Morgan (no chirality), and the
                    176-dimensional MACCS||physicochemical concatenation
  models          : logistic regression (L2, C=1, L-BFGS, <=2000 iterations)
                    and a one-hidden-layer 64-unit ReLU MLP (Adam)
  regimes         : DILI random / matched-scaffold / structural frontier,
                    CMNPD random / scaffold-grouped
  seeds           : 0-4
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
REPRESENTATIONS = ("physchem", "maccs", "morgan", "combined")
MODELS = ("logistic", "mlp")


# ---------------------------------------------------------------- chemistry

def _mol(smiles: str):
    if not isinstance(smiles, str) or not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def connectivity_key(smiles: str) -> str:
    mol = _mol(smiles)
    if mol is None:
        return "__INVALID__"
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def murcko_scaffold(smiles: str) -> str:
    mol = _mol(smiles)
    if mol is None:
        return "__INVALID__"
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return "__ACYCLIC__"
    Chem.RemoveStereochemistry(scaffold)
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)


def _bits(fp) -> np.ndarray:
    arr = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def physchem(smiles: str) -> np.ndarray:
    """Nine physicochemical descriptors for the representation grid."""
    mol = _mol(smiles)
    if mol is None:
        return np.full(9, np.nan, dtype=np.float32)
    return np.asarray(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumHDonors(mol),
            Descriptors.TPSA(mol),
            Lipinski.NumAromaticRings(mol),
            Lipinski.HeavyAtomCount(mol),
            Descriptors.qed(mol),
        ],
        dtype=np.float32,
    )


def maccs(smiles: str) -> np.ndarray:
    mol = _mol(smiles)
    if mol is None:
        return np.zeros(167, dtype=np.uint8)
    return _bits(MACCSkeys.GenMACCSKeys(mol))


def morgan(smiles: str, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    mol = _mol(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return _bits(gen.GetFingerprint(mol))


def build_features(smiles_list) -> dict:
    phys = np.stack([physchem(s) for s in smiles_list])
    mac = np.stack([maccs(s) for s in smiles_list])
    mor = np.stack([morgan(s) for s in smiles_list])
    return {
        "physchem": phys,
        "maccs": mac,
        "morgan": mor,
        "combined": np.hstack([mac, phys]),
    }


# ------------------------------------------------------------------ splits

def stratified_random(y: np.ndarray, seed: int):
    """Stratified 70/10/20; test = ceil(0.20 n), then val = 1/8 of the rest."""
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=0.20, stratify=y, random_state=seed)
    train, val = train_test_split(
        rest, test_size=0.125, stratify=y[rest], random_state=seed
    )
    return train, val, test


def partitions_from_manifest(path) -> dict:
    import pandas as pd

    man = pd.read_csv(path)
    part = man["partition"].to_numpy()
    return {
        "train": np.flatnonzero(part == "train"),
        "val": np.flatnonzero(part == "val"),
        "test": np.flatnonzero(part == "test"),
    }


def scaffold_grouped_split(smiles_list, groups, seed, fractions=(0.7, 0.1, 0.2)):
    """Seeded greedy group allocation, per paper_liver Methods."""
    rng = np.random.default_rng(seed)
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    items = list(by_group.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)

    n = len(smiles_list)
    want = {"train": fractions[0] * n, "val": fractions[1] * n, "test": fractions[2] * n}
    counts = {k: 0 for k in want}
    assign = np.empty(n, dtype=object)
    for _, members in items:
        # place in the feasible partition with the greatest remaining capacity
        best, best_slack = None, None
        for k in ("test", "val", "train"):
            slack = want[k] - counts[k]
            if slack <= 0:
                continue
            if best_slack is None or slack > best_slack:
                best, best_slack = k, slack
        if best is None:
            best = "train"
        for i in members:
            assign[i] = best
        counts[best] += len(members)
    out = {}
    for k in ("train", "val", "test"):
        idx = np.flatnonzero(assign == k)
        if k == "val" and len(idx) == 0:
            # keep sklearn-style non-empty validation when possible
            idx = np.array([], dtype=int)
        out[k] = idx
    return out


# ------------------------------------------------------------------ models

def fit_predict(representation, model, X, y, splits, seed):
    tr, va, te = splits["train"], splits["val"], splits["test"]
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])

    if model == "logistic":
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000)
        clf.fit(Xtr, y[tr])
    elif model == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            tol=1e-4,
            learning_rate_init=0.002,
            batch_size=64,
            max_iter=2000,
            early_stopping=False,
            random_state=seed,
        )
        clf.fit(Xtr, y[tr])
    else:
        raise ValueError(model)

    prob = clf.predict_proba(Xte)[:, 1]
    n_params = _n_params(clf, X.shape[1])
    return prob, n_params


def _n_params(clf, n_features: int) -> int:
    if isinstance(clf, LogisticRegression):
        return n_features + 1
    if isinstance(clf, MLPClassifier):
        a = n_features * clf.hidden_layer_sizes[0] + clf.hidden_layer_sizes[0]
        b = clf.hidden_layer_sizes[0] + 1
        return int(a + b)
    return -1


# ----------------------------------------------------------------- metrics

def expected_calibration_error(y, p, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    inner = edges[1:-1]
    idx = np.clip(np.digitize(p, inner), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(total)


def metrics(y, prob) -> dict:
    pred = (prob >= 0.5).astype(int)
    out = {
        "auroc": float(roc_auc_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "ece_10": expected_calibration_error(y, prob),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "prevalence": float(np.mean(y)),
        "test_n": int(len(y)),
    }
    try:
        from sklearn.metrics import average_precision_score

        out["auprc"] = float(average_precision_score(y, prob))
    except Exception:
        out["auprc"] = float("nan")
    return out
