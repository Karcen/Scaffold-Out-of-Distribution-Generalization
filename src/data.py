"""Dataset loading and harmonization.

CMNPD contains calculated descriptors and legacy model predictions.  The TDC
tables downloaded from Harvard Dataverse provide matched public endpoint tasks
for biological evaluation.  The two sources are never silently conflated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .chem import canonical_smiles, connectivity_key


@dataclass(frozen=True)
class TaskSpec:
    name: str
    path: str
    kind: str
    primary_metric: str
    tdc_task: str
    cmnpd_column: str


TASKS = (
    TaskSpec("BBB", "BBB_Martins.tab", "binary", "auroc", "BBB_Martins", "BLOOD_BRAIN_BARRIER_PENETRATION_LEVEL"),
    TaskSpec("HIA", "HIA_Hou.tab", "binary", "auroc", "HIA_Hou", "HUMAN_INTESTINAL_ABSORPTION_LEVEL"),
    TaskSpec("CYP2D6", "CYP2D6_Veith.tab", "binary", "auprc", "CYP2D6_Veith", "CYP2D6_BINDING_MODEL"),
    TaskSpec("PPBR", "PPBR_AZ.tab", "regression", "mae", "PPBR_AZ", "PLASMA_PROTEIN_BINDING_MODEL"),
    TaskSpec("Solubility", "Solubility_AqSolDB.tab", "regression", "mae", "Solubility_AqSolDB", "AQUEOUS_SOLUBILITY"),
    TaskSpec("DILI", "DILI.tab", "binary", "auroc", "DILI", "HEPATOTOXICITY_MODEL"),
)


def load_cmnpd(path: str | Path = "CMNPD_1.0_calc_prop.csv") -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_tdc_task(spec: TaskSpec, root: str | Path = "data/external") -> pd.DataFrame:
    path = Path(root) / spec.path
    df = pd.read_csv(path, sep="\t")
    if spec.name == "PPBR":
        df = df.loc[df["Species"].eq("Homo sapiens")].copy()
    df = df.rename(columns={"Drug": "smiles", "Y": "y", "Drug_ID": "source_id"})
    df = df.loc[df["smiles"].notna() & df["y"].notna(), ["source_id", "smiles", "y"]].copy()
    df["canonical_smiles"] = df["smiles"].map(canonical_smiles)
    df["connectivity_key"] = df["smiles"].map(connectivity_key)
    df = df.loc[df["canonical_smiles"].ne("__INVALID__")].copy()
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"])

    # Harmonize repeated structures before any split.  Repeated endpoint records
    # use a median; binary conflicts use majority vote (ties are discarded).
    rows: list[dict] = []
    for key, group in df.groupby("connectivity_key", sort=False):
        if spec.kind == "binary":
            mean_y = float(group["y"].mean())
            if mean_y == 0.5:
                continue
            y = int(mean_y > 0.5)
        else:
            y = float(group["y"].median())
        rows.append(
            {
                "source_id": "|".join(group["source_id"].astype(str)),
                "smiles": group["canonical_smiles"].iloc[0],
                "connectivity_key": key,
                "y": y,
                "replicate_n": len(group),
            }
        )
    out = pd.DataFrame(rows)
    if spec.kind == "binary":
        out["y"] = out["y"].astype(int)
    return out.reset_index(drop=True)


def cmnpd_teacher_target(df: pd.DataFrame, task_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Map CMNPD teacher outputs to a matched task where defensible.

    Returned mask excludes undefined teacher states.  These values are used only
    for surrogate agreement, never as experimental accuracy.
    """

    if task_name == "BBB":
        level = df["BLOOD_BRAIN_BARRIER_PENETRATION_LEVEL"].to_numpy()
        mask = level != 4
        y = np.isin(level, [0, 1]).astype(float)
    elif task_name == "HIA":
        level = df["HUMAN_INTESTINAL_ABSORPTION_LEVEL"].to_numpy()
        mask = np.ones(len(df), dtype=bool)
        y = (level == 0).astype(float)
    elif task_name == "CYP2D6":
        mask = np.ones(len(df), dtype=bool)
        y = df["CYP2D6_BINDING_MODEL"].astype(float).to_numpy()
    elif task_name == "PPBR":
        mask = np.ones(len(df), dtype=bool)
        y = df["PLASMA_PROTEIN_BINDING_MODEL"].astype(float).to_numpy()
    elif task_name == "Solubility":
        y = df["AQUEOUS_SOLUBILITY"].astype(float).to_numpy()
        mask = np.isfinite(y)
    elif task_name == "DILI":
        mask = np.ones(len(df), dtype=bool)
        y = df["HEPATOTOXICITY_MODEL"].astype(float).to_numpy()
    else:
        raise ValueError(task_name)
    return y, mask
