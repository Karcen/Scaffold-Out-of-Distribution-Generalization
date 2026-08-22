#!/usr/bin/env python3
"""Reproduce the Phase-1 data and provenance audit."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import rdkit
import sklearn
import torch

if "." not in sys.path:
    sys.path.insert(0, ".")

from src.chem import audit_molecule
from src.data import TASKS, load_cmnpd, load_tdc_task


RESULTS = Path("results")
RETRIEVED = "2026-07-11"
TDC_DOI = "10.7910/DVN/21LKWG"
TDC_VERSION = "105.0"
TDC_FILE_IDS = {
    "BBB_Martins.tab": 4259566,
    "HIA_Hou.tab": 4259591,
    "CYP2D6_Veith.tab": 4259580,
    "PPBR_AZ.tab": 6413140,
    "Solubility_AqSolDB.tab": 4259610,
    "DILI.tab": 4259585,
}


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def endpoint_rows(df: pd.DataFrame) -> list[dict]:
    columns = [
        "BLOOD_BRAIN_BARRIER_PENETRATION",
        "BLOOD_BRAIN_BARRIER_PENETRATION_LEVEL",
        "HUMAN_INTESTINAL_ABSORPTION_LEVEL",
        "AQUEOUS_SOLUBILITY",
        "AQUEOUS_SOLUBILITY_LEVEL",
        "CYP2D6_BINDING_MODEL",
        "HEPATOTOXICITY_MODEL",
        "PLASMA_PROTEIN_BINDING_MODEL",
    ]
    rows = []
    for column in columns:
        values = df[column]
        counts = values.value_counts(dropna=False).sort_index()
        rows.append(
            {
                "endpoint": column,
                "available_n": int(values.notna().sum()),
                "missing_n": int(values.isna().sum()),
                "unique_n": int(values.nunique(dropna=True)),
                "minimum": float(values.min()) if pd.api.types.is_numeric_dtype(values) else None,
                "median": float(values.median()) if pd.api.types.is_numeric_dtype(values) else None,
                "maximum": float(values.max()) if pd.api.types.is_numeric_dtype(values) else None,
                "distribution": json.dumps({str(key): int(value) for key, value in counts.items()}),
            }
        )
    return rows


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cmnpd_path = Path("CMNPD_1.0_calc_prop.csv")
    df = load_cmnpd(cmnpd_path)
    audits = [audit_molecule(smiles) for smiles in df["SMILES"]]
    scaffolds = pd.Series([audit.scaffold for audit in audits])
    scaffold_counts = scaffolds.value_counts()

    duplicate_conflicts = {}
    endpoint_columns = [row["endpoint"] for row in endpoint_rows(df)]
    for column in endpoint_columns:
        conflicts = 0
        groups = 0
        for _, group in df.groupby("STANDARD_INCHI_KEY"):
            if len(group) <= 1:
                continue
            groups += 1
            if group[column].nunique(dropna=False) > 1:
                conflicts += 1
        duplicate_conflicts[column] = {"duplicate_groups": groups, "conflicting_groups": conflicts}

    bbb_level = df["BLOOD_BRAIN_BARRIER_PENETRATION_LEVEL"]
    bbb_score = df["BLOOD_BRAIN_BARRIER_PENETRATION"]
    sol = df["AQUEOUS_SOLUBILITY"]
    diagnostics = {
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "sha256": file_hash(cmnpd_path),
            "bytes": cmnpd_path.stat().st_size,
        },
        "molecules": {
            "valid_smiles": int(sum(audit.valid for audit in audits)),
            "unique_raw_smiles": int(df["SMILES"].nunique()),
            "unique_canonical_isomeric": len({audit.canonical_isomeric for audit in audits}),
            "unique_connectivity": len({audit.connectivity for audit in audits}),
            "multi_fragment": int(sum(audit.fragments > 1 for audit in audits)),
            "nonzero_charge": int(sum(audit.formal_charge != 0 for audit in audits)),
            "metal_containing": int(sum(audit.contains_metal for audit in audits)),
        },
        "scaffolds": {
            "unique_including_acyclic": int(scaffold_counts.size),
            "unique_nonempty": int(scaffold_counts.drop(labels=["__ACYCLIC__"], errors="ignore").size),
            "acyclic_molecules": int((scaffolds == "__ACYCLIC__").sum()),
            "singleton_scaffolds": int((scaffold_counts == 1).sum()),
            "largest_nonempty": int(scaffold_counts.drop(labels=["__ACYCLIC__"], errors="ignore").max()),
            "top_100_molecule_fraction": float(scaffold_counts.head(100).sum() / len(df)),
        },
        "label_semantics": {
            "bbb_level4_exactly_missing_score": bool(((bbb_level == 4) == bbb_score.isna()).all()),
            "solubility_abs_gt_10": int((sol.abs() > 10).sum()),
            "solubility_abs_gt_100": int((sol.abs() > 100).sum()),
            "solubility_maximum": float(sol.max()),
        },
        "duplicate_inchikey_conflicts": duplicate_conflicts,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "rdkit": rdkit.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }
    (RESULTS / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    pd.DataFrame(endpoint_rows(df)).to_csv(RESULTS / "endpoint_summary.csv", index=False)

    task_rows = []
    provenance = {
        "CMNPD_1.0_calc_prop.csv": {
            "sha256": file_hash(cmnpd_path),
            "source": "https://docs.cmnpd.org/downloads",
            "admet_documentation": "https://docs.cmnpd.org/frequently-asked-questions/parameter-questions",
            "terms": "https://docs.cmnpd.org/terms-and-conditions",
            "retrieved": RETRIEVED,
            "license": "CC BY-NC-SA 4.0",
        }
    }
    for spec in TASKS:
        raw_path = Path("data/external") / spec.path
        raw = pd.read_csv(raw_path, sep="\t")
        clean = load_tdc_task(spec, Path("data/external"))
        task_rows.append(
            {
                "task": spec.name,
                "tdc_name": spec.tdc_task,
                "kind": spec.kind,
                "raw_rows": len(raw),
                "harmonized_rows": len(clean),
                "positive_fraction_or_median": float(clean["y"].mean() if spec.kind == "binary" else clean["y"].median()),
                "primary_metric": spec.primary_metric,
            }
        )
        provenance[str(raw_path)] = {
            "sha256": file_hash(raw_path),
            "source": "Harvard Dataverse via TDC metadata v1.1.15",
            "dataset_doi": TDC_DOI,
            "dataverse_version": TDC_VERSION,
            "dataverse_file_id": TDC_FILE_IDS[spec.path],
            "retrieved": RETRIEVED,
            "license": "CC0 1.0",
            "tdc_name": spec.tdc_task,
        }
    pd.DataFrame(task_rows).to_csv(RESULTS / "external_task_summary.csv", index=False)
    (RESULTS / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))
    print(pd.DataFrame(task_rows).to_string(index=False))


if __name__ == "__main__":
    main()
