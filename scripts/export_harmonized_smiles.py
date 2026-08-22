"""Export each task's harmonized SMILES (pipeline order) for external splitters.

Writes results/external_splits/<TASK>__data.csv with columns [idx, smiles, y].
The row order is identical to build_feature_cache / split_for, so partition
assignments produced by Lo-Hi / DataSAIL in a separate environment map back by idx.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if "." not in sys.path:
    sys.path.insert(0, ".")

from src.data import TASKS, load_tdc_task

OUT = Path("results/external_splits")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in TASKS:
        frame = load_tdc_task(spec, Path("data/external"))
        out = pd.DataFrame(
            {
                "idx": range(len(frame)),
                "smiles": frame["smiles"].astype(str).to_numpy(),
                "y": frame["y"].to_numpy(),
            }
        )
        path = OUT / f"{spec.name}__data.csv"
        out.to_csv(path, index=False)
        print(f"{spec.name}: {len(out):,} molecules -> {path}")


if __name__ == "__main__":
    main()
