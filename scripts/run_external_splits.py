"""Run ERM under externally generated Lo-Hi / DataSAIL splits.

Reuses the primary pipeline (build_feature_cache, build_novelty_environments,
run_one) so metric recording is identical to the scaffold/frontier runs.  Loads
results/external_splits/<TASK>__<method>__seed<k>.csv (idx, partition), builds a
SplitResult, and trains ERM with the MLP head.  Unassigned indices (dropped by a
low-similarity splitter) are excluded from all partitions.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path("results/external_splits")

_spec = importlib.util.spec_from_file_location("run_experiments", Path("scripts/run_experiments.py"))
rex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rex)

from src.environments import build_novelty_environments  # noqa: E402
from src.splits import SplitResult  # noqa: E402

TASKS = ["DILI", "HIA", "PPBR", "BBB", "Solubility", "CYP2D6"]
METHODS = ["lohi", "datasail"]
SEEDS = [0, 1, 2]


def main() -> None:
    task_map = {spec.name: spec for spec in rex.TASKS}
    train_config = rex.TrainConfig()
    completed, missing = 0, []
    for task_name in TASKS:
        spec = task_map[task_name]
        frame, feature_views = rex.build_feature_cache(spec)
        n = len(frame)
        for method in METHODS:
            for seed in SEEDS:
                path = DIR / f"{task_name}__{method}__seed{seed}.csv"
                if not path.exists():
                    missing.append(path.name)
                    continue
                table = pd.read_csv(path)
                assignment = np.array(["none"] * n, dtype=object)
                idx = table["idx"].to_numpy(dtype=int)
                assignment[idx] = table["partition"].to_numpy().astype(str)
                assignment = assignment.astype("<U8")
                if not np.isin(["train", "val", "test"], assignment).all():
                    missing.append(f"{path.name} (incomplete partitions)")
                    continue
                split_result = SplitResult(
                    assignment=assignment,
                    scaffold=np.array(["" ] * n),
                    frontier_score=np.zeros(n),
                    metadata={"split_kind": method, "source": path.name},
                )
                train_idx = np.where(assignment == "train")[0]
                environments = build_novelty_environments(feature_views, train_idx, seed=seed)
                rex.run_one(
                    spec,
                    frame,
                    feature_views,
                    method,
                    seed,
                    "ERM",
                    rex.MAIN_METHODS["ERM"],
                    "mlp",
                    split_result,
                    environments,
                    train_config,
                    False,
                )
                completed += 1
    metrics = rex.rebuild_metrics()
    print(f"\nExternal-split ERM runs launched: {completed}; missing: {len(missing)}")
    if missing:
        print("missing/incomplete:", missing)
    print(f"metrics.csv now has {len(metrics):,} rows")


if __name__ == "__main__":
    main()
