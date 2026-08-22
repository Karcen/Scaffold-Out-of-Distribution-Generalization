"""Compare frontier against Lo-Hi / DataSAIL hard splits over the matched control.

Reads results/metrics.csv (test partition, primary metric, ERM, MLP) and computes,
per task, the relative primary-error inflation of each hard split over the
ratio- and group-matched scaffold control (scaffold_clustered_matched):
    e = 1 - m (classification) or e = m (regression)
    inflation = (e_split - e_matched) / e_matched
Writes results/external_split_effects.csv and a compact summary printed to stdout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("results")
BASELINE = "scaffold_clustered_matched"
HARD = ["frontier", "lohi", "datasail"]


def main() -> None:
    m = pd.read_csv(RESULTS / "metrics.csv")
    m = m[(m["partition"] == "test") & (m["architecture"] == "mlp") & (m["method"] == "ERM")]
    m = m[m["metric"] == m["primary_metric"]]

    rows = []
    for task, g in m.groupby("task"):
        kind = g["kind"].iloc[0]
        metric = g["primary_metric"].iloc[0]

        def err(values: np.ndarray) -> np.ndarray:
            return (1.0 - values) if kind == "binary" else values

        base = g[g["split"] == BASELINE]
        if base.empty:
            continue
        base_err = float(np.mean(err(base["value"].to_numpy())))
        row = {"task": task, "metric": metric, "matched_error": base_err,
               "matched_primary": float(np.mean(base["value"].to_numpy()))}
        for split in HARD:
            sg = g[g["split"] == split]
            if sg.empty:
                row[f"{split}_primary"] = np.nan
                row[f"{split}_inflation_pct"] = np.nan
                row[f"{split}_n_seed"] = 0
                continue
            # per-seed inflation, then mean, so seeds pair fairly
            per_seed = []
            for seed, ssg in sg.groupby("seed"):
                e = float(np.mean(err(ssg["value"].to_numpy())))
                per_seed.append((e - base_err) / base_err)
            row[f"{split}_primary"] = float(np.mean(sg["value"].to_numpy()))
            row[f"{split}_inflation_pct"] = 100.0 * float(np.mean(per_seed))
            row[f"{split}_n_seed"] = int(sg["seed"].nunique())
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("task")
    out.to_csv(RESULTS / "external_split_effects.csv", index=False)

    print("Per-task relative primary-error inflation over the matched scaffold control (%):\n")
    show = out[["task", "metric", "matched_primary",
                "frontier_inflation_pct", "lohi_inflation_pct", "datasail_inflation_pct"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    for split in HARD:
        col = f"{split}_inflation_pct"
        vals = out[col].dropna()
        if len(vals):
            print(f"\n{split}: mean {vals.mean():.1f}%  median {vals.median():.1f}%  "
                  f"(n_tasks={len(vals)})")
    print(f"\nwrote {RESULTS / 'external_split_effects.csv'}")


if __name__ == "__main__":
    main()
