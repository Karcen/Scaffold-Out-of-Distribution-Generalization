"""Generate Lo-Hi and DataSAIL 70/10/20 partitions for each task.

Runs in a Python 3.11 environment with lohi_splitter (pip) and datasail-lite
(conda).  Reads results/external_splits/<TASK>__data.csv (exported from the main
pipeline) and writes <TASK>__<method>__seed<k>.csv with columns [idx, partition].

Partitions use the same 70/10/20 target as the frontier / matched-scaffold
contrast.  Lo-Hi produces a low-similarity train/test split; validation is carved
from its train block by a seeded draw.  DataSAIL uses cold ECFP clustering (C1e).
Each (task, method, seed) is isolated; failures are recorded and skipped.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path("results/external_splits")
SEEDS = [0, 1, 2]
# smallest -> largest so quick tasks land first and large ones can be capped
TASKS = ["DILI", "HIA", "PPBR", "BBB", "Solubility", "CYP2D6"]
SOLVER_MAX_SEC = 600.0


def carve_val(train_idx: np.ndarray, rng: np.random.Generator, n_total: int) -> tuple[np.ndarray, np.ndarray]:
    """Move ~10% of all molecules from train into validation."""
    n_val = max(1, round(0.10 * n_total))
    n_val = min(n_val, len(train_idx) - 1)
    perm = rng.permutation(len(train_idx))
    val = np.sort(train_idx[perm[:n_val]])
    train = np.sort(train_idx[perm[n_val:]])
    return train, val


def run_lohi(smiles: list[str], seed: int) -> dict[int, str]:
    import lohi_splitter as lohi

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(smiles))
    shuffled = [smiles[i] for i in order]
    # Lo-Hi's low-similarity MILP is often infeasible at a strict 0.7/0.2 target
    # because breaking every near-neighbour link forces many molecules to be
    # discarded.  We therefore use Lo-Hi's realistic operating point (larger
    # slack) and report the realized fractions rather than forcing 70/10/20.
    parts = lohi.hi_train_test_split(
        smiles=shuffled,
        similarity_threshold=0.4,
        train_min_frac=0.60,
        test_min_frac=0.10,
        coarsening_threshold=0.9 if len(smiles) > 2000 else None,
        max_mip_gap=0.2,
        verbose=False,
    )
    train_local, test_local = parts[0], parts[1]
    train_idx = order[np.asarray(train_local, dtype=int)]
    test_idx = order[np.asarray(test_local, dtype=int)]
    train_idx, val_idx = carve_val(train_idx, rng, len(smiles))
    assign: dict[int, str] = {}
    for i in train_idx:
        assign[int(i)] = "train"
    for i in val_idx:
        assign[int(i)] = "val"
    for i in test_idx:
        assign[int(i)] = "test"
    return assign


def run_datasail(smiles: list[str], seed: int) -> dict[int, str]:
    from datasail.sail import datasail

    e_data = {str(i): s for i, s in enumerate(smiles)}
    kwargs = dict(
        techniques=["C1e"],
        splits=[7, 1, 2],
        names=["train", "val", "test"],
        e_type="M",
        e_data=e_data,
        runs=1,
        solver="SCIP",
        seed=seed,
        max_sec=int(SOLVER_MAX_SEC),
    )
    try:
        res = datasail(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        kwargs.pop("max_sec", None)
        res = datasail(**kwargs)
    mapping = res[0]["C1e"]
    per = mapping[0] if isinstance(mapping, list) else mapping
    assign: dict[int, str] = {}
    for key, part in per.items():
        if part in ("train", "val", "test"):
            assign[int(key)] = part
    return assign


def main() -> None:
    only = set(sys.argv[1:])  # optional method filter, e.g. "lohi"
    summary = {"completed": [], "failed": []}
    for task in TASKS:
        data = pd.read_csv(DIR / f"{task}__data.csv")
        smiles = data["smiles"].astype(str).tolist()
        for method, fn in (("lohi", run_lohi), ("datasail", run_datasail)):
            if only and method not in only:
                continue
            for seed in SEEDS:
                tag = f"{task}__{method}__seed{seed}"
                if (DIR / f"{tag}.csv").exists():
                    print(f"SKIP {tag} (exists)", flush=True)
                    continue
                t0 = time.time()
                try:
                    assign = fn(smiles, seed)
                    if len(assign) < 0.5 * len(smiles):
                        raise RuntimeError(f"only {len(assign)}/{len(smiles)} assigned")
                    rows = pd.DataFrame(
                        {"idx": list(assign.keys()), "partition": list(assign.values())}
                    ).sort_values("idx")
                    rows.to_csv(DIR / f"{tag}.csv", index=False)
                    counts = rows["partition"].value_counts().to_dict()
                    dt = time.time() - t0
                    summary["completed"].append({"tag": tag, "seconds": round(dt, 1), "counts": counts})
                    print(f"OK   {tag}  {dt:5.1f}s  {counts}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    dt = time.time() - t0
                    summary["failed"].append({"tag": tag, "seconds": round(dt, 1), "error": repr(exc)[:200]})
                    print(f"FAIL {tag}  {dt:5.1f}s  {repr(exc)[:160]}", flush=True)
    (DIR / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDONE completed={len(summary['completed'])} failed={len(summary['failed'])}")


if __name__ == "__main__":
    sys.exit(main())
