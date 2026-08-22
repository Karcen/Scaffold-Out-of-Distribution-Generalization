#!/usr/bin/env python3
"""Generate deterministic booktabs tables from recorded result files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RESULTS = Path("results")
TABLES = Path("tables")


def write(name: str, body: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text(body.rstrip() + "\n", encoding="utf-8")


def pm(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def dataset_table() -> None:
    frame = pd.read_csv(RESULTS / "external_task_summary.csv")
    kind = {"binary": "Classification", "regression": "Regression"}
    metric = {"auroc": "AUROC $\\uparrow$", "auprc": "AUPRC $\\uparrow$", "mae": "MAE $\\downarrow$"}
    target = {
        "BBB": "Permeable class",
        "HIA": "Absorbed class",
        "CYP2D6": "Inhibitor class",
        "PPBR": "Human binding (percentage points)",
        "Solubility": "Aqueous logS ($\\log_{10}$ mol l$^{-1}$)",
        "DILI": "Injury-positive class",
    }
    display = {
        "BBB": "BBB permeability",
        "HIA": "Intestinal absorption",
        "CYP2D6": "CYP2D6 inhibition",
        "PPBR": "Plasma protein binding",
        "Solubility": "Aqueous solubility",
        "DILI": "Drug-induced liver injury",
    }
    rows = [
        f"{display[r.task]} & {int(r.harmonized_rows):,} & {kind[r.kind]} & {target[r.task]} & {metric[r.primary_metric]} \\\\"
        for r in frame.itertuples()
    ]
    write(
        "datasets.tex",
        "\\begin{tabular}{lrlll}\n"
        "\\toprule\nEndpoint & $n$ & Task & Target / unit & Primary metric \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}",
    )


def primary_table() -> None:
    frame = pd.read_csv(RESULTS / "primary_summary.csv")
    rows = []
    for task in ["BBB", "HIA", "CYP2D6", "DILI", "PPBR", "Solubility"]:
        subset = frame[frame.task.eq(task)]
        metric = subset.primary_metric.iloc[0]
        metric_text = {"auroc": "AUROC $\\uparrow$", "auprc": "AUPRC $\\uparrow$", "mae": "MAE $\\downarrow$"}[metric]

        def one(split: str, method: str) -> str:
            row = subset[subset.split.eq(split) & subset.method.eq(method)].iloc[0]
            return pm(float(row["mean"]), float(row.sd))

        rows.append(
            f"{task} & {metric_text} & {one('scaffold', 'ERM')} & "
            f"{one('scaffold_clustered_matched', 'ERM')} & {one('frontier', 'ERM')} & "
            f"{one('frontier', 'MV-FREX')} \\\\"
        )
    write(
        "primary_results.tex",
        "\\begin{tabular}{llcccc}\n"
        "\\toprule\nTask & Metric & Scaffold 80/10/10 & Matched scaffold & Frontier ERM & Frontier MV-FREX \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}",
    )


def method_effect_table() -> None:
    payload = json.loads((RESULTS / "statistical_summary.json").read_text(encoding="utf-8"))
    rows = []
    display = {
        "V-REx": "V-REx-style",
        "GroupDRO": "Smooth worst-group",
        "CVaR": "Sample-tail CVaR-style",
        "MV-FREX": "MV-FREX",
    }
    for method in ["V-REx", "GroupDRO", "CVaR", "MV-FREX"]:
        d = payload["methods_vs_erm_frontier"][method]
        rows.append(
            f"{display[method]} & {100*d['mean']:.3f} & [{100*d['ci95_low']:.3f}, {100*d['ci95_high']:.3f}] "
            f"& {d['wins']} / {d['ties']} / {d['losses']} \\\\"
        )
    write(
        "method_effects.tex",
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\nMethod & Error reduction (\\%) & 95\\% CI (\\%) & Win / tie / loss \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}",
    )


def ablation_table() -> None:
    frame = pd.read_csv(RESULTS / "ablation_summary.csv")
    methods = ["ERM", "FREX-single-view", "MV-FREX-no-slope", "MV-FREX"]
    labels = ["ERM", "Single view", "No slope", "Full MV-FREX"]
    rows = []
    for task in ["BBB", "HIA", "CYP2D6", "DILI", "PPBR", "Solubility"]:
        subset = frame[frame.task.eq(task)]
        values = []
        for method in methods:
            row = subset[subset.method.eq(method)].iloc[0]
            values.append(pm(float(row["mean"]), float(row.sd)))
        rows.append(f"{task} & " + " & ".join(values) + " \\\\")
    write(
        "ablations.tex",
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\nTask & " + " & ".join(labels) + " \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}",
    )


def cmnpd_table() -> None:
    frame = pd.read_csv(RESULTS / "cmnpd_teacher_agreement.csv")
    frame = frame[frame.method.eq("ERM")]

    def val(task: str, metric: str) -> float:
        return float(frame[frame.task.eq(task) & frame.metric.eq(metric)].value.iloc[0])

    rows = [
        f"BBB & AUROC & {val('BBB', 'auroc_agreement'):.3f} \\\\ ",
        f"HIA & AUROC & {val('HIA', 'auroc_agreement'):.3f} \\\\ ",
        f"CYP2D6 & AUROC / AUPRC & {val('CYP2D6', 'auroc_agreement'):.3f} / {val('CYP2D6', 'auprc_agreement'):.3f} \\\\ ",
        f"PPBR threshold & AUROC & {val('PPBR', 'auroc_agreement'):.3f} \\\\ ",
        f"Solubility & Spearman / full MAE / filtered MAE & {val('Solubility', 'spearman'):.3f} / {val('Solubility', 'mae_full'):.3f} / {val('Solubility', 'mae_abs_teacher_le_10'):.3f} \\\\ ",
        f"DILI & AUROC & {val('DILI', 'auroc_agreement'):.3f} \\\\ ",
    ]
    write(
        "cmnpd_agreement.tex",
        "\\begin{tabular}{llr}\n"
        "\\toprule\nEndpoint mapping & Agreement metric & Value \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}",
    )


def split_effect_table() -> None:
    frame = pd.read_csv(RESULTS / "strict_matched_split_effects.csv")
    rows = []
    for task in ["BBB", "HIA", "CYP2D6", "DILI", "PPBR", "Solubility"]:
        block = frame[frame.task.eq(task)]
        metric = str(block.metric.iloc[0])
        absolute = float(block.absolute_degradation.mean())
        relative = 100 * float(block.relative_error_inflation.mean())
        adjusted = 100 * float(block.chance_adjusted_relative_error_inflation.mean())
        absolute_text = f"{absolute:.3f}"
        adjusted_text = f"{adjusted:.1f}" if metric == "auprc" else "--"
        rows.append(
            f"{task} & {metric.upper()} & {absolute_text} & {relative:.1f} & {adjusted_text} \\\\"
        )
    write(
        "split_effects.tex",
        "\\begin{tabular}{llrrr}\n"
        "\\toprule\nTask & Metric & Absolute degradation & Error inflation (\\%) & Chance-adjusted (\\%) \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}",
    )


def split_counts_table() -> None:
    frame = pd.read_csv(RESULTS / "split_partition_summary.csv")
    rows = []
    labels = {
        "scaffold_clustered_matched": "Matched scaffold",
        "frontier": "Structural frontier",
    }
    for task in ["BBB", "HIA", "CYP2D6", "DILI", "PPBR", "Solubility"]:
        for split in ["scaffold_clustered_matched", "frontier"]:
            block = frame[(frame.task == task) & (frame.split == split)]
            cells = []
            for partition in ["train", "val", "test"]:
                part = block[block.partition == partition]
                n_lo, n_hi = int(part.molecules.min()), int(part.molecules.max())
                g_lo, g_hi = int(part.groups.min()), int(part.groups.max())
                n_text = f"{n_lo}" if n_lo == n_hi else f"{n_lo}--{n_hi}"
                g_text = f"{g_lo}" if g_lo == g_hi else f"{g_lo}--{g_hi}"
                cells.extend([n_text, g_text])
            rows.append(f"{task} & {labels[split]} & " + " & ".join(cells) + " \\\\")
    write(
        "split_counts.tex",
        "\\begin{tabular}{llrrrrrr}\n"
        "\\toprule\n& & \\multicolumn{2}{c}{Train} & \\multicolumn{2}{c}{Validation} & \\multicolumn{2}{c}{Test} \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
        "Task & Split & Molecules & Groups & Molecules & Groups & Molecules & Groups \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}",
    )


def cmnpd_mapping_table() -> None:
    rows = [
        "BBB & BBB penetration level & Levels 0--1 vs 2--3 & Level 4 excluded; approximate alignment \\\\",
        "HIA & HIA level & Level 0 vs levels 1--3 & Stricter teacher threshold; approximate alignment \\\\",
        "CYP2D6 & CYP2D6 binding model & Binary output & Legacy teacher vs TDC inhibition labels \\\\",
        "PPBR & Plasma protein binding model & Binary output & TDC regression prediction thresholded at 90\\% \\\\",
        "Solubility & Aqueous solubility & Continuous output & Full and $|\\widetilde y|\\leq10$ descriptive analyses \\\\",
        "DILI & Hepatotoxicity model & Binary output & Teacher toxicity vs curated human DILI label \\\\",
    ]
    write(
        "cmnpd_mapping.tex",
        "\\begin{tabular}{llll}\n"
        "\\toprule\nTask & CMNPD field & Mapping & Scope / handling \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}",
    )


def reviewer_shift_table() -> None:
    frame = pd.read_csv(
        RESULTS / "reviewer_priority1" / "endpoint_shift_decomposition_summary.csv"
    )
    rows = []
    for task in ["BBB", "HIA", "CYP2D6", "DILI", "PPBR", "Solubility"]:
        row = frame.loc[frame.task == task].iloc[0]
        if row.kind == "binary":
            label = f"Prev. {row.matched_label_location:.3f}$\\to${row.frontier_label_location:.3f}"
            ranking = f"AUROC {row.matched_ranking:.3f}$\\to${row.frontier_ranking:.3f}"
            loss = "Brier"
        else:
            label = f"Mean {row.matched_label_location:.2f}$\\to${row.frontier_label_location:.2f}"
            ranking = f"$\\rho$ {row.matched_ranking:.3f}$\\to${row.frontier_ranking:.3f}"
            loss = "MAE"
        rows.append(
            f"{task} & {label} & {ranking} & {loss} {row.total_loss_gap:.3f} "
            f"= {row.label_mixture_component:.3f} + {row.residual_component:.3f} \\\\"
        )
    write(
        "shift_decomposition.tex",
        "\\begin{tabular}{llll}\n"
        "\\toprule\nTask & Label location & Threshold-free ranking & Loss gap = mixture + residual \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}",
    )


def gnn_control_table() -> None:
    path = RESULTS / "gnn_primary_summary.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    split_effect = pd.read_csv(RESULTS / "gnn_split_effects.csv")
    method_effect = pd.read_csv(RESULTS / "gnn_method_effects.csv")
    rows = []
    for task in ["BBB", "HIA", "CYP2D6", "DILI", "PPBR", "Solubility"]:
        subset = frame.loc[frame.task == task]
        if subset.empty:
            continue
        metric = str(subset.primary_metric.iloc[0])
        metric_text = {"auroc": "AUROC", "auprc": "AUPRC", "mae": "MAE"}[metric]

        def one(split: str, method: str) -> str:
            row = subset.loc[(subset.split == split) & (subset.method == method)].iloc[0]
            return pm(float(row["mean"]), float(row.sd))

        gap = 100 * float(
            split_effect.loc[split_effect.task == task, "relative_error_inflation"].mean()
        )
        gain = 100 * float(
            method_effect.loc[
                method_effect.task == task, "normalized_error_reduction"
            ].mean()
        )
        rows.append(
            f"{task} & {metric_text} & {one('scaffold_clustered_matched', 'ERM')} & "
            f"{one('frontier', 'ERM')} & {one('frontier', 'MV-FREX')} & "
            f"{gap:.1f} & {gain:.2f} \\\\"
        )
    write(
        "gnn_control.tex",
        "\\begin{tabular}{llcccrr}\n"
        "\\toprule\nTask & Metric & Matched ERM & Frontier ERM & Frontier probe & "
        "Gap (\\%) & Probe gain (\\%) \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}",
    )


def main() -> None:
    dataset_table()
    primary_table()
    method_effect_table()
    ablation_table()
    cmnpd_table()
    split_effect_table()
    split_counts_table()
    cmnpd_mapping_table()
    reviewer_shift_table()
    gnn_control_table()


if __name__ == "__main__":
    main()
