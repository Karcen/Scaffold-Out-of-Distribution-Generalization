#!/usr/bin/env python3
"""Generate publication figures and source-data traces using Python only."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

if "." not in sys.path:
    sys.path.insert(0, ".")

from src.chem import descriptor_vector
from src.data import TASKS, load_cmnpd, load_tdc_task


RESULTS = Path("results")
FIGURES = Path("figures")
SOURCE = FIGURES / "source_data"

TASK_ORDER = ["BBB", "HIA", "CYP2D6", "DILI", "PPBR", "Solubility"]
VIEW_ORDER = ["ecfp", "maccs", "atompair", "physchem"]
METHOD_ORDER = ["V-REx", "GroupDRO", "CVaR", "MV-FREX"]
METHOD_DISPLAY = {
    "V-REx": "V-REx-style",
    "GroupDRO": "Smooth worst-group",
    "CVaR": "Sample-tail CVaR-style",
    "MV-FREX": "MV-FREX",
}
METHOD_HEATMAP_DISPLAY = {
    "V-REx": "V-REx-style",
    "GroupDRO": "Smooth max",
    "CVaR": "Tail CVaR-style",
    "MV-FREX": "MV-FREX",
}

COLORS = {
    "ERM": "#484878",
    "V-REx": "#7884B4",
    "GroupDRO": "#42949E",
    "CVaR": "#B4C0E4",
    "MV-FREX": "#B6436C",
    "MV-FREX-no-slope": "#D58AA9",
    "FREX-single-view": "#E4CCD8",
    "TDC": "#484878",
    "CMNPD": "#C45A7A",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "mathtext.fontset": "dejavusans",
        }
    )


def panel_label(ax, label: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontweight="bold", fontsize=9.5, va="bottom")


def save_bundle(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.svg")
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    tiff_path = FIGURES / f"{name}.tiff"
    fig.savefig(
        tiff_path,
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    # Submission portals are more reliable with opaque RGB TIFFs than RGBA.
    with Image.open(tiff_path) as image:
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            background.save(tiff_path, dpi=(600, 600), compression="tiff_lzw")
    plt.close(fig)


def annotate_heatmap(ax, matrix: np.ndarray, fmt: str = ".2f", threshold: float | None = None) -> None:
    finite = matrix[np.isfinite(matrix)]
    threshold = float(np.median(np.abs(finite))) if threshold is None and len(finite) else (threshold or 0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if not np.isfinite(value):
                continue
            color = "white" if abs(value) > threshold else "#272727"
            ax.text(column, row, format(value, fmt), ha="center", va="center", fontsize=6.6, color=color)


def vector_heatmap(ax, matrix: np.ndarray, **kwargs):
    """Draw each heatmap cell as a vector polygon in PDF/SVG outputs."""

    rows, columns = matrix.shape
    image = ax.pcolormesh(
        np.arange(columns + 1) - 0.5,
        np.arange(rows + 1) - 0.5,
        matrix,
        shading="flat",
        **kwargs,
    )
    ax.set_xlim(-0.5, columns - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    return image


def figure_ood_generalization() -> dict:
    split = pd.read_csv(RESULTS / "strict_matched_split_effects.csv")
    split["error_ratio"] = 1.0 + split["relative_error_inflation"]
    split["log2_error_ratio"] = np.log2(split["error_ratio"].clip(lower=1e-6))
    split_summary = (
        split.groupby("task", as_index=False)
        .agg(mean_log2_ratio=("log2_error_ratio", "mean"), sd_log2_ratio=("log2_error_ratio", "std"), mean_ratio=("error_ratio", "mean"))
    )
    split_summary["task"] = pd.Categorical(split_summary["task"], TASK_ORDER, ordered=True)
    split_summary = split_summary.sort_values("task")
    split_summary.to_csv(SOURCE / "figure1_split_error_ratio.csv", index=False)

    stats = json.loads((RESULTS / "statistical_summary.json").read_text())
    method_rows = []
    for method in METHOD_ORDER:
        record = stats["methods_vs_erm_frontier"][method]
        method_rows.append(
            {
                "method": method,
                "mean": 100 * record["mean"],
                "low": 100 * record["ci95_low"],
                "high": 100 * record["ci95_high"],
            }
        )
    method_frame = pd.DataFrame(method_rows)
    method_frame.to_csv(SOURCE / "figure1_method_effects.csv", index=False)

    corr = pd.read_csv(RESULTS / "view_novelty_correlations.csv")
    corr_matrix = np.eye(len(VIEW_ORDER))
    for row in corr.groupby(["view_a", "view_b"])["spearman"].mean().reset_index().itertuples():
        left, right = VIEW_ORDER.index(row.view_a), VIEW_ORDER.index(row.view_b)
        corr_matrix[left, right] = corr_matrix[right, left] = row.spearman
    pd.DataFrame(corr_matrix, index=VIEW_ORDER, columns=VIEW_ORDER).to_csv(SOURCE / "figure1_view_correlations.csv")

    fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0], wspace=0.42, hspace=0.55)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    y = np.arange(len(split_summary))[::-1]
    ax_a.errorbar(
        split_summary["mean_log2_ratio"],
        y,
        xerr=split_summary["sd_log2_ratio"],
        fmt="o",
        color=COLORS["ERM"],
        ecolor="#7884B4",
        elinewidth=1.2,
        capsize=2.5,
    )
    ax_a.axvline(0, color="#767676", ls="--", lw=0.9)
    ax_a.set_yticks(y, split_summary["task"].astype(str))
    ticks = np.arange(0, 2.6, 0.5)
    ax_a.set_xticks(ticks, [f"{2**value:.1f}x" for value in ticks])
    ax_a.set_xlim(-0.12, 2.65)
    ax_a.set_xlabel("Frontier / matched-scaffold error ratio (log$_2$ axis)")
    ax_a.set_title("Sparse structural frontiers expose larger errors", loc="left", fontweight="bold")
    for yi, ratio, xpos in zip(y, split_summary["mean_ratio"], split_summary["mean_log2_ratio"]):
        ax_a.text(xpos + 0.08, yi, f"{ratio:.1f}x", va="center", fontsize=7, color="#4D4D4D")
    panel_label(ax_a, "a")

    method_plot = method_frame.iloc[::-1].reset_index(drop=True)
    yb = np.arange(len(method_plot))
    for index, row in method_plot.iterrows():
        ax_b.plot([row["low"], row["high"]], [index, index], color=COLORS[row["method"]], lw=1.5)
        ax_b.plot(row["mean"], index, "o", color=COLORS[row["method"]], ms=4.5)
    ax_b.axvline(0, color="#767676", ls="--", lw=0.9)
    ax_b.set_yticks(yb, [METHOD_DISPLAY[name] for name in method_plot["method"]])
    ax_b.set_xlabel("Normalized error reduction vs ERM (%)")
    ax_b.set_title("Fixed robust penalties do not close the gap", loc="left", fontweight="bold")
    ax_b.set_xlim(-1.35, 1.05)
    panel_label(ax_b, "b")

    norm = TwoSlopeNorm(vmin=-0.1, vcenter=0.0, vmax=1.0)
    im = vector_heatmap(ax_c, corr_matrix, cmap="RdBu_r", norm=norm)
    ax_c.set_aspect("equal")
    ax_c.set_xticks(range(len(VIEW_ORDER)), ["ECFP", "MACCS", "Atom pair", "Physchem"], rotation=30, ha="right")
    ax_c.set_yticks(range(len(VIEW_ORDER)), ["ECFP", "MACCS", "Atom pair", "Physchem"])
    annotate_heatmap(ax_c, corr_matrix, ".2f", threshold=0.55)
    ax_c.set_title("Novelty rankings disagree", loc="left", fontweight="bold")
    ax_c.tick_params(length=0)
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.045, pad=0.03)
    cbar.ax.set_title(r"$\rho$", fontsize=7.5, pad=3)
    cbar.ax.tick_params(labelsize=6.8)
    panel_label(ax_c, "c")

    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.12, top=0.93)
    save_bundle(fig, "ood_generalization_curve")
    return {"name": "ood_generalization_curve", "sources": ["results/strict_matched_split_effects.csv", "results/statistical_summary.json", "results/view_novelty_correlations.csv"]}


def confidence_ellipse(ax, points: np.ndarray, color: str) -> None:
    covariance = np.cov(points.T)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2 * 2.4477 * np.sqrt(np.maximum(values, 1e-8))
    ellipse = Ellipse(points.mean(axis=0), width, height, angle=angle, facecolor="none", edgecolor=color, lw=1.2)
    ax.add_patch(ellipse)


def figure_embedding_shift() -> dict:
    rng = np.random.default_rng(20260711)
    tdc_smiles = []
    for spec in TASKS:
        tdc_smiles.extend(load_tdc_task(spec, Path("data/external"))["smiles"].tolist())
    tdc_smiles = np.asarray(sorted(set(tdc_smiles)), dtype=str)
    cmnpd = load_cmnpd(Path("CMNPD_1.0_calc_prop.csv"))
    cmnpd_smiles = cmnpd["SMILES"].astype(str).to_numpy()
    tdc_pick = rng.choice(len(tdc_smiles), size=min(6000, len(tdc_smiles)), replace=False)
    cmnpd_pick = rng.choice(len(cmnpd_smiles), size=6000, replace=False)
    labels = np.asarray(["Public endpoint tasks"] * len(tdc_pick) + ["CMNPD"] * len(cmnpd_pick))
    sample_smiles = np.concatenate([tdc_smiles[tdc_pick], cmnpd_smiles[cmnpd_pick]])
    descriptors = np.stack([descriptor_vector(smiles) for smiles in sample_smiles])
    scaled = RobustScaler(quantile_range=(10, 90)).fit_transform(descriptors)
    scaled = np.clip(scaled, -10, 10)
    pca = PCA(n_components=2, random_state=20260711)
    embedding = pca.fit_transform(scaled)
    pca_frame = pd.DataFrame({"pc1": embedding[:, 0], "pc2": embedding[:, 1], "source": labels})
    pca_frame.to_csv(SOURCE / "figure2_physchem_pca_points.csv", index=False)

    shift = pd.read_csv(RESULTS / "cmnpd_shift_summary.csv")
    median_matrix = shift.pivot(index="task", columns="view", values="cmnpd_median").reindex(index=TASK_ORDER, columns=VIEW_ORDER).to_numpy()
    beyond_matrix = 100 * shift.pivot(index="task", columns="view", values="cmnpd_fraction_beyond_train_max").reindex(index=TASK_ORDER, columns=VIEW_ORDER).to_numpy()
    shift.to_csv(SOURCE / "figure2_shift_summary.csv", index=False)

    fig = plt.figure(figsize=(7.2, 5.0))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], hspace=0.45, wspace=0.35)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    for source, color, marker in [("Public endpoint tasks", COLORS["TDC"], "o"), ("CMNPD", COLORS["CMNPD"], "^")]:
        points = pca_frame.loc[pca_frame["source"] == source, ["pc1", "pc2"]].to_numpy()
        ax_a.scatter(points[:, 0], points[:, 1], s=5, alpha=0.12, color=color, marker=marker, linewidths=0, label=source)
        confidence_ellipse(ax_a, points, color)
    ax_a.set_xlabel(f"Physchem PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% variance)")
    ax_a.set_ylabel(f"Physchem PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% variance)")
    ax_a.set_title("Marine natural products extend reference chemical space", loc="left", fontweight="bold")
    ax_a.legend(loc="upper right", markerscale=2.5)
    panel_label(ax_a, "a")

    limit = max(abs(np.nanmin(median_matrix)), abs(np.nanmax(median_matrix)))
    im_b = vector_heatmap(
        ax_b,
        median_matrix,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
    )
    ax_b.set_xticks(range(4), ["ECFP", "MACCS", "Atom pair", "Physchem"], rotation=25, ha="right")
    ax_b.set_yticks(range(6), TASK_ORDER)
    annotate_heatmap(ax_b, median_matrix, ".1f", threshold=2.5)
    ax_b.set_title("Median CMNPD novelty", loc="left", fontweight="bold")
    ax_b.tick_params(length=0)
    for spine in ax_b.spines.values(): spine.set_visible(False)
    cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.045, pad=0.03)
    cbar_b.set_label("Robust coordinate", fontsize=7.3)
    panel_label(ax_b, "b")

    im_c = vector_heatmap(
        ax_c,
        beyond_matrix,
        cmap="PuBu",
        vmin=0,
        vmax=max(20, np.nanmax(beyond_matrix)),
    )
    ax_c.set_xticks(range(4), ["ECFP", "MACCS", "Atom pair", "Physchem"], rotation=25, ha="right")
    ax_c.set_yticks(range(6), TASK_ORDER)
    annotate_heatmap(ax_c, beyond_matrix, ".0f", threshold=30)
    ax_c.set_title("Beyond maximum training novelty", loc="left", fontweight="bold")
    ax_c.tick_params(length=0)
    for spine in ax_c.spines.values(): spine.set_visible(False)
    cbar_c = fig.colorbar(im_c, ax=ax_c, fraction=0.045, pad=0.03)
    cbar_c.set_label("CMNPD molecules (%)", fontsize=7.3)
    panel_label(ax_c, "c")

    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.11, top=0.93)
    save_bundle(fig, "embedding_shift_pca")
    return {"name": "embedding_shift_pca", "sources": ["results/cmnpd_shift_summary.csv", "figures/source_data/figure2_physchem_pca_points.csv"]}


def figure_teacher_agreement() -> dict:
    seed_data = pd.read_csv(RESULTS / "cmnpd_teacher_agreement_by_novelty_seed.csv")
    seed_data = seed_data.loc[seed_data["method"] == "ERM"]
    seed_data.to_csv(SOURCE / "figure3_teacher_agreement_by_seed.csv", index=False)
    coverage = pd.read_csv(RESULTS / "cmnpd_teacher_coverage_by_novelty.csv")
    coverage.to_csv(SOURCE / "figure3_teacher_coverage.csv", index=False)
    compound_bootstrap = pd.read_csv(
        RESULTS / "cmnpd_teacher_agreement_compound_bootstrap.csv"
    )
    compound_bootstrap.to_csv(
        SOURCE / "figure3_teacher_compound_bootstrap.csv", index=False
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.13, top=0.92, hspace=0.52, wspace=0.38)
    task_colors = {
        "BBB": "#0F4D92",
        "HIA": "#B6436C",
        "CYP2D6": "#42949E",
        "DILI": "#9A4D8E",
        "PPBR": "#767676",
    }
    markers = {"BBB": "o", "HIA": "s", "CYP2D6": "^", "DILI": "D", "PPBR": "v"}
    auc = seed_data.loc[seed_data["metric"] == "auroc_agreement"]
    for task in ["BBB", "HIA", "CYP2D6", "DILI", "PPBR"]:
        block = auc.loc[auc["task"] == task].groupby("novelty_quintile")["value"].agg(["mean", "std"]).reset_index()
        ax_a.errorbar(block["novelty_quintile"], block["mean"], yerr=block["std"], color=task_colors[task], marker=markers[task], capsize=2, label=task)
    ax_a.axhline(0.5, color="#767676", ls="--", lw=0.9, label="Chance ranking")
    bbb_interval = compound_bootstrap.loc[compound_bootstrap["task"] == "BBB"].sort_values(
        "novelty_quintile"
    )
    ax_a.fill_between(
        bbb_interval["novelty_quintile"].to_numpy(),
        bbb_interval["ci95_low"].to_numpy(),
        bbb_interval["ci95_high"].to_numpy(),
        color=task_colors["BBB"],
        alpha=0.10,
        linewidth=0,
    )
    ax_a.set_xticks(range(5), ["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax_a.set_ylabel("Legacy-teacher agreement (AUROC)")
    ax_a.set_ylim(0.08, 0.92)
    ax_a.set_title("Agreement direction is endpoint-specific", loc="left", fontweight="bold")
    ax_a.legend(ncol=2, loc="lower left", fontsize=6.8)
    panel_label(ax_a, "a")

    bbb_coverage = coverage.loc[coverage["task"] == "BBB"]
    sol_coverage = coverage.loc[coverage["task"] == "Solubility"]
    ax_b.plot(
        bbb_coverage["novelty_quintile"],
        bbb_coverage["teacher_available_fraction"],
        color=task_colors["BBB"],
        marker="o",
        label="BBB teacher available",
    )
    ax_b.plot(
        sol_coverage["novelty_quintile"],
        sol_coverage["analysis_retained_fraction"],
        color=COLORS["CMNPD"],
        marker="s",
        label=r"Solubility $|\widetilde y|\leq10$",
    )
    ax_b.set_xticks(range(5), ["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax_b.set_ylim(-0.03, 1.05)
    ax_b.set_ylabel("Fraction of quintile retained")
    ax_b.set_title("BBB coverage collapses at the frontier", loc="left", fontweight="bold")
    ax_b.legend(loc="lower left")
    panel_label(ax_b, "b")

    for metric, color, marker, label in [
        ("mae_full", "#767676", "s", "Full teacher range"),
        ("mae_abs_teacher_le_10", COLORS["CMNPD"], "o", r"$|\widetilde y|\leq10$"),
    ]:
        mae = (
            seed_data.loc[
                (seed_data["task"] == "Solubility") & (seed_data["metric"] == metric)
            ]
            .groupby("novelty_quintile")["value"]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax_c.errorbar(
            mae["novelty_quintile"],
            mae["mean"],
            yerr=mae["std"],
            color=color,
            marker=marker,
            capsize=2,
            label=label,
        )
    ax_c.set_xticks(range(5), ["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax_c.set_ylim(bottom=0)
    ax_c.set_xlabel("CMNPD multi-view novelty quintile")
    ax_c.set_ylabel("Solubility teacher MAE")
    ax_c.set_title("Solubility error rises", loc="left", fontweight="bold")
    ax_c.legend(loc="upper left")
    panel_label(ax_c, "c")

    rho = seed_data.loc[
        (seed_data["task"] == "Solubility") & (seed_data["metric"] == "spearman")
    ].groupby("novelty_quintile")["value"].agg(["mean", "std"]).reset_index()
    ax_d.errorbar(rho["novelty_quintile"], rho["mean"], yerr=rho["std"], color=COLORS["ERM"], marker="o", capsize=2)
    ax_d.axhline(0, color="#767676", ls="--", lw=0.9)
    ax_d.set_xticks(range(5), ["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax_d.set_xlabel("CMNPD multi-view novelty quintile")
    ax_d.set_ylabel(r"Solubility teacher Spearman $\rho$")
    ax_d.set_title("Rank agreement vanishes", loc="left", fontweight="bold")
    panel_label(ax_d, "d")

    fig.text(0.995, 0.01, "CMNPD values are legacy model outputs; agreement is not biological accuracy.", ha="right", va="bottom", fontsize=7, color="#606060")
    save_bundle(fig, "teacher_agreement_by_novelty")
    return {
        "name": "teacher_agreement_by_novelty",
        "sources": [
            "results/cmnpd_teacher_agreement_by_novelty_seed.csv",
            "results/cmnpd_teacher_coverage_by_novelty.csv",
            "results/cmnpd_teacher_agreement_compound_bootstrap.csv",
        ],
    }


def gain_matrix(frame: pd.DataFrame, methods: list[str]) -> np.ndarray:
    matrix = np.full((len(TASK_ORDER), len(methods)), np.nan)
    for row_index, task in enumerate(TASK_ORDER):
        for column_index, method in enumerate(methods):
            block = frame.loc[(frame["task"] == task) & (frame["method"] == method)]
            if len(block):
                matrix[row_index, column_index] = 100 * block["normalized_error_reduction"].mean()
    return matrix


def figure_method_ablation() -> dict:
    effects = pd.read_csv(RESULTS / "method_effects.csv")
    primary_matrix = gain_matrix(effects, METHOD_ORDER)
    calibration = pd.read_csv(RESULTS / "calibration_summary.csv")
    brier = calibration.loc[(calibration["split"] == "frontier") & (calibration["metric"] == "brier")]
    binary_tasks = ["BBB", "HIA", "CYP2D6", "DILI"]
    cal_matrix = np.full((len(binary_tasks), len(METHOD_ORDER)), np.nan)
    for row_index, task in enumerate(binary_tasks):
        erm = float(brier.loc[(brier["task"] == task) & (brier["method"] == "ERM"), "mean"].iloc[0])
        for column_index, method in enumerate(METHOD_ORDER):
            value = float(brier.loc[(brier["task"] == task) & (brier["method"] == method), "mean"].iloc[0])
            cal_matrix[row_index, column_index] = 100 * (erm - value) / erm

    ablation = pd.read_csv(RESULTS / "ablation_summary.csv")
    ablation_methods = ["FREX-single-view", "MV-FREX-no-slope", "MV-FREX"]
    ablation_matrix = np.full((len(TASK_ORDER), len(ablation_methods)), np.nan)
    for row_index, task in enumerate(TASK_ORDER):
        task_block = ablation.loc[ablation["task"] == task]
        if not len(task_block): continue
        erm = float(task_block.loc[task_block["method"] == "ERM", "mean"].iloc[0])
        metric = str(task_block["primary_metric"].iloc[0])
        for column_index, method in enumerate(ablation_methods):
            value = float(task_block.loc[task_block["method"] == method, "mean"].iloc[0])
            if metric in {"mae", "rmse"}:
                gain = (erm - value) / abs(erm)
            else:
                gain = (value - erm) / max(1 - erm, 1e-8)
            ablation_matrix[row_index, column_index] = 100 * gain

    pd.DataFrame(primary_matrix, index=TASK_ORDER, columns=METHOD_ORDER).to_csv(SOURCE / "figure4_primary_gain.csv")
    pd.DataFrame(cal_matrix, index=binary_tasks, columns=METHOD_ORDER).to_csv(SOURCE / "figure4_brier_gain.csv")
    pd.DataFrame(ablation_matrix, index=TASK_ORDER, columns=ablation_methods).to_csv(SOURCE / "figure4_ablation_gain.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.35), gridspec_kw={"width_ratios": [1.2, 0.9, 1.0]})
    matrices = [primary_matrix, cal_matrix, ablation_matrix]
    rows = [TASK_ORDER, binary_tasks, TASK_ORDER]
    fixed_labels = [METHOD_HEATMAP_DISPLAY[method] for method in METHOD_ORDER]
    columns = [fixed_labels, fixed_labels, ["Single view", "No slope", "Full"]]
    titles = ["Primary error reduction", "Brier-score reduction", "MV-FREX ablations"]
    labels = ["a", "b", "c"]
    limits = [max(1.5, np.nanmax(np.abs(primary_matrix))), max(5, np.nanmax(np.abs(cal_matrix))), max(3, np.nanmax(np.abs(ablation_matrix)))]
    for ax, matrix, ylabels, xlabels, title, label, limit in zip(axes, matrices, rows, columns, titles, labels, limits):
        image = vector_heatmap(
            ax,
            matrix,
            cmap="RdBu",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        )
        ax.set_xticks(range(len(xlabels)), xlabels, rotation=35, ha="right")
        ax.set_yticks(range(len(ylabels)), ylabels)
        annotate_heatmap(ax, matrix, ".2f", threshold=0.55 * limit)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.tick_params(length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
        cbar.set_label("Reduction vs ERM (%)", fontsize=6.8)
        cbar.ax.tick_params(labelsize=6.2)
        panel_label(ax, label)
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.20, top=0.90, wspace=0.62)
    save_bundle(fig, "method_ablation")
    return {"name": "method_ablation", "sources": ["results/method_effects.csv", "results/calibration_summary.csv", "results/ablation_summary.csv"]}


def figure_bbb_sensitivity() -> dict:
    """Show endpoint leverage and diagnose the BBB rank reversal."""

    review = RESULTS / "reviewer_priority1"
    leave_one_out = pd.read_csv(review / "leave_one_endpoint_out.csv")
    selected = leave_one_out.loc[
        leave_one_out["omitted_task"].isin(["NONE", "BBB"])
    ].copy()
    selected["display"] = selected["omitted_task"].map(
        {"NONE": "All endpoints", "BBB": "Without BBB"}
    )
    selected.to_csv(SOURCE / "figure5_leave_bbb_out.csv", index=False)

    quintiles = pd.read_csv(review / "bbb_frontier_quintiles.csv")
    quintiles.to_csv(SOURCE / "figure5_bbb_frontier_quintiles.csv", index=False)
    rank = json.loads((review / "bbb_rank_reversal.json").read_text(encoding="utf-8"))
    rank_frame = pd.DataFrame(
        [
            {"comparison": "Overall", "auroc": rank["ensemble_auroc"]},
            {"comparison": "Within quintiles", "auroc": rank["novelty_within_bin_auroc"]},
            {"comparison": "Across quintiles", "auroc": rank["novelty_cross_bin_auroc"]},
        ]
    )
    rank_frame.to_csv(SOURCE / "figure5_bbb_auc_decomposition.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), gridspec_kw={"width_ratios": [1.15, 1.05, 0.9]})
    ax_a, ax_b, ax_c = axes

    selected = selected.set_index("display").loc[["All endpoints", "Without BBB"]].reset_index()
    y = np.arange(len(selected))[::-1]
    point = selected["mean_relative_error_inflation_percent"].to_numpy()
    low = selected["ci95_low_percent"].to_numpy()
    high = selected["ci95_high_percent"].to_numpy()
    ax_a.errorbar(
        point,
        y,
        xerr=np.vstack([point - low, high - point]),
        fmt="o",
        color=COLORS["ERM"],
        ecolor="#7884B4",
        capsize=2.5,
    )
    ax_a.axvline(0, color="#767676", ls="--", lw=0.9)
    ax_a.set_yticks(y, selected["display"])
    ax_a.set_xlabel("Mean primary-error inflation (%)")
    ax_a.set_title("BBB has high leverage", loc="left", fontweight="bold")
    ax_a.set_xlim(0, max(270, float(high.max()) * 1.05))
    for yi, value in zip(y, point):
        ax_a.text(value + 7, yi, f"{value:.1f}%", va="center", fontsize=7)
    panel_label(ax_a, "a", x=-0.17)

    x = quintiles["frontier_quintile"].to_numpy() + 1
    ax_b.plot(
        x,
        quintiles["positive_fraction"],
        "o-",
        color="#277A73",
        label="BBB-positive fraction",
    )
    ax_b.plot(
        x,
        quintiles["mean_prediction"],
        "s-",
        color="#B6436C",
        label="Mean predicted probability",
    )
    ax_b.set_xticks(x)
    ax_b.set_ylim(0, 1.02)
    ax_b.set_xlabel("Frontier-score quintile")
    ax_b.set_ylabel("Fraction / probability")
    ax_b.set_title("Scores and labels diverge", loc="left", fontweight="bold")
    ax_b.legend(loc="lower left", fontsize=6.7)
    panel_label(ax_b, "b", x=-0.16)

    colors = [COLORS["ERM"], "#42949E", "#B6436C"]
    bars = ax_c.bar(
        np.arange(len(rank_frame)), rank_frame["auroc"], color=colors, width=0.68
    )
    ax_c.axhline(0.5, color="#767676", ls="--", lw=0.9)
    ax_c.set_ylim(0.30, 0.64)
    ax_c.set_xticks(
        np.arange(len(rank_frame)), ["Overall", "Within\nquintiles", "Across\nquintiles"]
    )
    ax_c.set_ylabel("AUROC")
    ax_c.set_title("Cross-stratum inversion", loc="left", fontweight="bold")
    for bar, value in zip(bars, rank_frame["auroc"]):
        ax_c.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", fontsize=7)
    panel_label(ax_c, "c", x=-0.19)

    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.23, top=0.86, wspace=0.52)
    save_bundle(fig, "bbb_sensitivity")
    return {
        "name": "bbb_sensitivity",
        "sources": [
            "results/reviewer_priority1/leave_one_endpoint_out.csv",
            "results/reviewer_priority1/bbb_frontier_quintiles.csv",
            "results/reviewer_priority1/bbb_rank_reversal.json",
        ],
    }


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def figure_frontier_robustness() -> dict:
    """Frontier gap survives a graph encoder and exceeds published hard splits."""

    gnn = pd.read_csv(RESULTS / "gnn_primary_summary.csv")
    gnn_erm = gnn[gnn["method"] == "ERM"]
    gnn_tasks = ["BBB", "HIA", "CYP2D6", "DILI"]
    matched, frontier = [], []
    for task in gnn_tasks:
        row_m = gnn_erm[(gnn_erm["task"] == task) & (gnn_erm["split"] == "scaffold_clustered_matched")]
        row_f = gnn_erm[(gnn_erm["task"] == task) & (gnn_erm["split"] == "frontier")]
        matched.append(float(row_m["mean"].iloc[0]))
        frontier.append(float(row_f["mean"].iloc[0]))
    gnn_frame = pd.DataFrame({"task": gnn_tasks, "matched": matched, "frontier": frontier})
    gnn_frame.to_csv(SOURCE / "figure3_gnn_primary.csv", index=False)

    effects = pd.read_csv(RESULTS / "external_split_effects.csv").set_index("task")
    cross_tasks = [t for t in TASK_ORDER if t in effects.index]
    cross = effects.loc[cross_tasks, ["frontier_inflation_pct", "lohi_inflation_pct", "datasail_inflation_pct"]]
    cross.to_csv(SOURCE / "figure3_cross_split.csv")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [0.92, 1.15]})

    x = np.arange(len(gnn_tasks))
    width = 0.38
    ax_a.bar(x - width / 2, matched, width, color=COLORS["ERM"], label="Matched scaffold")
    ax_a.bar(x + width / 2, frontier, width, color=COLORS["MV-FREX"], label="Frontier")
    ax_a.axhline(0.5, color="#767676", ls="--", lw=0.9)
    ax_a.set_xticks(x, gnn_tasks, rotation=20, ha="right")
    ax_a.set_ylim(0, 1.0)
    ax_a.set_ylabel("Primary metric (graph network)")
    ax_a.set_title("Gap survives a graph encoder", loc="left", fontweight="bold")
    ax_a.legend(loc="upper right", fontsize=6.7)
    # Mark the MLP frontier value for BBB (0.409, below chance) against the GNN bar.
    ax_a.plot(0 + width / 2, 0.409, marker="o", mfc="white", mec=COLORS["MV-FREX"],
              ms=6.5, mew=1.4, zorder=6)
    ax_a.annotate(
        "MLP: 0.409",
        xy=(0 + width / 2, 0.409),
        xytext=(0.62, 0.30),
        fontsize=6.6,
        color="#4D4D4D",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#767676", lw=0.8),
    )
    panel_label(ax_a, "a", x=-0.17)

    labels = {"frontier_inflation_pct": "Frontier", "lohi_inflation_pct": "Lo-Hi", "datasail_inflation_pct": "DataSAIL"}
    palette = {"frontier_inflation_pct": COLORS["MV-FREX"], "lohi_inflation_pct": COLORS["GroupDRO"], "datasail_inflation_pct": COLORS["V-REx"]}
    xc = np.arange(len(cross_tasks))
    width = 0.26
    for offset, col in zip((-1, 0, 1), labels):
        vals = cross[col].to_numpy()
        ax_b.bar(xc + offset * width, np.nan_to_num(vals), width, color=palette[col], label=labels[col])
        for xi, v in zip(xc + offset * width, vals):
            if np.isnan(v):
                ax_b.text(xi, 4, "n/a", ha="center", va="bottom", fontsize=5.8, color="#767676", rotation=90)
    ax_b.axhline(0, color="#767676", lw=0.9)
    ax_b.set_xticks(xc, cross_tasks, rotation=20, ha="right")
    ax_b.set_ylabel("Error inflation vs matched (%)")
    ax_b.set_title("Frontier vs published hard splits", loc="left", fontweight="bold")
    ax_b.legend(loc="upper right", fontsize=6.7, ncol=1)
    panel_label(ax_b, "b", x=-0.13)

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.87, wspace=0.34)
    save_bundle(fig, "frontier_robustness")
    return {
        "name": "frontier_robustness",
        "sources": ["results/gnn_primary_summary.csv", "results/external_split_effects.csv"],
    }


def main() -> None:
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    artifacts = [
        figure_ood_generalization(),
        figure_embedding_shift(),
        figure_teacher_agreement(),
        figure_method_ablation(),
        figure_bbb_sensitivity(),
        figure_frontier_robustness(),
    ]
    script_hash = file_hash(Path("scripts/make_figures.py"))
    manifest = []
    limitations = {
        "ood_generalization_curve": "Normalized endpoint errors are comparable ratios, not common physical units.",
        "embedding_shift_pca": "PCA is descriptive; quantitative claims use the four-view heatmaps.",
        "teacher_agreement_by_novelty": "CMNPD teacher agreement is not biological accuracy.",
        "method_ablation": "Near-zero effects are not evidence of equivalence; intervals are reported elsewhere.",
        "bbb_sensitivity": "The outcome-standardized and stratum decompositions are descriptive, not causal identification.",
        "frontier_robustness": "GNN covers four binary tasks; Lo-Hi is infeasible for PPBR and DataSAIL is limited to four tasks; realized ratios differ (Lo-Hi test fraction is smaller).",
    }
    claims = {
        "ood_generalization_curve": "Frontier shift dominates robust-objective effects.",
        "embedding_shift_pca": "CMNPD shift depends strongly on the molecular view.",
        "teacher_agreement_by_novelty": "Teacher agreement changes heterogeneously with novelty.",
        "method_ablation": "No tested robust objective wins consistently across endpoints.",
        "bbb_sensitivity": "BBB raises the skew-sensitive mean and its sub-chance AUROC is a cross-support-stratum rank reversal, not a pure prevalence effect.",
        "frontier_robustness": "The matched frontier gap persists under a graph encoder and exceeds two published hard splits on average, though no split is uniformly hardest.",
    }
    for artifact in artifacts:
        manifest.append(
            {
                "artifact_id": artifact["name"],
                "source_data": artifact["sources"],
                "transformation": {"script": "scripts/make_figures.py", "sha256": script_hash},
                "caption_claim": claims[artifact["name"]],
                "supported_manuscript_claims": [claims[artifact["name"]]],
                "limitations": limitations[artifact["name"]],
                "exports": [f"figures/{artifact['name']}.{ext}" for ext in ["svg", "pdf", "tiff", "png"]],
            }
        )
    (FIGURES / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
