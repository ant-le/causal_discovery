from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from causal_meta.analysis.plots.utils import draw_point_plot

log = logging.getLogger(__name__)


def generate_structural_figure(df: pd.DataFrame, output_path: Path) -> None:
    # ---------------------------------------------------------
    # 1) Structural Metrics (e-shd, e-sid) - 1 Row, 2 Columns
    # ---------------------------------------------------------
    structural_metrics = [
        ("e-shd", "Expected SHD", "Level 1: Graph Structure (SHD) ↓"),
        ("e-sid", "Expected SID", "Level 2: Interventional Accuracy (SID) ↓"),
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(18, 7), sharey=True
    )  # Increased height for legend, share Y axis

    # Flatten axes in case we change dimensions later, though 1x2 is 1D array
    axes_flat = axes.flatten()

    for idx, (metric_id, ylabel, title) in enumerate(structural_metrics):
        draw_point_plot(
            axes_flat[idx],
            df,
            metric_id,
            ylabel,
            title,
            log_scale=False,  # Use linear scale as requested
            show_legend=False,  # We will add a shared legend
        )

    # Shared legend for Figure 1 - Top Center, No Border
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Model",
            loc="center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(labels),
            fontsize=12,
            frameon=False,  # Remove border
        )

    # Calculate global min/max for structural metrics to set tight ylim
    struct_df = df[df["Metric"].isin(["e-shd", "e-sid"])]
    if not struct_df.empty:
        # Cast to numeric to ensure calculations work
        means = pd.to_numeric(struct_df["Mean"], errors="coerce")
        sems = pd.to_numeric(struct_df["SEM"], errors="coerce").fillna(0)

        min_val = (means - sems).min()
        max_val = (means + sems).max()

        if min_val > 0:
            # Add 10% padding
            axes_flat[0].set_ylim(bottom=min_val * 0.9, top=max_val * 1.1)

    # Reserve top 10% for legend
    plt.tight_layout(rect=(0, 0, 1, 0.9))

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


def generate_performance_figure(df: pd.DataFrame, output_path: Path) -> None:
    # ---------------------------------------------------------
    # 2) Other Metrics - Grid Layout (2 Rows, 3 Columns)
    # ---------------------------------------------------------
    other_metrics = [
        ("auc", "AUC", "Causal Discovery AUC ↑"),
        ("graph_nll", "Graph NLL", "Graph Negative Log-Likelihood ↓"),
        ("ancestor_f1", "Ancestor F1", "Ancestor F1 Score ↑"),
        ("e-edgef1", "Edge F1", "Expected Edge F1 Score ↑"),
        ("edge_entropy", "Entropy", "Edge Entropy ↓"),
    ]

    # 2 rows, 3 columns = 6 slots. We have 5 metrics.
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes_flat = axes.flatten()

    for idx, (metric_id, ylabel, title) in enumerate(other_metrics):
        log_scale = metric_id == "graph_nll"
        draw_point_plot(
            axes_flat[idx],
            df,
            metric_id,
            ylabel,
            title,
            log_scale=log_scale,
            show_legend=False,  # We'll add a shared legend in the empty slot
        )

    # Handle the legend. We can put it in the 6th (empty) slot.
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        axes_flat[5].legend(handles, labels, title="Model", loc="center", fontsize=12)
        axes_flat[5].axis("off")  # Hide the axis for the legend slot
    else:
        axes_flat[5].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


# ── Helper: pivot long-format DF to wide per-dataset rows ──────────────


def _pivot_metrics(
    df: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Pivot the long-format DataFrame so each metric becomes its own column.

    Returns one row per (RunID, Model, DatasetKey) with columns for each
    metric's Mean value plus the enrichment columns (GraphType, SpectralDist, etc.).
    """
    if df.empty or "Metric" not in df.columns:
        return pd.DataFrame()
    subset = df[df["Metric"].isin(metrics)].copy()
    if subset.empty:
        return pd.DataFrame()

    # Enrich columns that are constant per (RunID, DatasetKey)
    enrich_cols = [
        c
        for c in (
            "RunID",
            "RunName",
            "Model",
            "ModelKey",
            "Dataset",
            "DatasetKey",
            "GraphType",
            "MechType",
            "NNodes",
            "SparsityParam",
            "SpectralDist",
            "KLDegreeDist",
        )
        if c in subset.columns
    ]

    pivoted = subset.pivot_table(
        index=enrich_cols,
        columns="Metric",
        values="Mean",
        aggfunc="first",
    ).reset_index()
    pivoted.columns.name = None
    return pivoted


def _ood_category(dataset_key: str) -> str:
    """Classify a dataset key as ID, OOD-Graph, OOD-Mech, or OOD-Both."""
    dk = dataset_key.lower()
    if dk.startswith("id_") or dk == "id_test":
        return "ID"
    if "both" in dk:
        return "OOD-Both"
    if "graph" in dk or "sbm" in dk:
        return "OOD-Graph"
    if "mech" in dk or any(t in dk for t in ("periodic", "square", "logistic", "pnl")):
        return "OOD-Mech"
    return "OOD"


# ── E.2  Entropy-vs-accuracy scatter (S2: Posterior Calibration) ────────


def generate_calibration_scatter(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter plot of edge_entropy (x) vs e-shd (y) per model, coloured by OOD category.

    An ideal calibrated model shows high entropy → high SHD and low entropy →
    low SHD.  Overconfident OOD predictions cluster at low-entropy / high-SHD.
    """
    wide = _pivot_metrics(df, ["edge_entropy", "e-shd"])
    if wide.empty or "edge_entropy" not in wide.columns or "e-shd" not in wide.columns:
        log.warning("Insufficient data for calibration scatter; skipping.")
        return

    wide["OODCategory"] = wide["DatasetKey"].apply(_ood_category)

    models = sorted(wide["Model"].unique())
    n_models = len(models)
    if n_models == 0:
        return

    category_colors = {
        "ID": "#2ca02c",
        "OOD-Graph": "#d62728",
        "OOD-Mech": "#ff7f0e",
        "OOD-Both": "#9467bd",
        "OOD": "#8c564b",
    }

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
    axes_flat: list[Axes] = list(axes.flatten())

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        mdf = wide[wide["Model"] == model]
        for cat, colour in category_colors.items():
            cdf = mdf[mdf["OODCategory"] == cat]
            if cdf.empty:
                continue
            ax.scatter(
                cdf["edge_entropy"],
                cdf["e-shd"],
                label=cat,
                color=colour,
                s=60,
                alpha=0.85,
                edgecolors="k",
                linewidths=0.3,
            )
        ax.set_xlabel("Edge Entropy", fontsize=11)
        ax.set_ylabel("E-SHD ↓", fontsize=11)
        ax.set_title(model, fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)

    # Shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(labels),
            fontsize=10,
            frameon=False,
        )

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved calibration scatter to %s", output_path)


# ── E.4  Distance-vs-degradation scatter (S5) ──────────────────────────


def generate_distance_degradation_scatter(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter of spectral distance (x) vs E-SID degradation relative to ID mean (y).

    One series per model with a linear trend line.  This is the key figure that
    upgrades the analysis from categorical to quantitative.
    """
    wide = _pivot_metrics(df, ["e-sid"])
    if wide.empty or "e-sid" not in wide.columns:
        log.warning("Insufficient data for distance-degradation scatter; skipping.")
        return

    if "SpectralDist" not in wide.columns:
        log.warning("No SpectralDist column; skipping distance-degradation scatter.")
        return

    wide["OODCategory"] = wide["DatasetKey"].apply(_ood_category)

    # Compute per-model ID baseline (mean E-SID across ID datasets)
    id_means = (
        wide[wide["OODCategory"] == "ID"]
        .groupby("Model")["e-sid"]
        .mean()
        .rename("id_baseline")
    )
    wide = wide.merge(id_means, on="Model", how="left")
    wide["degradation"] = wide["e-sid"] - wide["id_baseline"]

    # Drop rows with NaN distance or degradation
    plot_df = wide.dropna(subset=["SpectralDist", "degradation"])
    if plot_df.empty:
        log.warning("No valid rows for distance-degradation scatter; skipping.")
        return

    models = sorted(plot_df["Model"].unique())
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, model in enumerate(models):
        mdf = plot_df[plot_df["Model"] == model]
        colour = cmap(i)
        ax.scatter(
            mdf["SpectralDist"],
            mdf["degradation"],
            label=model,
            color=colour,
            s=60,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.3,
        )
        # Trend line
        x = mdf["SpectralDist"].to_numpy(dtype=float)
        y = mdf["degradation"].to_numpy(dtype=float)
        if len(x) >= 2:
            coeffs = np.polyfit(x, y, 1)
            x_line = np.linspace(float(x.min()), float(x.max()), 50)
            ax.plot(
                x_line,
                np.polyval(coeffs, x_line),
                color=colour,
                linestyle="--",
                alpha=0.6,
            )

    ax.set_xlabel("Spectral Distance from Training Distribution", fontsize=12)
    ax.set_ylabel("E-SID Degradation (vs. ID baseline) ↑ = worse", fontsize=12)
    ax.set_title(
        "Shift Distance vs. Performance Degradation", fontsize=14, fontweight="bold"
    )
    ax.axhline(0, color="grey", linestyle=":", alpha=0.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Model", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved distance-degradation scatter to %s", output_path)


# ── E.6  Density-stratified E-SID plot (S6) ────────────────────────────


def generate_density_stratified_figure(df: pd.DataFrame, output_path: Path) -> None:
    """Plot E-SID vs sparsity level for each model.

    Uses the ER families that vary sparsity (ER20/40/60) at fixed n_nodes=20.
    One line per model, x-axis = SparsityParam.
    """
    wide = _pivot_metrics(df, ["e-sid"])
    if wide.empty or "e-sid" not in wide.columns:
        log.warning("Insufficient data for density-stratified plot; skipping.")
        return

    if "SparsityParam" not in wide.columns:
        log.warning("No SparsityParam column; skipping density-stratified plot.")
        return

    # Filter to ER families only (those with non-null sparsity and graph_type == "er")
    er_df = wide[
        wide["SparsityParam"].notna() & (wide["GraphType"].str.lower() == "er")
    ].copy()

    if er_df.empty:
        log.warning("No ER families with sparsity data; skipping density plot.")
        return

    models = sorted(er_df["Model"].unique())
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, model in enumerate(models):
        mdf = er_df[er_df["Model"] == model].sort_values("SparsityParam")
        # Group by sparsity level in case there are multiple mech types
        grouped = (
            mdf.groupby("SparsityParam")["e-sid"].agg(["mean", "sem"]).reset_index()
        )
        colour = cmap(i)
        ax.errorbar(
            grouped["SparsityParam"],
            grouped["mean"],
            yerr=grouped["sem"],
            label=model,
            color=colour,
            marker="o",
            capsize=3,
            linewidth=2,
            markersize=7,
        )

    ax.set_xlabel("Sparsity Parameter (edge probability / expected edges)", fontsize=12)
    ax.set_ylabel("E-SID ↓", fontsize=12)
    ax.set_title(
        "Performance vs. Graph Density (ER families)", fontsize=14, fontweight="bold"
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Model", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved density-stratified figure to %s", output_path)
