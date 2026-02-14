from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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
