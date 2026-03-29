from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from causal_meta.analysis.common.thesis import metric_sem
from causal_meta.analysis.utils import MODEL_COLORS, MODEL_MARKERS, PAPER_MODEL_LABELS


def _model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#555555")


def generate_transfer_figure(
    raw_df: pd.DataFrame, *, axis: str, output_path: Path
) -> pd.DataFrame:
    if axis == "nodes":
        axis_categories = {"id", "nodes"}
        x_col = "NNodes"
        metric_specs = [
            ("ne-sid", "Normalized E-SID", False),
            ("ne-shd", "Normalized E-SHD", False),
            ("e-edgef1", "E-Edge F1", True),
        ]
        xlabel = "Target node count"
    else:
        axis_categories = {"id", "samples"}
        x_col = "SamplesPerTask"
        metric_specs = [
            ("ne-sid", "Normalized E-SID", False),
            ("ne-shd", "Normalized E-SHD", False),
            ("e-edgef1", "E-Edge F1", True),
        ]
        xlabel = "Observational samples per task"

    needed_metrics = [metric for metric, _, _ in metric_specs]
    subset = raw_df[
        raw_df["AxisCategory"].isin(axis_categories)
        & raw_df["Metric"].isin(needed_metrics)
    ].copy()
    subset = subset[subset[x_col].notna()]
    agg = (
        subset.groupby(["Model", x_col, "Metric"], dropna=False)["Value"]
        .agg(Mean="mean", SEM=metric_sem)
        .reset_index()
    )
    agg[x_col] = agg[x_col].astype(int)

    models = list(PAPER_MODEL_LABELS.values())
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(15.5, 4.8), squeeze=False)
    for axis_idx, (metric_name, ylabel, higher_is_better) in enumerate(metric_specs):
        ax = axes[0, axis_idx]
        metric_df = agg[agg["Metric"] == metric_name]
        for model in models:
            model_df = metric_df[metric_df["Model"] == model].sort_values(x_col)
            if model_df.empty:
                continue
            ax.errorbar(
                model_df[x_col],
                model_df["Mean"],
                yerr=model_df["SEM"],
                label=model,
                color=_model_color(model),
                marker=MODEL_MARKERS.get(model, "o"),
                linewidth=2,
                capsize=3,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(
            ylabel + (" $\\uparrow$" if higher_is_better else " $\\downarrow$")
        )
        ax.grid(True, linestyle="--", alpha=0.4)
        if axis_idx == 0:
            ax.legend(title="Model", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return agg
