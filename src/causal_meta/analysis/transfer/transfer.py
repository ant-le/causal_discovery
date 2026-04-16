from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_meta.analysis.common.thesis import (
    TRANSFER_ANCHOR_LABELS,
    graph_code_of,
    id_mechanism_of,
    metric_sem,
    transfer_anchor,
)
from causal_meta.analysis.utils import MODEL_COLORS, MODEL_MARKERS, PAPER_MODEL_LABELS

log = logging.getLogger(__name__)


def _model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#555555")


# ID anchor values for the transfer axes.
_ID_NODE_COUNT = 20
_ID_SAMPLE_COUNT = 500

# Full training-support node counts used during DG pre-training.
_TRAINING_NODE_COUNTS: set[int] = {10, 20, 30, 40}

# Ordered list of transfer ladder anchors: (mechanism, graph_code).
_TRANSFER_ANCHORS: list[tuple[str, str]] = [
    ("linear", "er20"),
    ("neuralnet", "sf2"),
]


def generate_transfer_figure(
    raw_df: pd.DataFrame, *, axis: str, output_path: Path
) -> pd.DataFrame:
    """Generate a dual-ladder transfer figure with one row per anchor.

    Layout: 2 rows (one per transfer ladder) × 2 columns (ne-SID, E-Edge F1).
    Each row shows a single anchor combination (e.g. ER-20 × Linear)
    with the x-axis representing the varying dimension (node count or
    sample count).

    Args:
        raw_df: Long-format raw task DataFrame.
        axis: ``"nodes"`` or ``"samples"``.
        output_path: Path for the output PDF figure.

    Returns:
        Aggregated DataFrame used for plotting.
    """
    if axis == "nodes":
        axis_categories = {"id", "nodes"}
        x_col = "NNodes"
        metric_specs = [
            ("ne-sid", "Normalized E-SID", False),
            ("e-edgef1", "E-Edge F1", True),
        ]
        xlabel = "Target node count"
        suptitle = "Node-Count Transfer"
        id_value = _ID_NODE_COUNT
    else:
        axis_categories = {"id", "samples"}
        x_col = "SamplesPerTask"
        metric_specs = [
            ("ne-sid", "Normalized E-SID", False),
            ("e-edgef1", "E-Edge F1", True),
        ]
        xlabel = "Observational samples per task"
        suptitle = "Sample-Count Transfer"
        id_value = _ID_SAMPLE_COUNT

    needed_metrics = [metric for metric, _, _ in metric_specs]
    subset = raw_df[
        raw_df["AxisCategory"].isin(axis_categories)
        & raw_df["Metric"].isin(needed_metrics)
    ].copy()
    subset = subset[subset[x_col].notna()]

    if subset.empty:
        log.warning("No transfer data for axis=%s; skipping.", axis)
        return pd.DataFrame()

    # Tag each row with its transfer anchor (mech, graph_code).
    # Transfer families get their anchor from the key. ID families are
    # matched by their mechanism + graph code.
    subset["_anchor"] = subset["DatasetKey"].map(transfer_anchor)
    id_mask = subset["AxisCategory"] == "id"
    subset.loc[id_mask, "_anchor"] = subset.loc[id_mask, "DatasetKey"].map(
        lambda dk: (
            (id_mechanism_of(dk), graph_code_of(dk))
            if id_mechanism_of(dk) is not None
            else None
        )
    )

    present_anchors = [
        a for a in _TRANSFER_ANCHORS if a in set(subset["_anchor"].dropna().tolist())
    ]
    if not present_anchors:
        log.warning("No recognised transfer anchors for axis=%s; skipping.", axis)
        return pd.DataFrame()

    n_rows = len(present_anchors)
    n_cols = len(metric_specs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 5.0 * n_rows),
        squeeze=False,
        sharey="col",
    )

    all_agg: list[pd.DataFrame] = []

    for row_idx, anchor in enumerate(present_anchors):
        anchor_data = subset[subset["_anchor"] == anchor]
        if anchor_data.empty:
            for col_idx in range(n_cols):
                axes[row_idx, col_idx].set_visible(False)
            continue

        agg = (
            anchor_data.groupby(["Model", x_col, "Metric"], dropna=False)["Value"]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )
        agg[x_col] = agg[x_col].astype(int)
        all_agg.append(agg)

        x_values = sorted(agg[x_col].unique())
        n_x = len(x_values)
        x_to_idx = {v: i for i, v in enumerate(x_values)}

        models = [m for m in PAPER_MODEL_LABELS.values() if m in agg["Model"].unique()]
        n_models = len(models)
        width = 0.6
        offset_step = width / max(n_models, 1)

        anchor_label = TRANSFER_ANCHOR_LABELS.get(anchor, f"{anchor[1]} × {anchor[0]}")

        for col_idx, (metric_name, ylabel, higher_is_better) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            metric_df = agg[agg["Metric"] == metric_name]

            for model_idx, model in enumerate(models):
                model_df = metric_df[metric_df["Model"] == model].sort_values(x_col)
                if model_df.empty:
                    continue
                xs: list[float] = []
                means: list[float] = []
                sems: list[float] = []
                for _, row in model_df.iterrows():
                    base_x = x_to_idx[int(row[x_col])]
                    offset = (model_idx - n_models / 2 + 0.5) * offset_step
                    xs.append(float(base_x) + offset)
                    means.append(float(row["Mean"]))
                    sems.append(float(row["SEM"]))
                ax.errorbar(
                    xs,
                    means,
                    yerr=sems,
                    fmt=MODEL_MARKERS.get(model, "o"),
                    label=model if row_idx == 0 else None,
                    color=_model_color(model),
                    capsize=3,
                    markersize=7,
                    alpha=0.9,
                )

            # Shade training-support region(s) in grey.
            if axis == "nodes":
                # Shade each in-training-support node count individually.
                support_indices = sorted(
                    x_to_idx[v] for v in _TRAINING_NODE_COUNTS if v in x_to_idx
                )
                for si in support_indices:
                    ax.axvspan(
                        si - 0.5,
                        si + 0.5,
                        color="#f2f2f2",
                        alpha=0.35,
                        zorder=0,
                    )
                # Place a boundary line after the last in-support value.
                if support_indices:
                    boundary = max(support_indices)
                    if boundary < n_x - 1:
                        ax.axvline(
                            boundary + 0.5,
                            color="#999999",
                            linestyle=":",
                            linewidth=1.0,
                        )
            elif id_value in x_to_idx:
                id_idx = x_to_idx[id_value]
                ax.axvspan(
                    id_idx - 0.5,
                    id_idx + 0.5,
                    color="#f2f2f2",
                    alpha=0.35,
                    zorder=0,
                )
                if id_idx < n_x - 1:
                    ax.axvline(
                        id_idx + 0.5, color="#999999", linestyle=":", linewidth=1.0
                    )

            ax.set_xticks(np.arange(n_x))
            ax.set_xticklabels([str(v) for v in x_values], fontsize=10)
            if row_idx == n_rows - 1:
                ax.set_xlabel(xlabel, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(
                    ylabel + (" $\\uparrow$" if higher_is_better else " $\\downarrow$"),
                    fontsize=11,
                )
            if row_idx == 0:
                ax.set_title(
                    ylabel + (" $\\uparrow$" if higher_is_better else " $\\downarrow$"),
                    fontsize=12,
                    fontweight="bold",
                )
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        # Row label on far-left axis
        axes[row_idx, 0].annotate(
            anchor_label,
            xy=(-0.30, 0.5),
            xycoords="axes fraction",
            fontsize=11,
            fontweight="bold",
            rotation=90,
            ha="center",
            va="center",
        )

    # Shared legend on top — collect from first row only (labels set there).
    all_handles: list = []
    all_labels: list[str] = []
    seen: set[str] = set()
    for ax in axes[0]:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in seen:
                all_handles.append(handle)
                all_labels.append(label)
                seen.add(label)
    if all_handles:
        fig.legend(
            all_handles,
            all_labels,
            title="Model",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(all_labels),
            fontsize=10,
            frameon=False,
        )

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    combined = pd.concat(all_agg, ignore_index=True) if all_agg else pd.DataFrame()
    return combined
