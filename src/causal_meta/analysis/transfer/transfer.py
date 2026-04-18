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
from causal_meta.analysis.utils import (
    ERROR_SPECS,
    MODEL_COLORS,
    MODEL_MARKERS,
    PAPER_MODEL_LABELS,
    save_figure_data,
)

log = logging.getLogger(__name__)


def _model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#555555")


# DiBS uses a distinct orange palette in DAG-validity rows so that the
# two models (AviCi = blue, DiBS = orange) are easy to distinguish in the
# bar chart even without reading the legend.
_DAG_ROW_COLORS: dict[str, str] = {
    "DiBS": "#e67e22",  # warm orange
}


def _dag_row_color(model: str) -> str:
    """Return the colour for a model in the DAG-validity row."""
    return _DAG_ROW_COLORS.get(model, _model_color(model))


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
                # Shade the entire interpolation range as a single band.
                support_indices = sorted(
                    x_to_idx[v] for v in _TRAINING_NODE_COUNTS if v in x_to_idx
                )
                if support_indices:
                    lo = min(support_indices)
                    hi = max(support_indices)
                    ax.axvspan(
                        lo - 0.5,
                        hi + 0.5,
                        color="#d9d9d9",
                        alpha=0.40,
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
                    color="#d9d9d9",
                    alpha=0.40,
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
    save_figure_data(output_path, combined)
    return combined


# =====================================================================
# RQ2-specific 2×3 transfer figures
# =====================================================================

_ERROR_METRICS = ERROR_SPECS


def generate_rq2_transfer_figure(
    raw_df: pd.DataFrame,
    *,
    axis: str,
    output_path: Path,
) -> pd.DataFrame:
    """Generate a 3-row × N-col transfer figure for the RQ2 thesis section.

    Layout: 3 rows × N columns (one per transfer anchor):
      row 0 (top)    – DAG validity % (AviCi + DiBS, sampled & thresholded)
      row 1 (middle) – ne-SHD (lower is better, all models)
      row 2 (bottom) – Error decomposition (stacked FP / FN / Reversed)

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
        xlabel = "Target node count"
        suptitle = "Node-Count Transfer"
        id_value = _ID_NODE_COUNT
    else:
        axis_categories = {"id", "samples"}
        x_col = "SamplesPerTask"
        xlabel = "Observational samples per task"
        suptitle = "Sample-Count Transfer"
        id_value = _ID_SAMPLE_COUNT

    # Metrics needed: ne-shd for metric row, valid_dag_pct + threshold for
    # DAG row, fp/fn/reversed for error row.
    line_metrics = {"ne-shd", "valid_dag_pct", "threshold_valid_dag_pct"}
    error_metrics = {m for m, _, _ in _ERROR_METRICS}
    all_needed = line_metrics | error_metrics

    subset = raw_df[
        raw_df["AxisCategory"].isin(axis_categories) & raw_df["Metric"].isin(all_needed)
    ].copy()
    subset = subset[subset[x_col].notna()]

    if subset.empty:
        log.warning("No RQ2 transfer data for axis=%s; skipping.", axis)
        return pd.DataFrame()

    # Tag each row with its transfer anchor.
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
        log.warning(
            "No recognised transfer anchors for axis=%s; skipping RQ2 figure.", axis
        )
        return pd.DataFrame()

    n_cols = len(present_anchors)
    height_ratios = [1.2, 3.0, 1.5]  # DAG, metric, error
    row_names = ["dag", "metric", "error"]
    n_rows = 3

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, sum(height_ratios) * 1.1 + 1.5),
        sharex="col",
        sharey=False,
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )

    dag_row = 0
    metric_row = 1
    err_row = 2

    _DAG_MODELS = ["AviCi", "DiBS"]
    _DAG_METRICS = [
        ("threshold_valid_dag_pct", "Thresholded", 0.55),
        ("valid_dag_pct", "Sampled", 0.85),
    ]

    all_agg: list[pd.DataFrame] = []

    for col_idx, anchor in enumerate(present_anchors):
        anchor_data = subset[subset["_anchor"] == anchor]
        if anchor_data.empty:
            for r in range(n_rows):
                axes[r, col_idx].set_visible(False)
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

        anchor_label = TRANSFER_ANCHOR_LABELS.get(
            anchor, f"{anchor[1]} \u00d7 {anchor[0]}"
        )

        # Column header on topmost row.
        axes[0, col_idx].set_title(anchor_label, fontsize=11, fontweight="bold")

        # ── Row 0: DAG validity (AviCi + DiBS) ──────────────────────
        dag_ax = axes[dag_row, col_idx]
        dag_model_names = [m for m in _DAG_MODELS if m in agg["Model"].unique()]
        n_dag_models = len(dag_model_names)
        bar_width = 0.25
        for dm_idx, dag_model in enumerate(dag_model_names):
            for m_idx, (m_name, m_label, m_alpha) in enumerate(_DAG_METRICS):
                m_data = agg[(agg["Model"] == dag_model) & (agg["Metric"] == m_name)]
                if m_data.empty:
                    continue
                dag_xs: list[float] = []
                dag_means: list[float] = []
                dag_sems: list[float] = []
                for v in x_values:
                    row = m_data[m_data[x_col] == v]
                    if row.empty:
                        continue
                    # Offset: model group first, then sampled/thresholded within
                    group_offset = (dm_idx - n_dag_models / 2 + 0.5) * bar_width * 2.2
                    inner_offset = (m_idx - 0.5) * bar_width
                    dag_xs.append(float(x_to_idx[v]) + group_offset + inner_offset)
                    dag_means.append(float(row.iloc[0]["Mean"]))
                    dag_sems.append(float(row.iloc[0]["SEM"]))
                if dag_xs:
                    lbl = f"{dag_model} {m_label}" if col_idx == 0 else None
                    dag_ax.bar(
                        dag_xs,
                        dag_means,
                        yerr=dag_sems,
                        width=bar_width,
                        color=_dag_row_color(dag_model),
                        alpha=m_alpha,
                        capsize=2,
                        label=lbl,
                        edgecolor="white",
                        linewidth=0.3,
                    )

        dag_ax.set_ylim(0, 105)
        dag_ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if col_idx == 0:
            dag_ax.set_ylabel("DAG validity %", fontsize=9)
            dag_handles, dag_labels = dag_ax.get_legend_handles_labels()
            if dag_handles:
                dag_ax.legend(
                    dag_handles,
                    dag_labels,
                    fontsize=7,
                    loc="lower left",
                    frameon=True,
                    framealpha=0.8,
                )

        # ── Row 1: ne-SHD (all models) ──────────────────────────────
        _plot_transfer_line(
            axes[metric_row, col_idx],
            agg,
            metric_name="ne-shd",
            x_col=x_col,
            x_values=x_values,
            x_to_idx=x_to_idx,
            models=models,
            offset_step=offset_step,
            n_models=n_models,
            ylabel=r"ne-SHD $\downarrow$",
            add_legend=(col_idx == 0),
        )

        # ── Row 2: Error decomposition ──────────────────────────────
        _plot_error_decomposition(
            axes[err_row, col_idx],
            agg,
            x_col=x_col,
            x_values=x_values,
            models=models,
            add_legend=(col_idx == 0),
        )

        # Shade training-support / ID region for all rows.
        for r in range(n_rows):
            ax = axes[r, col_idx]
            _shade_support_region(ax, axis, x_to_idx, n_x, id_value)

        # X-tick labels on bottom row only.
        bottom_ax = axes[n_rows - 1, col_idx]
        bottom_ax.set_xticks(np.arange(n_x))
        bottom_ax.set_xticklabels([str(v) for v in x_values], fontsize=10)
        bottom_ax.set_xlabel(xlabel, fontsize=11)

        for r in range(n_rows):
            axes[r, col_idx].grid(True, axis="y", linestyle="--", alpha=0.4)

    # Shared legend on top — models only (from the metric row).
    handles, labels = axes[metric_row, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Model",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(labels),
            fontsize=9,
            frameon=False,
        )

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    combined = pd.concat(all_agg, ignore_index=True) if all_agg else pd.DataFrame()
    save_figure_data(output_path, combined)
    return combined


def _plot_transfer_line(
    ax: plt.Axes,
    agg: pd.DataFrame,
    *,
    metric_name: str,
    x_col: str,
    x_values: list[int],
    x_to_idx: dict[int, int],
    models: list[str],
    offset_step: float,
    n_models: int,
    ylabel: str,
    add_legend: bool,
) -> None:
    """Plot a single metric as a line-with-errorbars panel."""
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
            label=model if add_legend else None,
            color=_model_color(model),
            capsize=3,
            markersize=7,
            alpha=0.9,
        )
    ax.set_ylabel(ylabel, fontsize=11)


def _plot_error_decomposition(
    ax: plt.Axes,
    agg: pd.DataFrame,
    *,
    x_col: str,
    x_values: list[int],
    models: list[str],
    add_legend: bool,
) -> None:
    """Plot stacked FP/FN/reversed bars per model at each x position.

    Each model gets a thin stacked bar; the three error types are stacked
    vertically with distinct colours.
    """
    n_x = len(x_values)
    n_models = len(models)
    bar_width = 0.7 / max(n_models, 1)

    for model_idx, model in enumerate(models):
        offset = (model_idx - n_models / 2 + 0.5) * bar_width
        bottoms = np.zeros(n_x)
        for metric_key, metric_label, colour in _ERROR_METRICS:
            metric_df = agg[
                (agg["Metric"] == metric_key) & (agg["Model"] == model)
            ].set_index(x_col)
            heights = np.array(
                [
                    float(metric_df.loc[v, "Mean"]) if v in metric_df.index else 0.0
                    for v in x_values
                ]
            )
            bar_label = f"{model} {metric_label}" if add_legend else None
            ax.bar(
                np.arange(n_x) + offset,
                heights,
                bar_width,
                bottom=bottoms,
                color=colour,
                alpha=0.75,
                label=bar_label,
                edgecolor="white",
                linewidth=0.3,
            )
            bottoms += heights

    ax.set_ylabel("Mean error count", fontsize=11)


def _shade_support_region(
    ax: plt.Axes,
    axis: str,
    x_to_idx: dict[int, int],
    n_x: int,
    id_value: int,
) -> None:
    """Shade the in-training-support region(s) on an axis.

    For the *nodes* axis the entire interpolation range (min to max of
    ``_TRAINING_NODE_COUNTS``) is shaded as a single contiguous grey band
    to clearly mark the ID regime.  For *samples* only the single ID
    tick is highlighted.
    """
    if axis == "nodes":
        support_indices = sorted(
            x_to_idx[v] for v in _TRAINING_NODE_COUNTS if v in x_to_idx
        )
        if support_indices:
            lo = min(support_indices)
            hi = max(support_indices)
            ax.axvspan(lo - 0.5, hi + 0.5, color="#d9d9d9", alpha=0.40, zorder=0)
            if hi < n_x - 1:
                ax.axvline(hi + 0.5, color="#999999", linestyle=":", linewidth=1.0)
    elif id_value in x_to_idx:
        id_idx = x_to_idx[id_value]
        ax.axvspan(id_idx - 0.5, id_idx + 0.5, color="#d9d9d9", alpha=0.40, zorder=0)
        if id_idx < n_x - 1:
            ax.axvline(id_idx + 0.5, color="#999999", linestyle=":", linewidth=1.0)
