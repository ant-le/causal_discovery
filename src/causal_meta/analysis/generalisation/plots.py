from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_meta.analysis.common.thesis import (
    GRAPH_ANCHOR_LABELS,
    ID_MECHANISM_LABELS,
    TRANSFER_ANCHOR_LABELS,
    axis_category,
    format_value,
    graph_code_of,
    graph_family_of,
    id_mechanism_of,
    is_fixed_size_task_frame,
    mech_shift_graph_anchor,
    metric_sem,
    noise_shift_anchor,
    paper_model_label,
    thesis_dataset_label,
    transfer_anchor,
)
from causal_meta.analysis.utils import (
    ERROR_SPECS,
    GRAPH_DESCRIPTION_MAP,
    MECH_DESCRIPTION_MAP,
    MODEL_COLORS,
    MODEL_MARKERS,
    PAPER_MODEL_LABELS,
    save_figure_data,
)

log = logging.getLogger(__name__)


def _model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#555555")


def _resolve_models(model_filter: Sequence[str] | None = None) -> list[str]:
    """Return the ordered model list, optionally restricted to *model_filter*."""
    if model_filter is not None:
        filter_set = set(model_filter)
        return [m for m in PAPER_MODEL_LABELS.values() if m in filter_set]
    return list(PAPER_MODEL_LABELS.values())


def _bold_if_best(value: str, *, is_best: bool) -> str:
    return r"\textbf{" + value + "}" if is_best else value


_SHIFT_AXIS_SPECS: dict[str, tuple[str, str]] = {
    "graph": ("graph", "Graph Shift"),
    "mechanism": ("mechanism", "Mechanism Shift"),
    "noise": ("noise", "Noise Shift"),
    "compound": ("compound", "Compound Shift"),
}

_COMPOUND_ID_REPRESENTATIVES: tuple[tuple[str, str], ...] = (
    ("linear", "er20"),
    ("neuralnet", "sf2"),
    ("gpcde", "er60"),
)

_COMPOUND_OOD_GRAPHS: list[str] = ["sbm", "ws", "grg"]

_COMPOUND_OOD_MECH_ORDER: dict[str, int] = {
    "periodic": 0,
    "logistic_map": 1,
    "pnl_tanh": 2,
    "square": 3,
}


# ── Graph-shift multi-panel helpers ────────────────────────────────────


def _graph_shift_label(dataset_key: str) -> str:
    """Produce a short graph-topology label for graph shift panels.

    OOD-graph datasets return the OOD graph name (e.g. ``"SBM"``).
    ID datasets return the ID graph spec with ``(ID)`` suffix.
    """
    dk = dataset_key.lower()
    body = re.sub(r"_d\d+_n\d+$", "", dk)

    if dk.startswith("ood_graph_"):
        first_token = body[len("ood_graph_") :].split("_")[0]
        return GRAPH_DESCRIPTION_MAP.get(first_token, first_token.upper())

    # ID datasets — show full graph spec
    for code in sorted(GRAPH_DESCRIPTION_MAP.keys(), key=len, reverse=True):
        if f"_{code}" in body:
            return f"{GRAPH_DESCRIPTION_MAP[code]} (ID)"

    return dataset_key


# ── Shared row helpers for 3-row shift figures ────────────────────────


def _plot_dag_row_panel(
    dag_ax: plt.Axes,
    dag_panel_data: pd.DataFrame | None,
    datasets: list[str],
    x_base: np.ndarray,
    label_fn,
    *,
    panel_idx: int = 0,
    id_count: int = 0,
) -> None:
    """Plot AviCi DAG validity (sampled + thresholded) in a single panel axis."""
    if dag_panel_data is None or dag_panel_data.empty:
        return

    _DAG_METRICS = [
        ("threshold_valid_dag_pct", "Thresholded", 0.55),
        ("valid_dag_pct", "Sampled", 0.85),
    ]
    bar_width = 0.25
    for m_idx, (m_name, m_label, m_alpha) in enumerate(_DAG_METRICS):
        m_data = dag_panel_data[dag_panel_data["Metric"] == m_name]
        if m_data.empty:
            continue
        dag_agg = (
            m_data.groupby(["DatasetKey", "Dataset", "AxisCategory"], dropna=False)[
                "Value"
            ]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )
        dag_agg["DatasetLabel"] = dag_agg["DatasetKey"].map(label_fn)

        dag_xs: list[float] = []
        dag_means: list[float] = []
        dag_sems: list[float] = []
        for i, ds in enumerate(datasets):
            row = dag_agg[dag_agg["DatasetLabel"] == ds]
            if row.empty:
                continue
            offset = (m_idx - 0.5) * bar_width
            dag_xs.append(float(x_base[i]) + offset)
            dag_means.append(float(row.iloc[0]["Mean"]))
            dag_sems.append(float(row.iloc[0]["SEM"]))

        if dag_xs:
            dag_ax.bar(
                dag_xs,
                dag_means,
                yerr=dag_sems,
                width=bar_width,
                color=_model_color("AviCi"),
                alpha=m_alpha,
                capsize=2,
                label=m_label if panel_idx == 0 else None,
            )

    dag_ax.set_ylim(0, 105)
    dag_ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    if panel_idx == 0:
        dag_ax.set_ylabel("AviCi\nDAG %", fontsize=9)
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

    if 0 < id_count < len(datasets):
        dag_ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
        dag_ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)


def _plot_error_row_panel(
    err_ax: plt.Axes,
    error_panel_data: pd.DataFrame | None,
    datasets: list[str],
    x_base: np.ndarray,
    models: list[str],
    label_fn,
    *,
    panel_idx: int = 0,
    id_count: int = 0,
) -> None:
    """Plot stacked error decomposition (FP/FN/Reversed) in a single panel axis."""
    if error_panel_data is None or error_panel_data.empty:
        return

    n_models = len(models)
    bar_width = 0.6 / max(n_models, 1)

    error_agg = (
        error_panel_data.groupby(["Model", "DatasetKey", "Metric"], dropna=False)[
            "Value"
        ]
        .mean()
        .reset_index()
        .rename(columns={"Value": "Mean"})
    )
    error_agg["DatasetLabel"] = error_agg["DatasetKey"].map(label_fn)

    for model_idx, model in enumerate(models):
        model_err = error_agg[error_agg["Model"] == model]
        offset = (model_idx - n_models / 2 + 0.5) * bar_width
        bottoms = np.zeros(len(datasets))
        for err_key, err_label, err_color in ERROR_SPECS:
            heights = np.zeros(len(datasets))
            for ds_idx, ds in enumerate(datasets):
                row = model_err[
                    (model_err["DatasetLabel"] == ds) & (model_err["Metric"] == err_key)
                ]
                if not row.empty:
                    heights[ds_idx] = float(row.iloc[0]["Mean"])
            label = err_label if model_idx == 0 and panel_idx == 0 else None
            err_ax.bar(
                x_base + offset,
                heights,
                bar_width,
                bottom=bottoms,
                color=err_color,
                alpha=0.75,
                label=label,
                edgecolor="white",
                linewidth=0.3,
            )
            bottoms += heights

    err_ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    if panel_idx == 0:
        err_ax.set_ylabel("Error\nCount", fontsize=9)
        err_handles, err_labels = err_ax.get_legend_handles_labels()
        if err_handles:
            err_ax.legend(
                err_handles,
                err_labels,
                fontsize=7,
                loc="upper right",
                frameon=True,
                framealpha=0.8,
            )

    if 0 < id_count < len(datasets):
        err_ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
        err_ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)


def _generate_graph_shift_panels(
    subset: pd.DataFrame,
    metric_name: str,
    axis_title: str,
    output_path: Path,
    *,
    model_filter: Sequence[str] | None = None,
    avici_dag_df: pd.DataFrame | None = None,
    error_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate a multi-panel graph shift figure, one column per ID mechanism family.

    Three-row layout when auxiliary data is supplied:
      Row 0 (top):    AviCi DAG validity (sampled + thresholded).
      Row 1 (middle): Primary metric (E-SID / NE-SID).
      Row 2 (bottom): Error decomposition (FP / FN / Reversed).
    """
    subset = subset.copy()
    subset["MechFamily"] = subset["DatasetKey"].map(id_mechanism_of)
    subset = subset[subset["MechFamily"].notna()]

    mech_families = [
        (mk, ml)
        for mk, ml in ID_MECHANISM_LABELS.items()
        if mk in subset["MechFamily"].unique()
    ]
    if not mech_families:
        return pd.DataFrame()

    n_panels = len(mech_families)
    models = _resolve_models(model_filter)
    n_models = len(models)

    has_dag = avici_dag_df is not None and not avici_dag_df.empty
    has_err = error_df is not None and not error_df.empty

    # Tag auxiliary DataFrames with MechFamily the same way.
    dag_sub: pd.DataFrame | None = None
    if has_dag:
        dag_sub = avici_dag_df.copy()
        dag_sub["MechFamily"] = dag_sub["DatasetKey"].map(id_mechanism_of)
        dag_sub = dag_sub[dag_sub["MechFamily"].notna()]
    err_sub: pd.DataFrame | None = None
    if has_err:
        err_sub = error_df.copy()
        err_sub["MechFamily"] = err_sub["DatasetKey"].map(id_mechanism_of)
        err_sub = err_sub[err_sub["MechFamily"].notna()]

    # Build row structure: DAG (optional) | metric | error (optional).
    height_ratios: list[float] = []
    row_names: list[str] = []
    if has_dag:
        height_ratios.append(1.2)
        row_names.append("dag")
    height_ratios.append(3.0)
    row_names.append("metric")
    if has_err:
        height_ratios.append(1.5)
        row_names.append("error")
    n_rows = len(row_names)

    fig, axes = plt.subplots(
        n_rows,
        n_panels,
        figsize=(5.0 * n_panels, sum(height_ratios) * 1.1 + 1.5),
        sharex="col",
        sharey=False,
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )

    metric_row = row_names.index("metric")
    dag_row = row_names.index("dag") if has_dag else None
    err_row = row_names.index("error") if has_err else None

    all_agg: list[pd.DataFrame] = []

    for panel_idx, (mech_key, mech_label) in enumerate(mech_families):
        ax = axes[metric_row, panel_idx]
        panel_data = subset[subset["MechFamily"] == mech_key]

        agg = (
            panel_data.groupby(
                ["Model", "DatasetKey", "Dataset", "AxisCategory"], dropna=False
            )["Value"]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )
        agg["DatasetLabel"] = agg["DatasetKey"].map(_graph_shift_label)
        agg["_sort"] = agg["AxisCategory"].map({"id": 0}).fillna(1)
        agg = agg.sort_values(["_sort", "DatasetLabel"]).drop(columns=["_sort"])
        all_agg.append(agg)

        datasets = list(agg["DatasetLabel"].unique())
        axis_lookup = (
            agg[["DatasetLabel", "AxisCategory"]]
            .drop_duplicates()
            .set_index("DatasetLabel")["AxisCategory"]
            .to_dict()
        )
        n_datasets = len(datasets)
        width = 0.6
        offset_step = width / max(n_models, 1)
        x_base = np.arange(n_datasets)
        id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")

        # ── Metric row ────────────────────────────────────────────────
        for model_idx, model in enumerate(models):
            model_agg = agg[agg["Model"] == model]
            xs: list[float] = []
            means: list[float] = []
            sems: list[float] = []
            for i, ds in enumerate(datasets):
                row = model_agg[model_agg["DatasetLabel"] == ds]
                if row.empty:
                    continue
                offset = (model_idx - n_models / 2 + 0.5) * offset_step
                xs.append(float(x_base[i]) + offset)
                means.append(float(row.iloc[0]["Mean"]))
                sems.append(float(row.iloc[0]["SEM"]))
            if xs:
                ax.errorbar(
                    xs,
                    means,
                    yerr=sems,
                    fmt=MODEL_MARKERS.get(model, "o"),
                    label=model if panel_idx == 0 else None,
                    color=_model_color(model),
                    capsize=3,
                    markersize=7,
                    alpha=0.9,
                )

        # Column header on the topmost row (dag if present, else metric).
        axes[0, panel_idx].set_title(mech_label, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if panel_idx == 0:
            ylabel = (
                r"Normalized $\mathbb{E}$-SID $\downarrow$"
                if metric_name == "ne-sid"
                else r"$\mathbb{E}$-SID $\downarrow$"
            )
            ax.set_ylabel(ylabel, fontsize=11)

        # Grey ID region on metric row
        if 0 < id_count < len(datasets):
            ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
            ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

        # ── DAG row (top) ─────────────────────────────────────────────
        if dag_row is not None:
            dag_panel = (
                dag_sub[dag_sub["MechFamily"] == mech_key]
                if dag_sub is not None
                else pd.DataFrame()
            )
            _plot_dag_row_panel(
                axes[dag_row, panel_idx],
                dag_panel,
                datasets,
                x_base,
                _graph_shift_label,
                panel_idx=panel_idx,
                id_count=id_count,
            )

        # ── Error row (bottom) ────────────────────────────────────────
        if err_row is not None:
            err_panel = (
                err_sub[err_sub["MechFamily"] == mech_key]
                if err_sub is not None
                else pd.DataFrame()
            )
            _plot_error_row_panel(
                axes[err_row, panel_idx],
                err_panel,
                datasets,
                x_base,
                models,
                _graph_shift_label,
                panel_idx=panel_idx,
                id_count=id_count,
            )

    # X tick labels on the bottom row only (sharex handles the rest).
    bottom_row = n_rows - 1
    for pi in range(n_panels):
        axes[bottom_row, pi].set_xticks(np.arange(len(datasets)))
        # tick labels are set automatically by sharex on the bottom row

    # Shared legend on top
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

    fig.suptitle(axis_title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    result = pd.concat(all_agg, ignore_index=True) if all_agg else pd.DataFrame()
    save_figure_data(output_path, result)
    return result


# ── RQ2: Worst-task identification and comparison ─────────────────────


def identify_worst_ood_families(
    raw_df: pd.DataFrame,
    *,
    amortised_models: Sequence[str] | None = None,
    metric_name: str = "ne-sid",
    top_k: int = 10,
) -> pd.DataFrame:
    """Identify the OOD families where amortised models degrade most.

    For each OOD family the degradation ratio is computed as
    ``mean(OOD metric) / mean(ID metric)`` where the ID baseline is
    the anchored ID subset for the family's shift axis.  Families are
    ranked by the *maximum* degradation ratio across the amortised models
    and the top-*k* worst are returned.

    Args:
        raw_df: Long-format raw task DataFrame.
        amortised_models: Display names of amortised models.
            Defaults to :data:`AMORTISED_MODELS` from ``utils``.
        metric_name: Metric to use for degradation ranking.
        top_k: Number of worst families to return.

    Returns:
        DataFrame with columns ``DatasetKey``, ``AxisCategory``,
        ``Model``, ``OOD_Mean``, ``ID_Mean``, ``Ratio``, sorted by
        worst ``MaxRatio`` descending.
    """
    from causal_meta.analysis.utils import AMORTISED_MODELS as _AMORTISED

    if amortised_models is None:
        amortised_models = sorted(_AMORTISED)

    subset = raw_df[raw_df["Metric"].eq(metric_name)].copy()
    fixed = subset[is_fixed_size_task_frame(subset)]

    # Build per-axis ID baselines for amortised models
    id_means: dict[tuple[str, str], float] = {}  # (model, shift_key) -> mean
    for shift_key in _DEGRADATION_SHIFT_AXES:
        id_data = _degradation_id_subset(fixed, shift_key)
        for model in amortised_models:
            model_id = id_data[id_data["Model"] == model]["Value"]
            if not model_id.empty:
                id_means[(model, shift_key)] = float(model_id.mean())

    # Compute per-family degradation
    ood_cats = set(_DEGRADATION_SHIFT_AXES.keys()) - {"stress"}
    ood_data = subset[subset["AxisCategory"].isin(ood_cats)]
    # For fixed-size axes use the fixed subset; transfer axes keep all sizes
    fixed_ood = fixed[fixed["AxisCategory"].isin(ood_cats - {"nodes", "samples"})]
    transfer_ood = ood_data[ood_data["AxisCategory"].isin({"nodes", "samples"})]
    # Stress families are compound-category but identified by dataset key
    stress_ood = subset[subset["DatasetKey"].isin(_EXTREME_FAMILIES)]
    ood_pool = pd.concat([fixed_ood, transfer_ood, stress_ood], ignore_index=True)

    rows: list[dict[str, object]] = []
    for (model, dk), grp in ood_pool.groupby(["Model", "DatasetKey"]):
        model_str = str(model)
        if model_str not in amortised_models:
            continue
        # Stress families get their own axis key
        dk_str = str(dk)
        if dk_str in _EXTREME_FAMILIES:
            axis_cat = "stress"
        else:
            axis_cat = str(grp["AxisCategory"].iloc[0])
        ood_mean = float(grp["Value"].mean())
        id_val = id_means.get((model_str, axis_cat))
        if id_val is None or id_val <= 0:
            continue
        ratio = ood_mean / id_val
        rows.append(
            {
                "DatasetKey": str(dk),
                "AxisCategory": axis_cat,
                "Model": model_str,
                "OOD_Mean": ood_mean,
                "ID_Mean": id_val,
                "Ratio": ratio,
            }
        )

    if not rows:
        log.warning("No OOD degradation data for worst-task identification.")
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # Rank by maximum ratio across amortised models per family
    max_ratio = (
        result.groupby("DatasetKey")["Ratio"].max().rename("MaxRatio").reset_index()
    )
    result = result.merge(max_ratio, on="DatasetKey")
    result = result.sort_values("MaxRatio", ascending=False)

    # Select top-K unique families
    top_families = list(result["DatasetKey"].unique()[:top_k])
    return result[result["DatasetKey"].isin(top_families)].copy()


def generate_rq2_worst_task_comparison(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
    worst_families_df: pd.DataFrame | None = None,
    metric_name: str = "ne-sid",
    top_k: int = 10,
) -> pd.DataFrame:
    """Generate a two-row figure comparing all models on the worst OOD families.

    Top row: grouped ne-SID comparison across families.
    Bottom row: stacked error decomposition (FP / FN / Reversed) for the
    same families.

    Args:
        raw_df: Long-format raw task DataFrame.
        output_path: Path for the output PDF figure.
        worst_families_df: Pre-computed worst families (optional).
        metric_name: Metric to plot in top row.
        top_k: Number of worst families to show.

    Returns:
        Aggregated DataFrame used for plotting.
    """
    if worst_families_df is None or worst_families_df.empty:
        worst_families_df = identify_worst_ood_families(
            raw_df,
            metric_name=metric_name,
            top_k=top_k,
        )
    if worst_families_df.empty:
        log.warning("No worst families identified; skipping RQ2 comparison.")
        return pd.DataFrame()

    target_families = list(worst_families_df["DatasetKey"].unique()[:top_k])

    # ── Top row data: primary metric ──────────────────────────────────
    subset = raw_df[
        raw_df["Metric"].eq(metric_name) & raw_df["DatasetKey"].isin(target_families)
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    models = _resolve_models()
    n_models = len(models)

    agg = (
        subset.groupby(
            ["Model", "DatasetKey", "Dataset", "AxisCategory"], dropna=False
        )["Value"]
        .agg(Mean="mean", SEM=metric_sem)
        .reset_index()
    )
    agg["DatasetLabel"] = agg.apply(
        lambda row: thesis_dataset_label(str(row["DatasetKey"]), str(row["Dataset"])),
        axis=1,
    )

    # Sort by shift-type group then by worst degradation ratio within group.
    _GROUP_ORDER = {
        "graph": 0,
        "mechanism": 1,
        "noise": 2,
        "compound": 3,
        "stress": 4,
        "nodes": 5,
        "samples": 6,
    }
    agg["_axis_group"] = agg["DatasetKey"].apply(
        lambda dk: "stress" if str(dk) in _EXTREME_FAMILIES else None
    )
    mask_no_group = agg["_axis_group"].isna()
    agg.loc[mask_no_group, "_axis_group"] = agg.loc[
        mask_no_group, "AxisCategory"
    ].astype(str)

    agg["_group_sort"] = agg["_axis_group"].map(_GROUP_ORDER).fillna(99)
    family_order = {fk: i for i, fk in enumerate(target_families)}
    agg["_family_sort"] = agg["DatasetKey"].map(family_order).fillna(999)
    agg = agg.sort_values(["_group_sort", "_family_sort"]).drop(
        columns=["_group_sort", "_family_sort"]
    )

    datasets = list(dict.fromkeys(agg["DatasetLabel"]))
    n_datasets = len(datasets)
    width = 0.6
    offset_step = width / max(n_models, 1)
    x_base = np.arange(n_datasets)

    # ── Bottom row data: error decomposition ──────────────────────────
    error_metric_names = {"fp_count", "fn_count", "reversed_count"}
    error_subset = raw_df[
        raw_df["Metric"].isin(error_metric_names)
        & raw_df["DatasetKey"].isin(target_families)
    ].copy()

    error_agg = pd.DataFrame()
    if not error_subset.empty:
        error_agg = (
            error_subset.groupby(
                ["Model", "DatasetKey", "Dataset", "AxisCategory", "Metric"],
                dropna=False,
            )["Value"]
            .mean()
            .reset_index()
            .rename(columns={"Value": "Mean"})
        )
        error_agg["DatasetLabel"] = error_agg.apply(
            lambda row: thesis_dataset_label(
                str(row["DatasetKey"]), str(row["Dataset"])
            ),
            axis=1,
        )

    # ── Create two-row figure ─────────────────────────────────────────
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(max(8, 1.6 * n_datasets), 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    # Axis background colours and group labels helper.
    axis_lookup = (
        agg[["DatasetLabel", "_axis_group"]]
        .drop_duplicates()
        .set_index("DatasetLabel")["_axis_group"]
        .to_dict()
    )
    _AXIS_BG: dict[str, str] = {
        "graph": "#e8f0fe",
        "mechanism": "#fef3e0",
        "noise": "#f0f8e8",
        "compound": "#fde8e8",
        "stress": "#fde8fe",
        "nodes": "#f0e8fe",
        "samples": "#e8fef0",
    }
    _AXIS_GROUP_LABELS: dict[str, str] = {
        "graph": "Graph Shift",
        "mechanism": "Mechanism Shift",
        "noise": "Noise Shift",
        "compound": "Compound Shift",
        "stress": "Stress Test",
        "nodes": "Node Transfer",
        "samples": "Sample Transfer",
    }

    def _apply_axis_bg(ax: plt.Axes) -> None:
        for i, ds in enumerate(datasets):
            bg = _AXIS_BG.get(axis_lookup.get(ds, ""), None)
            if bg:
                ax.axvspan(i - 0.5, i + 0.5, color=bg, alpha=0.4, zorder=0)
        ordered_groups = [axis_lookup.get(ds, "") for ds in datasets]
        prev_group = None
        for i, grp in enumerate(ordered_groups):
            if grp != prev_group:
                if prev_group is not None:
                    ax.axvline(
                        i - 0.5, color="#888888", linestyle="-", linewidth=1.0, zorder=2
                    )
                prev_group = grp

    # ── Top row: ne-SID comparison ────────────────────────────────────
    _apply_axis_bg(ax_top)

    for model_idx, model in enumerate(models):
        model_agg = agg[agg["Model"] == model]
        xs: list[float] = []
        means: list[float] = []
        sems: list[float] = []
        for i, ds in enumerate(datasets):
            row = model_agg[model_agg["DatasetLabel"] == ds]
            if row.empty:
                continue
            offset = (model_idx - n_models / 2 + 0.5) * offset_step
            xs.append(float(x_base[i]) + offset)
            means.append(float(row.iloc[0]["Mean"]))
            sems.append(float(row.iloc[0]["SEM"]))
        if xs:
            ax_top.errorbar(
                xs,
                means,
                yerr=sems,
                fmt=MODEL_MARKERS.get(model, "o"),
                label=model,
                color=_model_color(model),
                capsize=3,
                markersize=7,
                alpha=0.9,
            )

    ylabel = (
        r"Normalized $\mathbb{E}$-SID $\downarrow$"
        if metric_name == "ne-sid"
        else r"$\mathbb{E}$-SID $\downarrow$"
    )
    ax_top.set_ylabel(ylabel, fontsize=12)
    ax_top.grid(True, axis="y", linestyle="--", alpha=0.4)

    handles, labels = ax_top.get_legend_handles_labels()
    if handles:
        ax_top.legend(
            handles,
            labels,
            title="Model",
            loc="upper right",
            fontsize=8,
            frameon=True,
            framealpha=0.9,
        )

    # ── Bottom row: stacked error decomposition ───────────────────────
    _apply_axis_bg(ax_bottom)

    _ERR_SPECS = ERROR_SPECS
    bar_width = 0.6 / max(n_models, 1)

    if not error_agg.empty:
        for model_idx, model in enumerate(models):
            model_err = error_agg[error_agg["Model"] == model]
            offset = (model_idx - n_models / 2 + 0.5) * bar_width
            bottoms = np.zeros(n_datasets)
            for eidx, (err_key, err_label, err_color) in enumerate(_ERR_SPECS):
                heights = np.zeros(n_datasets)
                for ds_idx, ds in enumerate(datasets):
                    row = model_err[
                        (model_err["DatasetLabel"] == ds)
                        & (model_err["Metric"] == err_key)
                    ]
                    if not row.empty:
                        heights[ds_idx] = float(row.iloc[0]["Mean"])
                label = f"{err_label}" if model_idx == 0 else None
                ax_bottom.bar(
                    x_base + offset,
                    heights,
                    bar_width,
                    bottom=bottoms,
                    color=err_color,
                    alpha=0.75,
                    label=label,
                    edgecolor="white",
                    linewidth=0.3,
                )
                bottoms += heights

    ax_bottom.set_ylabel("Mean error count (symlog scale)", fontsize=12)
    # Use symlog scale so stress-test counts don't dwarf other shift axes.
    ax_bottom.set_yscale("symlog", linthresh=10)
    ax_bottom.annotate(
        "Note: y-axis uses symmetric-log scale (linear below 10, log above).",
        xy=(0.0, -0.18),
        xycoords="axes fraction",
        fontsize=7,
        fontstyle="italic",
        color="#555555",
    )
    ax_bottom.set_xticks(x_base)
    ax_bottom.set_xticklabels(datasets, rotation=35, ha="right", fontsize=8)
    ax_bottom.grid(True, axis="y", linestyle="--", alpha=0.4)

    err_handles, err_labels = ax_bottom.get_legend_handles_labels()
    if err_handles:
        ax_bottom.legend(
            err_handles,
            err_labels,
            loc="upper right",
            fontsize=8,
            frameon=True,
            framealpha=0.9,
        )

    fig.suptitle(
        "All Models on Hardest OOD Families (Grouped by Shift Type)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    result = agg.drop(columns=["_axis_group"], errors="ignore")
    save_figure_data(output_path, result)
    return result


def generate_shift_figure(
    raw_df: pd.DataFrame,
    *,
    shift_axis: str,
    output_path: Path,
    model_filter: Sequence[str] | None = None,
    with_avici_dag_row: bool = False,
) -> pd.DataFrame:
    axis_cat, axis_title = _SHIFT_AXIS_SPECS[shift_axis]
    metric_name = "ne-sid" if shift_axis == "compound" else "e-sid"
    subset = raw_df[
        raw_df["Metric"].eq(metric_name) & raw_df["AxisCategory"].isin(["id", axis_cat])
    ].copy()
    subset = subset[is_fixed_size_task_frame(subset)]

    # Filter to target models when requested
    if model_filter is not None:
        subset = subset[subset["Model"].isin(model_filter)]

    # Prepare AviCi DAG validity data when requested.
    # Include both sampled (valid_dag_pct) and thresholded
    # (threshold_valid_dag_pct) DAG rates.
    avici_dag_df: pd.DataFrame | None = None
    error_df: pd.DataFrame | None = None
    if with_avici_dag_row:
        avici_dag_df = raw_df[
            raw_df["Metric"].isin(["valid_dag_pct", "threshold_valid_dag_pct"])
            & raw_df["AxisCategory"].isin(["id", axis_cat])
            & raw_df["Model"].eq("AviCi")
        ].copy()
        avici_dag_df = avici_dag_df[is_fixed_size_task_frame(avici_dag_df)]
        if avici_dag_df.empty:
            avici_dag_df = None

        # Error decomposition (FP / FN / Reversed) for all models in scope.
        _error_metrics = [k for k, _, _ in ERROR_SPECS]
        error_df = raw_df[
            raw_df["Metric"].isin(_error_metrics)
            & raw_df["AxisCategory"].isin(["id", axis_cat])
        ].copy()
        error_df = error_df[is_fixed_size_task_frame(error_df)]
        if model_filter is not None:
            error_df = error_df[error_df["Model"].isin(model_filter)]
        if error_df.empty:
            error_df = None

    # ── Graph shift: multi-panel by mechanism family ──────────────────
    if shift_axis == "graph":
        return _generate_graph_shift_panels(
            subset,
            metric_name,
            axis_title,
            output_path,
            model_filter=model_filter,
            avici_dag_df=avici_dag_df,
            error_df=error_df,
        )

    # ── Mechanism shift: multi-panel by graph anchor ──────────────────
    if shift_axis == "mechanism":
        return _generate_mech_shift_panels(
            subset,
            metric_name,
            axis_title,
            output_path,
            model_filter=model_filter,
            avici_dag_df=avici_dag_df,
            error_df=error_df,
        )

    # ── Noise shift: multi-panel by (mechanism, graph) anchor ─────────
    if shift_axis == "noise":
        return _generate_noise_shift_panels(
            subset,
            metric_name,
            axis_title,
            output_path,
            model_filter=model_filter,
            avici_dag_df=avici_dag_df,
            error_df=error_df,
        )

    # ── Compound shift: multi-panel by fixed OOD graph ────────────────
    if shift_axis == "compound":
        return _generate_compound_shift_panels(
            subset,
            metric_name,
            axis_title,
            output_path,
            model_filter=model_filter,
            avici_dag_df=avici_dag_df,
            error_df=error_df,
        )

    agg = (
        subset.groupby(
            ["Model", "DatasetKey", "Dataset", "AxisCategory"], dropna=False
        )["Value"]
        .agg(Mean="mean", SEM=metric_sem)
        .reset_index()
    )
    agg["DatasetLabel"] = agg.apply(
        lambda row: thesis_dataset_label(str(row["DatasetKey"]), str(row["Dataset"])),
        axis=1,
    )
    agg["_sort"] = agg["AxisCategory"].map({"id": 0}).fillna(1)
    agg = agg.sort_values(["_sort", "DatasetLabel"]).drop(columns=["_sort"])

    datasets = list(agg["DatasetLabel"].unique())
    models = _resolve_models(model_filter)
    axis_lookup = (
        agg[["DatasetLabel", "AxisCategory"]]
        .drop_duplicates()
        .set_index("DatasetLabel")["AxisCategory"]
        .to_dict()
    )
    n_datasets = len(datasets)
    n_models = len(models)
    width = 0.6
    offset_step = width / max(n_models, 1)
    x_base = np.arange(n_datasets)
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * n_datasets), 5))

    for model_idx, model in enumerate(models):
        model_agg = agg[agg["Model"] == model]
        xs: list[float] = []
        means: list[float] = []
        sems: list[float] = []
        for i, ds in enumerate(datasets):
            row = model_agg[model_agg["DatasetLabel"] == ds]
            if row.empty:
                continue
            offset = (model_idx - n_models / 2 + 0.5) * offset_step
            xs.append(float(x_base[i]) + offset)
            means.append(float(row.iloc[0]["Mean"]))
            sems.append(float(row.iloc[0]["SEM"]))
        if xs:
            ax.errorbar(
                xs,
                means,
                yerr=sems,
                fmt=MODEL_MARKERS.get(model, "o"),
                label=model,
                color=_model_color(model),
                capsize=3,
                markersize=7,
                alpha=0.9,
            )

    ax.set_xticks(x_base)
    ax.set_xticklabels(datasets, rotation=25, ha="right", fontsize=9)
    ylabel = (
        r"Normalized $\mathbb{E}$-SID $\downarrow$"
        if metric_name == "ne-sid"
        else r"$\mathbb{E}$-SID $\downarrow$"
    )
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")
    if 0 < id_count < len(datasets):
        ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
        ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

    # Shared legend on top.
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Model",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(labels),
            fontsize=10,
            frameon=False,
        )

    fig.suptitle(axis_title, fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, agg)
    return agg


def generate_compound_and_stress_figure(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
    model_filter: Sequence[str] | None = None,
    with_avici_dag_row: bool = False,
) -> pd.DataFrame:
    """Generate a merged compound-shift + stress-test figure.

    Each panel shows a single OOD graph topology with three zones:
    ID baselines (grey) → fixed-size compound OOD → extreme stress-test
    families (light red).

    Args:
        raw_df: Long-format raw task DataFrame.
        output_path: Path for the output PDF figure.
        model_filter: If given, only include these model display names.
        with_avici_dag_row: If True, add DAG-validity top row and error
            decomposition bottom row.

    Returns:
        Aggregated DataFrame used for plotting.
    """
    metric_name = "ne-sid"

    # Fixed-size compound data (same filter as generate_shift_figure compound)
    compound_subset = raw_df[
        raw_df["Metric"].eq(metric_name)
        & raw_df["AxisCategory"].isin(["id", "compound"])
    ].copy()
    compound_subset = compound_subset[is_fixed_size_task_frame(compound_subset)]

    # Stress-test data (extreme families, not fixed-size)
    stress_subset = raw_df[
        raw_df["Metric"].eq(metric_name) & raw_df["DatasetKey"].isin(_EXTREME_FAMILIES)
    ].copy()

    # Filter to target models when requested
    if model_filter is not None:
        compound_subset = compound_subset[compound_subset["Model"].isin(model_filter)]
        stress_subset = stress_subset[stress_subset["Model"].isin(model_filter)]

    # Prepare AviCi DAG validity + error decomposition data
    avici_dag_df: pd.DataFrame | None = None
    error_df: pd.DataFrame | None = None
    if with_avici_dag_row:
        avici_dag_df = raw_df[
            raw_df["Metric"].isin(["valid_dag_pct", "threshold_valid_dag_pct"])
            & raw_df["AxisCategory"].isin(["id", "compound"])
            & raw_df["Model"].eq("AviCi")
        ].copy()
        avici_dag_df = avici_dag_df[is_fixed_size_task_frame(avici_dag_df)]
        if avici_dag_df.empty:
            avici_dag_df = None

        _error_metrics = [k for k, _, _ in ERROR_SPECS]
        error_df = raw_df[
            raw_df["Metric"].isin(_error_metrics)
            & raw_df["AxisCategory"].isin(["id", "compound"])
        ].copy()
        error_df = error_df[is_fixed_size_task_frame(error_df)]
        if model_filter is not None:
            error_df = error_df[error_df["Model"].isin(model_filter)]
        if error_df.empty:
            error_df = None

    return _generate_compound_shift_panels(
        compound_subset,
        metric_name,
        "Compound Shift",
        output_path,
        stress_df=stress_subset,
        model_filter=model_filter,
        avici_dag_df=avici_dag_df,
        error_df=error_df,
    )


# ── Shift-progression figure (ID → Single → Compound → Stress) ────────

_PROGRESSION_STAGES: list[tuple[str, str, list[str]]] = [
    ("ID", "ID", ["id"]),
    ("Graph\nShift", "Graph Shift", ["graph"]),
    ("Mechanism\nShift", "Mechanism Shift", ["mechanism"]),
    ("Compound\nShift", "Compound Shift", ["compound"]),
]
"""(tick_label, legend_label, axis_categories) for each non-stress stage."""


def generate_compound_progression_figure(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
    model_filter: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate a shift-progression figure: ID -> Single -> Compound -> Stress.

    Shows how performance degrades as the number of simultaneous
    distribution shifts increases, answering whether OOD degradation
    compounds uniformly.  A bottom row shows AviCi DAG validity
    (sampled and thresholded) across the same progression stages.

    Stages:
        1. **ID** -- in-distribution baseline.
        2. **Graph Shift** -- graph topology OOD only.
        3. **Mechanism Shift** -- functional mechanism OOD only.
        4. **Compound Shift** -- graph + mechanism OOD (fixed size).
        5. **Stress Test** -- compound shift + large nodes + small samples.

    For each stage, individual family means are shown as jittered dots
    behind the per-model mean +/- SEM markers, connected by lines to
    emphasise the degradation trend.

    Args:
        raw_df: Long-format raw task DataFrame.
        output_path: Path for the output PDF figure.
        model_filter: If given, only include these model display names.

    Returns:
        Aggregated DataFrame used for plotting.
    """
    metric_name = "ne-sid"

    base = raw_df[raw_df["Metric"].eq(metric_name)].copy()
    fixed = base[is_fixed_size_task_frame(base)]
    stress = base[base["DatasetKey"].isin(_EXTREME_FAMILIES)]

    if model_filter is not None:
        fixed = fixed[fixed["Model"].isin(model_filter)]
        stress = stress[stress["Model"].isin(model_filter)]

    models = _resolve_models(model_filter)
    n_models = len(models)

    # ── Prepare AviCi DAG validity data ───────────────────────────────
    dag_base = raw_df[
        raw_df["Metric"].isin(["valid_dag_pct", "threshold_valid_dag_pct"])
        & raw_df["Model"].eq("AviCi")
    ].copy()
    dag_fixed = dag_base[is_fixed_size_task_frame(dag_base)]
    dag_stress = dag_base[dag_base["DatasetKey"].isin(_EXTREME_FAMILIES)]
    has_dag_row = not dag_fixed.empty

    # ── Per-family aggregation ────────────────────────────────────────
    all_stages: list[str] = [s[1] for s in _PROGRESSION_STAGES] + ["Stress Test"]
    tick_labels: list[str] = [s[0] for s in _PROGRESSION_STAGES] + ["Stress\nTest"]

    rows: list[dict[str, object]] = []
    for tick, stage, cats in _PROGRESSION_STAGES:
        stage_data = fixed[fixed["AxisCategory"].isin(cats)]
        if stage_data.empty:
            continue
        fam_means = (
            stage_data.groupby(["Model", "DatasetKey"])["Value"]
            .mean()
            .reset_index()
            .rename(columns={"Value": "FamilyMean"})
        )
        fam_means["Stage"] = stage
        rows.extend(fam_means.to_dict("records"))

    # Stress-test stage
    if not stress.empty:
        fam_means = (
            stress.groupby(["Model", "DatasetKey"])["Value"]
            .mean()
            .reset_index()
            .rename(columns={"Value": "FamilyMean"})
        )
        fam_means["Stage"] = "Stress Test"
        rows.extend(fam_means.to_dict("records"))

    if not rows:
        log.warning("No data for compound progression figure.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── DAG validity aggregation per stage ────────────────────────────
    dag_rows: list[dict[str, object]] = []
    if has_dag_row:
        for tick, stage, cats in _PROGRESSION_STAGES:
            stage_dag = dag_fixed[dag_fixed["AxisCategory"].isin(cats)]
            if stage_dag.empty:
                continue
            fam_dag = (
                stage_dag.groupby(["Metric", "DatasetKey"])["Value"]
                .mean()
                .reset_index()
                .rename(columns={"Value": "FamilyMean"})
            )
            fam_dag["Stage"] = stage
            dag_rows.extend(fam_dag.to_dict("records"))

        if not dag_stress.empty:
            fam_dag = (
                dag_stress.groupby(["Metric", "DatasetKey"])["Value"]
                .mean()
                .reset_index()
                .rename(columns={"Value": "FamilyMean"})
            )
            fam_dag["Stage"] = "Stress Test"
            dag_rows.extend(fam_dag.to_dict("records"))

    dag_df = pd.DataFrame(dag_rows) if dag_rows else pd.DataFrame()
    has_dag_row = not dag_df.empty

    # ── Plot ──────────────────────────────────────────────────────────
    n_stages = len(all_stages)
    x_base = np.arange(n_stages)

    if has_dag_row:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(9, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1.2]},
        )
        ax = axes[0]
        dag_ax = axes[1]
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        dag_ax = None

    width = 0.5
    offset_step = width / max(n_models, 1)
    rng = np.random.default_rng(42)

    for model_idx, model in enumerate(models):
        model_data = df[df["Model"] == model]
        offset = (model_idx - n_models / 2 + 0.5) * offset_step
        color = _model_color(model)
        marker = MODEL_MARKERS.get(model, "o")

        stage_xs: list[float] = []
        stage_means: list[float] = []
        stage_sems: list[float] = []

        for stage_idx, stage in enumerate(all_stages):
            vals = model_data.loc[model_data["Stage"] == stage, "FamilyMean"]
            if vals.empty:
                continue

            x_center = float(x_base[stage_idx]) + offset
            stage_xs.append(x_center)
            stage_means.append(float(vals.mean()))
            stage_sems.append(float(vals.sem()) if len(vals) > 1 else 0.0)

            # Individual family dots (jittered)
            jitter = rng.uniform(
                -offset_step * 0.25, offset_step * 0.25, size=len(vals)
            )
            ax.scatter(
                [x_center + j for j in jitter],
                vals.values,
                color=color,
                alpha=0.25,
                s=12,
                zorder=2,
                edgecolors="none",
            )

        if stage_xs:
            # Connecting line (subtle)
            ax.plot(
                stage_xs,
                stage_means,
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.5,
                zorder=3,
            )
            # Markers with error bars
            ax.errorbar(
                stage_xs,
                stage_means,
                yerr=stage_sems,
                fmt=marker,
                label=model,
                color=color,
                capsize=4,
                markersize=9,
                alpha=0.9,
                zorder=4,
            )

    # ── Stage-region shading ──────────────────────────────────────────
    stage_bg = ["#f2f2f2", "#e6f0ff", "#e6f0ff", "#fff3e6", "#ffe0e0"]
    shade_axes = [ax] + ([dag_ax] if dag_ax is not None else [])
    for cur_ax in shade_axes:
        for i, bg in enumerate(stage_bg[:n_stages]):
            cur_ax.axvspan(i - 0.5, i + 0.5, color=bg, alpha=0.35, zorder=0)
        for i in range(1, n_stages):
            cur_ax.axvline(
                i - 0.5, color="#999999", linestyle=":", linewidth=1.0, zorder=1
            )

    if not has_dag_row:
        ax.set_xticks(x_base)
        ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel(r"Normalized $\mathbb{E}$-SID $\downarrow$", fontsize=12)
    ax.set_title("Shift Progression", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            title="Model",
            loc="upper left",
            fontsize=10,
            frameon=True,
        )

    # ── Bottom row: AviCi DAG validity ────────────────────────────────
    if dag_ax is not None and not dag_df.empty:
        _DAG_METRICS_PROG = [
            ("threshold_valid_dag_pct", "Thresholded", 0.55),
            ("valid_dag_pct", "Sampled", 0.85),
        ]
        bar_width = 0.2
        for m_idx, (m_name, m_label, m_alpha) in enumerate(_DAG_METRICS_PROG):
            m_data = dag_df[dag_df["Metric"] == m_name]
            if m_data.empty:
                continue
            dag_stage_xs: list[float] = []
            dag_stage_means: list[float] = []
            dag_stage_sems: list[float] = []

            for stage_idx, stage in enumerate(all_stages):
                vals = m_data.loc[m_data["Stage"] == stage, "FamilyMean"]
                if vals.empty:
                    continue
                offset = (m_idx - 0.5) * bar_width
                dag_stage_xs.append(float(x_base[stage_idx]) + offset)
                dag_stage_means.append(float(vals.mean()))
                dag_stage_sems.append(float(vals.sem()) if len(vals) > 1 else 0.0)

            if dag_stage_xs:
                dag_ax.bar(
                    dag_stage_xs,
                    dag_stage_means,
                    yerr=dag_stage_sems,
                    width=bar_width,
                    color=_model_color("AviCi"),
                    alpha=m_alpha,
                    capsize=2,
                    label=m_label,
                )

        dag_ax.set_ylim(0, 105)
        dag_ax.set_xticks(x_base)
        dag_ax.set_xticklabels(tick_labels, fontsize=10)
        dag_ax.set_ylabel("AviCi\nDAG %", fontsize=9)
        dag_ax.grid(True, axis="y", linestyle="--", alpha=0.4)
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

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    log.info("Saved compound progression figure to %s", output_path)
    save_figure_data(output_path, df)
    return df


# ── Valid DAG % shift figure ───────────────────────────────────────────

_VALID_DAG_SHIFT_SPECS: dict[str, tuple[str, str]] = {
    "graph": ("graph", "Graph Shift"),
    "mechanism": ("mechanism", "Mechanism Shift"),
    "compound": ("compound", "Compound Shift"),
    "nodes": ("nodes", "Node-Count Transfer"),
    "samples": ("samples", "Sample-Count Transfer"),
}
"""Shift axes for the valid DAG % figure (noise omitted — explicit models
lack noise-shift data, and both amortised models are constant at 100 / ~65)."""


def _compute_dibs_valid_dag_pct(
    run_dirs: Sequence[Path],
    *,
    include_threshold: bool = False,
) -> pd.DataFrame:
    """Compute per-task valid DAG % for DiBS from cached inference artifacts.

    DiBS does not record ``valid_dag_pct`` in ``metrics.json``.  Instead we
    load each ``seed_*.pt.gz`` artifact and check acyclicity of individual
    posterior samples via the matrix-exponential trace characterisation:
    a binary matrix *A* is a DAG iff ``tr(exp(A)) == n``.

    Args:
        run_dirs: Run directories containing DiBS inference artifacts.
        include_threshold: If True, also compute ``threshold_valid_dag_pct``
            (acyclicity of the thresholded posterior-mean graph) and return
            rows for both metrics.

    Returns:
        Long-format DataFrame with columns matching the raw_df schema:
        ``Model``, ``ModelKey``, ``DatasetKey``, ``Dataset``, ``TaskIdx``,
        ``Metric`` (== ``"valid_dag_pct"`` or ``"threshold_valid_dag_pct"``),
        ``Value``, ``AxisCategory``, ``NNodes``, ``SamplesPerTask``.
    """
    import json
    import torch

    from causal_meta.analysis.diagnostics.posterior import _discover_artifacts
    from causal_meta.analysis.utils import (
        _as_mapping,
        map_dataset_description,
    )
    from causal_meta.runners.utils.artifacts import torch_load as _torch_load

    rows: list[dict[str, object]] = []

    for run_dir in run_dirs:
        run_dir = Path(run_dir).resolve()
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r") as f:
            payload = json.load(f)
        metadata = (
            _as_mapping(payload.get("metadata")) if isinstance(payload, dict) else {}
        )
        model_name = str(metadata.get("model_name", "unknown"))
        if "dibs" not in model_name.lower():
            continue

        fam_meta = (
            _as_mapping(payload.get("family_metadata"))
            if isinstance(payload, dict)
            else {}
        )

        inference_root_raw = metadata.get("inference_root")
        inference_root: Path | None = None
        if inference_root_raw is not None and str(inference_root_raw).strip():
            candidate = Path(str(inference_root_raw)).expanduser()
            if not candidate.is_absolute():
                candidate = run_dir / candidate
            inference_root = candidate

        layout_raw = str(metadata.get("inference_layout", "")).strip().lower()
        use_model_subdir: bool | None = None
        if layout_raw == "model_dataset":
            use_model_subdir = True
        elif layout_raw == "dataset":
            use_model_subdir = False

        artifacts = _discover_artifacts(
            run_dir,
            model_name=model_name,
            inference_root=inference_root,
            use_model_subdir=use_model_subdir,
        )
        if not artifacts:
            artifacts = _discover_artifacts(run_dir, model_name=model_name)
        if not artifacts:
            continue

        for dataset_key, artifact_path in artifacts:
            try:
                artifact = _torch_load(artifact_path)
            except Exception:
                log.warning("Failed to load DiBS artifact %s", artifact_path)
                continue

            graph_samples = artifact.get("graph_samples")
            if graph_samples is None:
                continue

            if graph_samples.ndim == 4 and graph_samples.shape[0] == 1:
                graph_samples = graph_samples.squeeze(0)
            elif graph_samples.ndim != 3:
                continue

            # Binarise
            graph_samples = (graph_samples > 0.5).float()
            s, n, _ = graph_samples.shape

            # DAG check: tr(expm(A)) == n  ⟺  A is a DAG.
            # For binary matrices this is exact.
            traces = torch.zeros(s)
            for i in range(s):
                traces[i] = torch.trace(torch.matrix_exp(graph_samples[i]))
            valid_count = int((torch.abs(traces - n) < 0.5).sum().item())
            valid_pct = (valid_count / s * 100.0) if s > 0 else 0.0

            # Threshold metric: mean of samples → threshold → single DAG check
            if include_threshold:
                mean_graph = (graph_samples.mean(dim=0) > 0.5).float()
                tr_mean = torch.trace(torch.matrix_exp(mean_graph))
                threshold_valid = 100.0 if abs(tr_mean.item() - n) < 0.5 else 0.0

            seed = artifact.get("seed", -1)
            idx = artifact.get("idx", -1)

            ds_fam = _as_mapping(fam_meta.get(dataset_key))
            n_nodes = ds_fam.get("n_nodes") if ds_fam else None
            samples_per_task = ds_fam.get("samples_per_task") if ds_fam else None

            base_row = {
                "Model": paper_model_label(model_name),
                "ModelKey": model_name,
                "DatasetKey": dataset_key,
                "Dataset": map_dataset_description(dataset_key),
                "TaskIdx": int(idx) if idx != -1 else int(seed),
                "AxisCategory": axis_category(dataset_key),
                "NNodes": int(n_nodes) if n_nodes is not None else None,
                "SamplesPerTask": int(samples_per_task)
                if samples_per_task is not None
                else None,
            }
            rows.append({**base_row, "Metric": "valid_dag_pct", "Value": valid_pct})
            if include_threshold:
                rows.append(
                    {
                        **base_row,
                        "Metric": "threshold_valid_dag_pct",
                        "Value": threshold_valid,
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Dataset"] = df.apply(
            lambda row: thesis_dataset_label(
                str(row["DatasetKey"]), str(row["Dataset"])
            ),
            axis=1,
        )
    n_metrics = df["Metric"].nunique() if not df.empty else 0
    log.info(
        "Computed DiBS valid DAG metrics for %d rows (%d metrics)",
        len(df),
        n_metrics,
    )
    return df


def generate_valid_dag_shift_figure(
    raw_df: pd.DataFrame,
    run_dirs: Sequence[Path],
    *,
    output_path: Path,
) -> pd.DataFrame:
    """Generate a multi-panel figure showing valid DAG % across shift axes.

    One panel per shift axis (graph, mechanism, compound, nodes, samples).
    AviCi and DiBS are plotted with errorbars; BCNP and BayesDAG are shown
    as horizontal reference lines at 100%.

    Both ``valid_dag_pct`` (sample-level, solid markers) and
    ``threshold_valid_dag_pct`` (thresholded mean graph, transparent markers)
    are shown to distinguish posterior-sample acyclicity from point-estimate
    acyclicity.

    Args:
        raw_df: Long-format raw task DataFrame (must contain ``valid_dag_pct``
            and optionally ``threshold_valid_dag_pct``).
        run_dirs: Run directories (needed to compute DiBS valid_dag_pct from
            inference artifacts).
        output_path: Path for the output PDF figure.

    Returns:
        Aggregated DataFrame used for plotting.
    """
    # ── 1. Gather data for both metrics ───────────────────────────────
    _METRICS = ["valid_dag_pct", "threshold_valid_dag_pct"]
    vdp = raw_df[raw_df["Metric"].isin(_METRICS)].copy()

    # Compute DiBS from artifacts (including threshold) and append
    dibs_vdp = _compute_dibs_valid_dag_pct(run_dirs, include_threshold=True)
    if not dibs_vdp.empty:
        shared_cols = [c for c in vdp.columns if c in dibs_vdp.columns]
        vdp = pd.concat([vdp[shared_cols], dibs_vdp[shared_cols]], ignore_index=True)

    if vdp.empty:
        log.warning("No valid_dag_pct data available; skipping valid DAG shift figure.")
        return pd.DataFrame()

    # Values are already in percentage (0–100) for both raw_df and DiBS artifacts.

    # Models that actually have per-task variation
    variable_models = {"AviCi", "DiBS"}
    # Models that are constant at 100%
    constant_models = {"BCNP", "BayesDAG"}

    # Metric display config: (label_suffix, alpha, marker_size, linestyle)
    _METRIC_STYLE: dict[str, tuple[str, float, int, str]] = {
        "valid_dag_pct": ("Sample", 0.9, 7, "-"),
        "threshold_valid_dag_pct": ("Threshold", 0.35, 6, "--"),
    }

    shift_keys = list(_VALID_DAG_SHIFT_SPECS.keys())
    n_panels = len(shift_keys)
    # 2×3 grid: top row = graph/mechanism/compound, bottom = nodes/samples/empty
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.0 * n_cols, 5.0 * n_rows), squeeze=False
    )
    # Map panel index to (row, col)
    _panel_pos = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

    # Hide unused cell(s)
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) not in _panel_pos[:n_panels]:
                axes[r, c].set_visible(False)

    all_agg_rows: list[pd.DataFrame] = []

    for panel_idx, shift_key in enumerate(shift_keys):
        r, c = _panel_pos[panel_idx]
        ax = axes[r, c]
        axis_cat, axis_title = _VALID_DAG_SHIFT_SPECS[shift_key]

        # Filter to ID + this shift axis
        panel_df = vdp[vdp["AxisCategory"].isin(["id", axis_cat])].copy()

        # For non-transfer axes, restrict to fixed size
        if shift_key not in ("nodes", "samples"):
            panel_df = panel_df[is_fixed_size_task_frame(panel_df)]

        # ── Isolate the shift axis ────────────────────────────────────
        if shift_key == "graph":
            # Use only Linear mechanism (representative) for this overview
            panel_df = panel_df[
                panel_df["DatasetKey"].map(lambda dk: id_mechanism_of(dk) == "linear")
            ].copy()
        elif shift_key == "mechanism":
            # Use ER-20 as representative graph anchor (matches first
            # anchor in the multi-panel mechanism shift figure).
            # Filter both ID and OOD-mech to ER-20 to avoid mixing anchors.
            panel_df = _restrict_to_graph_anchor(panel_df, "er20")
        elif shift_key == "compound":
            # Restrict ID to the same 3 representatives as the main
            # compound figure to avoid cluttering the overview panel.
            _COMPOUND_ID_REPS = {
                ("linear", "er20"),
                ("neuralnet", "sf2"),
                ("gpcde", "er60"),
            }
            id_mask = panel_df["AxisCategory"] == "id"
            keep_id = id_mask & panel_df["DatasetKey"].map(
                lambda dk: (id_mechanism_of(dk), graph_code_of(dk)) in _COMPOUND_ID_REPS
            )
            panel_df = panel_df[keep_id | ~id_mask].copy()

        if panel_df.empty:
            ax.set_visible(False)
            continue

        # Aggregate per (Model, DatasetKey, Metric)
        agg = (
            panel_df.groupby(
                ["Model", "DatasetKey", "Dataset", "AxisCategory", "Metric"],
                dropna=False,
            )["Value"]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )

        # For graph shift panels use short graph-topology labels
        if shift_key == "graph":
            agg["DatasetLabel"] = agg["DatasetKey"].map(_graph_shift_label)
        elif shift_key == "mechanism":
            agg["DatasetLabel"] = agg["DatasetKey"].map(_mech_shift_label)
        else:
            agg["DatasetLabel"] = agg.apply(
                lambda row: thesis_dataset_label(
                    str(row["DatasetKey"]), str(row["Dataset"])
                ),
                axis=1,
            )
        agg["_sort"] = agg["AxisCategory"].map({"id": 0}).fillna(1)
        agg = agg.sort_values(["_sort", "DatasetLabel"]).drop(columns=["_sort"])
        all_agg_rows.append(agg)

        # Dataset ordering is derived from sample-level metric only to keep
        # x-axis stable; threshold rows share the same dataset labels.
        agg_sample = agg[agg["Metric"] == "valid_dag_pct"]
        datasets = list(agg_sample["DatasetLabel"].unique())
        axis_lookup = (
            agg_sample[["DatasetLabel", "AxisCategory"]]
            .drop_duplicates()
            .set_index("DatasetLabel")["AxisCategory"]
            .to_dict()
        )
        n_datasets = len(datasets)
        variable_present = [
            m
            for m in PAPER_MODEL_LABELS.values()
            if m in variable_models and m in agg["Model"].unique()
        ]
        n_var = len(variable_present)
        width = 0.6
        offset_step = width / max(n_var, 1)
        x_base = np.arange(n_datasets)

        # Plot both metrics for each variable model
        for metric_key, (label_sfx, alpha, msize, _ls) in _METRIC_STYLE.items():
            metric_agg = agg[agg["Metric"] == metric_key]
            if metric_agg.empty:
                continue

            for model_idx, model in enumerate(variable_present):
                model_agg = metric_agg[metric_agg["Model"] == model]
                xs: list[float] = []
                means: list[float] = []
                sems: list[float] = []
                for i, ds in enumerate(datasets):
                    row = model_agg[model_agg["DatasetLabel"] == ds]
                    if row.empty:
                        continue
                    offset = (model_idx - n_var / 2 + 0.5) * offset_step
                    xs.append(float(x_base[i]) + offset)
                    means.append(float(row.iloc[0]["Mean"]))
                    sems.append(float(row.iloc[0]["SEM"]))
                if xs:
                    # Use filled marker for sample-level, open for threshold
                    marker = MODEL_MARKERS.get(model, "o")
                    fillstyle = "full" if metric_key == "valid_dag_pct" else "none"
                    label = (
                        f"{model} ({label_sfx})"
                        if metric_key == "threshold_valid_dag_pct"
                        else model
                    )
                    ax.errorbar(
                        xs,
                        means,
                        yerr=sems,
                        fmt=marker,
                        label=label,
                        color=_model_color(model),
                        capsize=3,
                        markersize=msize,
                        alpha=alpha,
                        fillstyle=fillstyle,
                    )

        # Reference lines for constant-100% models
        for const_model in sorted(constant_models):
            ax.axhline(
                100.0,
                color=_model_color(const_model),
                linestyle="--",
                linewidth=1.2,
                alpha=0.6,
                label=f"{const_model} (100%)",
            )

        ax.set_xticks(x_base)
        ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Valid DAG (%) $\\uparrow$", fontsize=10)
        ax.set_ylim(-5, 110)
        if shift_key == "graph":
            panel_title = f"{axis_title} (Linear)"
        elif shift_key == "mechanism":
            panel_title = f"{axis_title} (ER-20)"
        else:
            panel_title = axis_title
        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        # Grey ID region
        id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")
        if 0 < id_count < len(datasets):
            ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
            ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

    # Shared legend on top — deduplicate handles from first visible axis
    all_handles: list = []
    all_labels: list[str] = []
    seen: set[str] = set()
    for ax in axes.flatten():
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
            ncol=min(len(all_labels), 6),
            fontsize=9,
            frameon=False,
        )

    fig.suptitle(
        "DAG Structural Validity Under Distribution Shift",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    combined = (
        pd.concat(all_agg_rows, ignore_index=True) if all_agg_rows else pd.DataFrame()
    )
    save_figure_data(output_path, combined)
    return combined


# ── Error decomposition (FP / FN / Reversed) shift table ──────────────

_ERROR_DECOMP_SHIFT_SPECS: dict[str, tuple[str, str]] = {
    "graph": ("graph", "Graph Shift"),
    "mechanism": ("mechanism", "Mechanism Shift"),
    "noise": ("noise", "Noise Shift"),
    "compound": ("compound", "Compound Shift"),
}


def generate_error_decomposition_table(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
    model_filter: Sequence[str] | None = None,
    label_prefix: str = "",
) -> pd.DataFrame:
    """Generate a LaTeX table decomposing SHD into FP/FN/reversed per model per shift.

    Layout: rows = models, column groups = shift axes (graph, mechanism,
    noise, compound), sub-columns = mean FP / FN / Reversed counts.

    Args:
        raw_df: Long-format raw task DataFrame (must contain ``fp_count``,
            ``fn_count``, ``reversed_count``).
        output_path: Path for the output ``.tex`` file.

    Returns:
        Aggregated DataFrame used for the table.
    """
    needed = {"fp_count", "fn_count", "reversed_count"}
    available = set(raw_df["Metric"].unique()) if not raw_df.empty else set()
    if not needed.issubset(available):
        log.warning(
            "Missing error decomposition metrics %s; skipping table.",
            needed - available,
        )
        return pd.DataFrame()

    error_df = raw_df[raw_df["Metric"].isin(needed)].copy()
    error_df = error_df[is_fixed_size_task_frame(error_df)]

    if error_df.empty:
        log.warning("No fixed-size error decomposition data; skipping table.")
        return pd.DataFrame()

    from causal_meta.analysis.diagnostics.failure_modes import _pivot_raw_wide

    wide = _pivot_raw_wide(error_df)
    if wide.empty:
        return pd.DataFrame()

    for col in needed:
        if col not in wide.columns:
            log.warning(
                "Column %s not found after pivot; skipping error decomp table.", col
            )
            return pd.DataFrame()

    wide["AxisCategory"] = wide["DatasetKey"].map(axis_category)

    shift_specs = list(_ERROR_DECOMP_SHIFT_SPECS.items())
    all_models = _resolve_models(model_filter)
    models = [m for m in all_models if m in wide["Model"].unique()]

    if not models:
        return pd.DataFrame()

    all_agg: list[pd.DataFrame] = []

    # Aggregate: mean FP / FN / reversed per (Model, ShiftAxis)
    for shift_key, (axis_cat, axis_title) in shift_specs:
        panel = wide[wide["AxisCategory"].isin(["id", axis_cat])].copy()

        # Apply the same filtering as the figure version
        if shift_key == "graph":
            panel = panel[
                panel["DatasetKey"].map(lambda dk: id_mechanism_of(dk) == "linear")
            ].copy()
        elif shift_key == "mechanism":
            panel = _restrict_to_graph_anchor(panel, "er20")
        elif shift_key == "compound":
            _COMPOUND_ID_REPS = {
                ("linear", "er20"),
                ("neuralnet", "sf2"),
                ("gpcde", "er60"),
            }
            id_mask = panel["AxisCategory"] == "id"
            keep_id = id_mask & panel["DatasetKey"].map(
                lambda dk: (id_mechanism_of(dk), graph_code_of(dk)) in _COMPOUND_ID_REPS
            )
            panel = panel[keep_id | ~id_mask].copy()

        # Only OOD rows for the degradation summary (ID is reference)
        ood_panel = panel[panel["AxisCategory"] == axis_cat]
        if ood_panel.empty:
            continue

        agg = (
            ood_panel.groupby("Model")[list(needed)].agg(["mean", "sem"]).reset_index()
        )
        agg.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0] for col in agg.columns
        ]
        agg["ShiftAxis"] = axis_title
        all_agg.append(agg)

    if not all_agg:
        return pd.DataFrame()

    combined = pd.concat(all_agg, ignore_index=True)

    # Build LaTeX table
    n_shifts = len(shift_specs)
    comp_labels = [("FP", "fp_count"), ("FN", "fn_count"), ("Rev.", "reversed_count")]
    n_sub = len(comp_labels)

    lines: list[str] = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(
        r"\caption{Mean SHD error decomposition (false positive, false negative,"
        r" and reversed edges) per model under each OOD shift axis. Values are"
        r" averaged over all OOD families in the respective shift axis"
        r" (graph, mechanism, noise, and compound).}"
    )
    lines.append(rf"\label{{tab:{label_prefix}error_decomposition}}")

    # Column spec: Model + 3 sub-cols per shift axis
    col_spec = "l" + ("|" + "r" * n_sub) * n_shifts
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: shift axis group headers
    header1 = r"\textbf{Model}"
    for _sk, (_ac, at) in shift_specs:
        header1 += rf" & \multicolumn{{{n_sub}}}{{c}}{{\textbf{{{at}}}}}"
    header1 += r" \\"
    lines.append(header1)

    # Cmidrules
    cmidrules = ""
    col_start = 2
    for _ in shift_specs:
        col_end = col_start + n_sub - 1
        cmidrules += rf"\cmidrule(lr){{{col_start}-{col_end}}} "
        col_start = col_end + 1
    lines.append(cmidrules)

    # Header row 2: sub-column labels
    header2 = ""
    for _ in shift_specs:
        for label, _ in comp_labels:
            header2 += rf" & \textbf{{{label}}}"
    header2 += r" \\"
    lines.append(header2)
    lines.append(r"\midrule")

    # Data rows
    # Pre-compute best (lowest) value per metric per shift axis.
    best_vals: dict[tuple[str, str], float] = {}
    for _sk, (_ac, at) in shift_specs:
        for _, metric_key in comp_labels:
            col = f"{metric_key}_mean"
            axis_rows = combined[combined["ShiftAxis"] == at]
            if not axis_rows.empty and col in axis_rows.columns:
                valid = axis_rows[col].dropna()
                if not valid.empty:
                    best_vals[(at, metric_key)] = float(valid.min())

    for model in models:
        row_str = model
        for _sk, (_ac, at) in shift_specs:
            model_row = combined[
                (combined["Model"] == model) & (combined["ShiftAxis"] == at)
            ]
            if model_row.empty:
                for _ in comp_labels:
                    row_str += " & --"
            else:
                r = model_row.iloc[0]
                for _, metric_key in comp_labels:
                    mean_val = r.get(f"{metric_key}_mean", float("nan"))
                    sem_val = r.get(f"{metric_key}_sem", float("nan"))
                    if np.isnan(mean_val):
                        row_str += " & --"
                    else:
                        cell = rf"${mean_val:.1f} \pm {sem_val:.1f}$"
                        best = best_vals.get((at, metric_key))
                        if best is not None and abs(mean_val - best) < 1e-9:
                            cell = r"\textbf{" + cell + "}"
                        row_str += " & " + cell
        row_str += r" \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    output_path.write_text("\n".join(lines) + "\n")
    log.info("Saved error decomposition table to %s", output_path)
    return combined


def generate_error_decomposition_figure(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
    model_filter: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate a grouped stacked-bar chart of FP / FN / reversed per model per shift.

    Layout: one panel (axis group) per column, models on the x-axis within each
    panel.  Each bar is stacked with three segments: false positives (FP),
    false negatives (FN), and reversed edges.

    Args:
        raw_df: Long-format raw task DataFrame (must contain ``fp_count``,
            ``fn_count``, ``reversed_count``).
        output_path: Path for the output PDF figure.
        model_filter: If given, only plot these models (display names).

    Returns:
        Aggregated DataFrame used for plotting.
    """
    needed = {"fp_count", "fn_count", "reversed_count"}
    available = set(raw_df["Metric"].unique()) if not raw_df.empty else set()
    if not needed.issubset(available):
        log.warning(
            "Missing error decomposition metrics %s; skipping figure.",
            needed - available,
        )
        return pd.DataFrame()

    error_df = raw_df[raw_df["Metric"].isin(needed)].copy()
    error_df = error_df[is_fixed_size_task_frame(error_df)]

    if error_df.empty:
        log.warning("No fixed-size error decomposition data; skipping figure.")
        return pd.DataFrame()

    from causal_meta.analysis.diagnostics.failure_modes import _pivot_raw_wide

    wide = _pivot_raw_wide(error_df)
    if wide.empty:
        return pd.DataFrame()

    for col in needed:
        if col not in wide.columns:
            log.warning(
                "Column %s not found after pivot; skipping error decomp figure.", col
            )
            return pd.DataFrame()

    wide["AxisCategory"] = wide["DatasetKey"].map(axis_category)

    shift_specs = list(_ERROR_DECOMP_SHIFT_SPECS.items())
    all_models = _resolve_models(model_filter)
    models = [m for m in all_models if m in wide["Model"].unique()]

    if not models:
        return pd.DataFrame()

    all_agg: list[pd.DataFrame] = []

    # Aggregate: mean FP / FN / reversed per (Model, ShiftAxis)
    for shift_key, (axis_cat, axis_title) in shift_specs:
        panel = wide[wide["AxisCategory"].isin(["id", axis_cat])].copy()

        if shift_key == "graph":
            panel = panel[
                panel["DatasetKey"].map(lambda dk: id_mechanism_of(dk) == "linear")
            ].copy()
        elif shift_key == "mechanism":
            panel = _restrict_to_graph_anchor(panel, "er20")
        elif shift_key == "compound":
            _COMPOUND_ID_REPS = {
                ("linear", "er20"),
                ("neuralnet", "sf2"),
                ("gpcde", "er60"),
            }
            id_mask = panel["AxisCategory"] == "id"
            keep_id = id_mask & panel["DatasetKey"].map(
                lambda dk: (id_mechanism_of(dk), graph_code_of(dk)) in _COMPOUND_ID_REPS
            )
            panel = panel[keep_id | ~id_mask].copy()

        ood_panel = panel[panel["AxisCategory"] == axis_cat]
        if ood_panel.empty:
            continue

        agg = ood_panel.groupby("Model")[list(needed)].mean().reset_index()
        agg["ShiftAxis"] = axis_title
        all_agg.append(agg)

    if not all_agg:
        return pd.DataFrame()

    combined = pd.concat(all_agg, ignore_index=True)

    # ── Plot: one panel per shift axis ────────────────────────────────
    n_panels = len(shift_specs)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.2 * n_panels, 4.5), sharey=False, squeeze=False
    )
    axes_flat = axes[0]

    comp_keys = ERROR_SPECS

    for panel_idx, (_shift_key, (_axis_cat, axis_title)) in enumerate(shift_specs):
        ax = axes_flat[panel_idx]
        panel_data = combined[combined["ShiftAxis"] == axis_title]

        x = np.arange(len(models))
        bottoms = np.zeros(len(models))

        for comp_col, comp_label, comp_color in comp_keys:
            heights = []
            for model in models:
                row = panel_data[panel_data["Model"] == model]
                if row.empty:
                    heights.append(0.0)
                else:
                    heights.append(float(row[comp_col].iloc[0]))
            heights_arr = np.array(heights)
            ax.bar(
                x,
                heights_arr,
                bottom=bottoms,
                width=0.55,
                label=comp_label if panel_idx == 0 else None,
                color=comp_color,
                edgecolor="white",
                linewidth=0.5,
            )
            bottoms += heights_arr

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9, rotation=30, ha="right")
        ax.set_title(axis_title, fontsize=11, fontweight="bold")
        if panel_idx == 0:
            ax.set_ylabel("Mean edge error count", fontsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Shared legend on top
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Error Type",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(labels),
            fontsize=10,
            frameon=False,
        )

    fig.suptitle(
        "SHD Error Decomposition Across Shift Axes",
        fontsize=13,
        fontweight="bold",
        y=1.07,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    log.info("Saved error decomposition figure to %s", output_path)
    save_figure_data(output_path, combined)
    return combined


# ── Extreme stress-test constants (used by compound+stress merge) ──────

_EXTREME_FAMILIES: list[str] = [
    "ood_both_sbm_periodic_d60_n100",
    "ood_both_ws_pnl_tanh_d60_n50",
    "ood_both_grg_logistic_map_d60_n50",
]
"""Dataset keys for the three extreme compound-shift stress-test families.

These families combine OOD graph + OOD mechanism with large node counts and
severely reduced sample budgets (d=60, n=50 or n=100), so they are excluded
by :func:`is_fixed_size_task_frame` and do not appear in the standard
fixed-size shift figures.
"""

_EXTREME_LABELS: dict[str, str] = {
    "ood_both_sbm_periodic_d60_n100": "SBM × Periodic\n($d{=}60$, $n{=}100$)",
    "ood_both_ws_pnl_tanh_d60_n50": "WS × PNL-tanh\n($d{=}60$, $n{=}50$)",
    "ood_both_grg_logistic_map_d60_n50": "GRG × Logistic\n($d{=}60$, $n{=}50$)",
}


# ── RQ1/RQ2 cross-axis summary table ──────────────────────────────────

_SUMMARY_TABLE_AXES: dict[str, str] = {
    "id": "ID",
    "graph": "Graph",
    "mechanism": "Mech.",
    "noise": "Noise",
    "compound": "Compound",
    "nodes": "Nodes",
    "samples": "Samples",
}
"""Column groups for the cross-axis summary table."""

_SUMMARY_TABLE_METRICS: list[tuple[str, str, bool]] = [
    ("ne-sid", r"ne-SID $\downarrow$", False),
    ("e-edgef1", r"E-F1 $\uparrow$", True),
    ("ne-shd", r"ne-SHD $\downarrow$", False),
]
"""(metric_key, display_label, higher_is_better) for the summary table."""


def generate_cross_axis_summary_table(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
    model_filter: Sequence[str] | None = None,
    label_prefix: str = "",
) -> pd.DataFrame:
    """Generate a LaTeX table summarising three core metrics per model per shift axis.

    Rows = models, column groups = shift axes (ID, Graph, Mechanism, Noise,
    Compound, Nodes, Samples), sub-columns = ne-SID, E-F1, ne-SHD.
    Per-column best values are bolded.

    For the four fixed-size axes (graph, mechanism, noise, compound) only
    ``d=20, n=500`` families are included.  Transfer axes (nodes, samples)
    include all sizes.  The ID column always uses fixed-size families.

    Args:
        raw_df: Long-format raw task DataFrame.
        output_path: Path for the output ``.tex`` file.
        model_filter: If given, only include these model display names.

    Returns:
        Aggregated DataFrame used for the table.
    """
    metric_keys = [m for m, _, _ in _SUMMARY_TABLE_METRICS]
    subset = raw_df[raw_df["Metric"].isin(metric_keys)].copy()
    if subset.empty:
        log.warning("No data for cross-axis summary table; skipping.")
        return pd.DataFrame()

    fixed = subset[is_fixed_size_task_frame(subset)]
    all_models = _resolve_models(model_filter)
    models = [m for m in all_models if m in subset["Model"].unique()]
    if not models:
        log.warning("No models for cross-axis summary table; skipping.")
        return pd.DataFrame()

    # Aggregate: mean ± SEM per (Model, AxisCategory, Metric)
    all_agg: list[pd.DataFrame] = []
    for axis_key, axis_label in _SUMMARY_TABLE_AXES.items():
        if axis_key in ("nodes", "samples"):
            pool = subset[subset["AxisCategory"] == axis_key]
        elif axis_key == "id":
            pool = fixed[fixed["AxisCategory"] == "id"]
        else:
            pool = fixed[fixed["AxisCategory"] == axis_key]

        if pool.empty:
            continue

        agg = (
            pool.groupby(["Model", "Metric"], dropna=False)["Value"]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )
        agg["Axis"] = axis_label
        all_agg.append(agg)

    if not all_agg:
        return pd.DataFrame()

    combined = pd.concat(all_agg, ignore_index=True)

    # Determine per-column best (best model for each axis × metric)
    best_lookup: dict[tuple[str, str], float] = {}
    for _, row in combined.iterrows():
        key = (str(row["Axis"]), str(row["Metric"]))
        mean = float(row["Mean"])
        higher = next(
            (hib for mk, _, hib in _SUMMARY_TABLE_METRICS if mk == row["Metric"]),
            False,
        )
        current = best_lookup.get(key)
        if current is None:
            best_lookup[key] = mean
        elif higher and mean > current:
            best_lookup[key] = mean
        elif not higher and mean < current:
            best_lookup[key] = mean

    # ── Build LaTeX ────────────────────────────────────────────────────
    axes_list = [
        (ak, al)
        for ak, al in _SUMMARY_TABLE_AXES.items()
        if al in combined["Axis"].unique()
    ]
    n_axes = len(axes_list)
    n_sub = len(_SUMMARY_TABLE_METRICS)

    # Split axes into two groups for a landscape-friendly two-part table
    # when there are more than 4 axes.
    if n_axes > 4:
        split = (n_axes + 1) // 2  # e.g. 4 and 3 for 7 axes
        axis_groups = [axes_list[:split], axes_list[split:]]
    else:
        axis_groups = [axes_list]

    lines: list[str] = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(
        r"\caption{Cross-axis performance summary for the amortised models."
        r" Each cell reports the task-level mean $\pm$ SEM for the indicated"
        r" metric and shift axis."
        r" Per-column best values (excluding Random) are \textbf{bolded}.}"
    )
    lines.append(rf"\label{{tab:{label_prefix}cross_axis_summary}}")

    for grp_idx, grp_axes in enumerate(axis_groups):
        n_grp = len(grp_axes)
        lines.append(r"\resizebox{\textwidth}{!}{%")

        col_spec = "l" + ("|" + "r" * n_sub) * n_grp
        lines.append(r"\begin{tabular}{" + col_spec + "}")
        lines.append(r"\toprule")

        # Header row 1: axis group headers
        header1 = r"\textbf{Model}"
        for _, axis_label in grp_axes:
            header1 += rf" & \multicolumn{{{n_sub}}}{{c}}{{\textbf{{{axis_label}}}}}"
        header1 += r" \\"
        lines.append(header1)

        # Cmidrules
        cmidrules = ""
        col_start = 2
        for _ in grp_axes:
            col_end = col_start + n_sub - 1
            cmidrules += rf"\cmidrule(lr){{{col_start}-{col_end}}} "
            col_start = col_end + 1
        lines.append(cmidrules)

        # Header row 2: metric sub-column labels
        header2 = ""
        for _ in grp_axes:
            for _, mlabel, _ in _SUMMARY_TABLE_METRICS:
                header2 += rf" & \textbf{{{mlabel}}}"
        header2 += r" \\"
        lines.append(header2)
        lines.append(r"\midrule")

        # Data rows
        for model in models:
            row_str = model
            for _, axis_label in grp_axes:
                for metric_key, _, higher_is_better in _SUMMARY_TABLE_METRICS:
                    cell_row = combined[
                        (combined["Model"] == model)
                        & (combined["Axis"] == axis_label)
                        & (combined["Metric"] == metric_key)
                    ]
                    if cell_row.empty:
                        row_str += " & --"
                        continue
                    mean = float(cell_row.iloc[0]["Mean"])
                    sem = float(cell_row.iloc[0]["SEM"])
                    cell = format_value(mean, sem)
                    best_val = best_lookup.get((axis_label, metric_key))
                    # Exclude Random from bolding contest
                    if (
                        best_val is not None
                        and model != "Random"
                        and abs(mean - best_val) < 1e-6
                    ):
                        cell = _bold_if_best(cell, is_best=True)
                    row_str += f" & {cell}"
            row_str += r" \\"
            lines.append(row_str)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}")

        # Add vertical spacing between sub-tables
        if grp_idx < len(axis_groups) - 1:
            lines.append(r"\vspace{0.5em}")

    lines.append(r"\end{table}")

    output_path.write_text("\n".join(lines) + "\n")
    log.info("Saved cross-axis summary table to %s", output_path)
    return combined


# ── Structural decomposition summary table ────────────────────────────

_STRUCTURAL_TABLE_METRICS: list[tuple[str, str, bool]] = [
    ("skeleton_f1", r"Skel.\ F1 $\uparrow$", True),
    ("orientation_accuracy", r"Orient.\ Acc $\uparrow$", True),
    ("sparsity_ratio", r"Sparsity", False),  # neither direction is universally better
]
"""(metric_key, display_label, higher_is_better) for the structural table.

For ``sparsity_ratio`` bolding is disabled because neither direction is
universally better — it depends on the ground-truth graph density.
"""


def generate_structural_summary_table(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
    model_filter: Sequence[str] | None = None,
    label_prefix: str = "",
) -> pd.DataFrame:
    """Generate a LaTeX table for skeleton F1, orientation accuracy, and sparsity.

    Layout mirrors :func:`generate_cross_axis_summary_table`: rows = models,
    column groups = shift axes, sub-columns = the three structural metrics.

    Args:
        raw_df: Long-format raw task DataFrame.
        output_path: Path for the output ``.tex`` file.
        model_filter: If given, only include these model display names.

    Returns:
        Aggregated DataFrame used for the table.
    """
    metric_keys = [m for m, _, _ in _STRUCTURAL_TABLE_METRICS]
    subset = raw_df[raw_df["Metric"].isin(metric_keys)].copy()
    if subset.empty:
        log.warning("No data for structural summary table; skipping.")
        return pd.DataFrame()

    fixed = subset[is_fixed_size_task_frame(subset)]
    all_models = _resolve_models(model_filter)
    models = [m for m in all_models if m in subset["Model"].unique()]
    if not models:
        log.warning("No models for structural summary table; skipping.")
        return pd.DataFrame()

    # Aggregate: mean ± SEM per (Model, AxisCategory, Metric)
    all_agg: list[pd.DataFrame] = []
    for axis_key, axis_label in _SUMMARY_TABLE_AXES.items():
        if axis_key in ("nodes", "samples"):
            pool = subset[subset["AxisCategory"] == axis_key]
        elif axis_key == "id":
            pool = fixed[fixed["AxisCategory"] == "id"]
        else:
            pool = fixed[fixed["AxisCategory"] == axis_key]

        if pool.empty:
            continue

        agg = (
            pool.groupby(["Model", "Metric"], dropna=False)["Value"]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )
        agg["Axis"] = axis_label
        all_agg.append(agg)

    if not all_agg:
        return pd.DataFrame()

    combined = pd.concat(all_agg, ignore_index=True)

    # Determine per-column best (best model for each axis × metric)
    best_lookup: dict[tuple[str, str], float] = {}
    for _, row in combined.iterrows():
        key = (str(row["Axis"]), str(row["Metric"]))
        mean = float(row["Mean"])
        higher = next(
            (hib for mk, _, hib in _STRUCTURAL_TABLE_METRICS if mk == row["Metric"]),
            False,
        )
        # Skip bolding for sparsity_ratio (higher_is_better=False is overloaded)
        if row["Metric"] == "sparsity_ratio":
            continue
        current = best_lookup.get(key)
        if current is None:
            best_lookup[key] = mean
        elif higher and mean > current:
            best_lookup[key] = mean
        elif not higher and mean < current:
            best_lookup[key] = mean

    # ── Build LaTeX ────────────────────────────────────────────────────
    axes_list = [
        (ak, al)
        for ak, al in _SUMMARY_TABLE_AXES.items()
        if al in combined["Axis"].unique()
    ]
    n_axes = len(axes_list)
    n_sub = len(_STRUCTURAL_TABLE_METRICS)

    lines: list[str] = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(
        r"\caption{Structural error decomposition across shift axes."
        r" Skeleton F1 measures undirected edge recovery,"
        r" Orientation Accuracy measures correct directionality of true"
        r" positives, and Sparsity Ratio shows predicted density relative"
        r" to ground truth. Per-column best values (excluding Random)"
        r" are \textbf{bolded} where applicable.}"
    )
    lines.append(rf"\label{{tab:{label_prefix}structural_summary}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    col_spec = "l" + ("|" + "r" * n_sub) * n_axes
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header1 = r"\textbf{Model}"
    for _, axis_label in axes_list:
        header1 += rf" & \multicolumn{{{n_sub}}}{{c}}{{\textbf{{{axis_label}}}}}"
    header1 += r" \\"
    lines.append(header1)

    cmidrules = ""
    col_start = 2
    for _ in axes_list:
        col_end = col_start + n_sub - 1
        cmidrules += rf"\cmidrule(lr){{{col_start}-{col_end}}} "
        col_start = col_end + 1
    lines.append(cmidrules)

    header2 = ""
    for _ in axes_list:
        for _, mlabel, _ in _STRUCTURAL_TABLE_METRICS:
            header2 += rf" & \textbf{{{mlabel}}}"
    header2 += r" \\"
    lines.append(header2)
    lines.append(r"\midrule")

    for model in models:
        row_str = model
        for _, axis_label in axes_list:
            for metric_key, _, higher_is_better in _STRUCTURAL_TABLE_METRICS:
                cell_row = combined[
                    (combined["Model"] == model)
                    & (combined["Axis"] == axis_label)
                    & (combined["Metric"] == metric_key)
                ]
                if cell_row.empty:
                    row_str += " & --"
                    continue
                mean = float(cell_row.iloc[0]["Mean"])
                sem = float(cell_row.iloc[0]["SEM"])
                cell = format_value(mean, sem)
                best_val = best_lookup.get((axis_label, metric_key))
                if (
                    best_val is not None
                    and model != "Random"
                    and abs(mean - best_val) < 1e-6
                ):
                    cell = _bold_if_best(cell, is_best=True)
                row_str += f" & {cell}"
        row_str += r" \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    output_path.write_text("\n".join(lines) + "\n")
    log.info("Saved structural summary table to %s", output_path)
    return combined


# ── RQ1: Non-uniform degradation bar chart ────────────────────────────


def generate_rq1_nonuniform_degradation_bar(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
    model_filter: Sequence[str] | None = None,
    metric_name: str = "ne-sid",
) -> pd.DataFrame:
    """Generate a grouped bar chart showing per-family degradation ratios.

    For each OOD dataset family the degradation ratio
    ``mean(OOD metric) / mean(ID metric)`` is computed, where the ID baseline
    is anchored per shift axis.  Families are grouped by shift axis and bars
    are coloured by model, directly answering *"Does OOD performance degrade
    uniformly across shift axes?"* — the answer is visibly **no**.

    Args:
        raw_df: Long-format raw task DataFrame.
        output_path: Path for the output PDF figure.
        model_filter: Optional list of model display names to include.
        metric_name: Metric to use for the ratio (default ``"ne-sid"``).

    Returns:
        DataFrame with columns ``Model``, ``Family``, ``Axis``, ``Ratio``.
    """
    subset = raw_df[raw_df["Metric"].eq(metric_name)].copy()
    fixed_subset = subset[is_fixed_size_task_frame(subset)]

    if fixed_subset.empty:
        log.warning(
            "No %s data for non-uniform degradation bar; skipping.", metric_name
        )
        return pd.DataFrame()

    models = _resolve_models(model_filter)

    # Only fixed-size axes (graph, mechanism, noise, compound)
    _BAR_AXES: dict[str, str] = {
        "graph": "Graph",
        "mechanism": "Mechanism",
        "noise": "Noise",
        "compound": "Compound",
    }

    rows: list[dict[str, object]] = []
    for shift_key, shift_label in _BAR_AXES.items():
        id_data = _degradation_id_subset(fixed_subset, shift_key)
        id_means = id_data.groupby("Model")["Value"].mean().to_dict()

        ood_data = fixed_subset[fixed_subset["AxisCategory"] == shift_key]
        # Per-family breakdown
        for (model, dk), grp in ood_data.groupby(["Model", "DatasetKey"]):
            model_str = str(model)
            if model_str not in models:
                continue
            id_val = id_means.get(model_str)
            if id_val is None or id_val <= 0:
                continue
            ood_mean = float(grp["Value"].mean())
            ratio = ood_mean / id_val
            # Short family label
            family_label = thesis_dataset_label(
                str(dk),
                str(grp["Dataset"].iloc[0]) if "Dataset" in grp.columns else str(dk),
            )
            rows.append(
                {
                    "Model": model_str,
                    "Family": family_label,
                    "Axis": shift_label,
                    "Ratio": ratio,
                }
            )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        log.warning("No data for non-uniform degradation bar; skipping.")
        return pd.DataFrame()

    # Sort families within each axis by mean degradation (worst first)
    axis_order = list(_BAR_AXES.values())
    family_sort = (
        result_df.groupby(["Axis", "Family"])["Ratio"]
        .mean()
        .reset_index()
        .sort_values(["Axis", "Ratio"], ascending=[True, False])
    )

    # Build figure: one panel per axis
    n_axes = len(axis_order)
    fig, axes_arr = plt.subplots(1, n_axes, figsize=(4.5 * n_axes, 4.5), squeeze=False)

    for ax_idx, axis_label in enumerate(axis_order):
        ax = axes_arr[0, ax_idx]
        ax_data = result_df[result_df["Axis"] == axis_label]
        if ax_data.empty:
            ax.set_visible(False)
            continue

        # Family order: worst average degradation first
        fam_order = list(
            family_sort[family_sort["Axis"] == axis_label]["Family"].values
        )
        present_models = [m for m in models if m in ax_data["Model"].unique()]
        n_models = len(present_models)
        n_fam = len(fam_order)
        bar_width = 0.7 / max(n_models, 1)
        x_base = np.arange(n_fam)

        for m_idx, model in enumerate(present_models):
            model_data = ax_data[ax_data["Model"] == model]
            vals = []
            for fam in fam_order:
                fam_row = model_data[model_data["Family"] == fam]
                vals.append(
                    float(fam_row["Ratio"].iloc[0]) if not fam_row.empty else 0.0
                )
            offset = (m_idx - n_models / 2 + 0.5) * bar_width
            ax.bar(
                x_base + offset,
                vals,
                width=bar_width,
                color=_model_color(model),
                label=model if ax_idx == 0 else None,
                edgecolor="white",
                linewidth=0.5,
            )

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.set_xticks(x_base)
        ax.set_xticklabels(fam_order, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{axis_label} Shift", fontsize=11, fontweight="bold")
        ax.set_ylabel("OOD / ID Ratio" if ax_idx == 0 else "", fontsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Shared legend
    handles, labels = axes_arr[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(labels),
            fontsize=9,
            frameon=False,
        )

    fig.suptitle(
        "Per-Family Degradation Ratio by Shift Axis",
        fontsize=13,
        fontweight="bold",
        y=1.06,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    log.info("Saved non-uniform degradation bar chart to %s", output_path)
    save_figure_data(output_path, result_df)
    return result_df


# ── RQ1: AviCi-only DAG validity figure ──────────────────────────────


def generate_rq1_avici_dag_validity_figure(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
) -> pd.DataFrame:
    """Generate an AviCi-only valid DAG figure with sampled vs. thresholded markers.

    Unlike the all-model :func:`generate_valid_dag_shift_figure`, this figure:

    * Shows **only AviCi** (no DiBS, no BCNP/BayesDAG reference lines).
    * Uses a **three-panel layout** (Graph, Mechanism, Noise+Compound) to keep
      the figure compact.
    * **Includes all OOD mechanism families** in the graph panel, not just
      linear—this is important because the linear case is where AviCi
      struggles most, and showing all families reveals the mechanism-dependent
      variation in DAG validity.
    * Distinguishes sampled (solid) vs. thresholded (open) DAG rates per the
      methodology.

    Does **not** require DiBS posterior artifacts (no run_dirs parameter).

    Args:
        raw_df: Long-format raw task DataFrame (must contain ``valid_dag_pct``
            and optionally ``threshold_valid_dag_pct``).
        output_path: Path for the output PDF figure.

    Returns:
        Aggregated DataFrame used for plotting.
    """
    _METRICS = ["valid_dag_pct", "threshold_valid_dag_pct"]
    vdp = raw_df[
        (raw_df["Metric"].isin(_METRICS)) & (raw_df["Model"] == "AviCi")
    ].copy()

    if vdp.empty:
        log.warning("No AviCi valid_dag_pct data; skipping RQ1 DAG validity figure.")
        return pd.DataFrame()

    # Metric display config
    _METRIC_STYLE: dict[str, tuple[str, float, int]] = {
        "valid_dag_pct": ("Sampled", 0.9, 7),
        "threshold_valid_dag_pct": ("Thresholded", 0.45, 6),
    }

    # Panel specs: (shift_key, axis_categories, title, filter_fn)
    # Graph panel: show ALL mechanism families (ID + OOD-graph), not just linear
    # Mechanism panel: show ER-20 anchor
    # Combined panel: noise + compound together for compactness

    _PANELS: list[tuple[str, list[str], str]] = [
        ("graph", ["id", "graph"], "Graph Shift"),
        ("mechanism", ["id", "mechanism"], "Mechanism Shift (ER-20)"),
        ("combined", ["id", "noise", "compound"], "Noise + Compound Shift"),
    ]

    n_panels = len(_PANELS)
    fig, axes_arr = plt.subplots(
        1, n_panels, figsize=(5.5 * n_panels, 5.0), squeeze=False
    )

    all_agg: list[pd.DataFrame] = []
    avici_color = _model_color("AviCi")

    for panel_idx, (panel_key, cats, panel_title) in enumerate(_PANELS):
        ax = axes_arr[0, panel_idx]
        panel_df = vdp[vdp["AxisCategory"].isin(cats)].copy()

        # Apply fixed-size filter
        panel_df = panel_df[is_fixed_size_task_frame(panel_df)]

        # Panel-specific filtering
        if panel_key == "mechanism":
            panel_df = _restrict_to_graph_anchor(panel_df, "er20")

        if panel_df.empty:
            ax.set_visible(False)
            continue

        # Aggregate per (DatasetKey, Metric)
        agg = (
            panel_df.groupby(
                ["DatasetKey", "Dataset", "AxisCategory", "Metric"],
                dropna=False,
            )["Value"]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )

        # Labels
        if panel_key == "graph":
            agg["DatasetLabel"] = agg["DatasetKey"].map(_graph_shift_label)
        elif panel_key == "mechanism":
            agg["DatasetLabel"] = agg["DatasetKey"].map(_mech_shift_label)
        else:
            agg["DatasetLabel"] = agg.apply(
                lambda row: thesis_dataset_label(
                    str(row["DatasetKey"]), str(row["Dataset"])
                ),
                axis=1,
            )

        agg["_sort"] = agg["AxisCategory"].map({"id": 0}).fillna(1)
        agg = agg.sort_values(["_sort", "DatasetLabel"]).drop(columns=["_sort"])
        all_agg.append(agg)

        # x-axis from sample-level metric
        agg_sample = agg[agg["Metric"] == "valid_dag_pct"]
        datasets = list(agg_sample["DatasetLabel"].unique())
        axis_lookup = (
            agg_sample[["DatasetLabel", "AxisCategory"]]
            .drop_duplicates()
            .set_index("DatasetLabel")["AxisCategory"]
            .to_dict()
        )
        n_ds = len(datasets)
        x_base = np.arange(n_ds)

        for metric_key, (label_sfx, alpha, msize) in _METRIC_STYLE.items():
            metric_agg = agg[agg["Metric"] == metric_key]
            if metric_agg.empty:
                continue

            xs, means, sems = [], [], []
            for i, ds in enumerate(datasets):
                row = metric_agg[metric_agg["DatasetLabel"] == ds]
                if row.empty:
                    continue
                xs.append(float(x_base[i]))
                means.append(float(row.iloc[0]["Mean"]))
                sems.append(float(row.iloc[0]["SEM"]))

            if xs:
                marker = "o"
                fillstyle = "full" if metric_key == "valid_dag_pct" else "none"
                ax.errorbar(
                    xs,
                    means,
                    yerr=sems,
                    fmt=marker,
                    label=label_sfx,
                    color=avici_color,
                    capsize=3,
                    markersize=msize,
                    alpha=alpha,
                    fillstyle=fillstyle,
                )

        ax.set_xticks(x_base)
        ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("Valid DAG (%) $\\uparrow$", fontsize=10)
        ax.set_ylim(-5, 110)
        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        # Grey ID region
        id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")
        if 0 < id_count < len(datasets):
            ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
            ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

    # Shared legend
    handles, labels = [], []
    seen: set[str] = set()
    for ax in axes_arr.flatten():
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in seen:
                handles.append(handle)
                labels.append(label)
                seen.add(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(labels),
            fontsize=10,
            frameon=False,
        )

    fig.suptitle(
        "AviCi DAG Validity Under Distribution Shift",
        fontsize=13,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    combined = pd.concat(all_agg, ignore_index=True) if all_agg else pd.DataFrame()
    log.info("Saved RQ1 AviCi DAG validity figure to %s", output_path)
    save_figure_data(output_path, combined)
    return combined
