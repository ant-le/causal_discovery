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
    GRAPH_DESCRIPTION_MAP,
    MECH_DESCRIPTION_MAP,
    MODEL_COLORS,
    MODEL_MARKERS,
    PAPER_MODEL_LABELS,
)

log = logging.getLogger(__name__)


def _model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#555555")


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


def _generate_graph_shift_panels(
    subset: pd.DataFrame,
    metric_name: str,
    axis_title: str,
    output_path: Path,
) -> pd.DataFrame:
    """Generate a multi-panel graph shift figure, one row per ID mechanism family.

    Each panel shows only the ID dataset with that mechanism plus the
    OOD-graph datasets evaluated with the same mechanism, isolating graph
    topology as the sole shift variable.
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
    models = list(PAPER_MODEL_LABELS.values())
    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.0 * n_panels, 4.5), sharey=True, squeeze=False
    )

    all_agg: list[pd.DataFrame] = []

    for panel_idx, (mech_key, mech_label) in enumerate(mech_families):
        ax = axes[0, panel_idx]
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

        ax.set_xticks(x_base)
        ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=8)
        ax.set_title(mech_label, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if panel_idx == 0:
            ylabel = (
                r"Normalized $\mathbb{E}$-SID $\downarrow$"
                if metric_name == "ne-sid"
                else r"$\mathbb{E}$-SID $\downarrow$"
            )
            ax.set_ylabel(ylabel, fontsize=11)

        # Grey ID region
        id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")
        if 0 < id_count < len(datasets):
            ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
            ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

    # Shared legend on top
    handles, labels = axes[0, 0].get_legend_handles_labels()
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

    return pd.concat(all_agg, ignore_index=True) if all_agg else pd.DataFrame()


# ── ID baseline restriction helpers ────────────────────────────────────


def _restrict_id_to_anchor(
    subset: pd.DataFrame, *, mech_key: str | None, graph_code: str | None
) -> pd.DataFrame:
    """Keep only the ID dataset matching the given mechanism and graph anchor.

    For mechanism shift panels, ``mech_key`` is ``None`` (mechanism is OOD),
    so we match only on graph.  For noise shift, both are set.
    """
    id_mask = subset["AxisCategory"] == "id"

    def _keep(dk: str) -> bool:
        if graph_code is not None and graph_code_of(dk) != graph_code:
            return False
        if mech_key is not None and id_mechanism_of(dk) != mech_key:
            return False
        return True

    keep_id = id_mask & subset["DatasetKey"].map(_keep)
    return subset[keep_id | ~id_mask].copy()


def _restrict_to_graph_anchor(subset: pd.DataFrame, graph_code: str) -> pd.DataFrame:
    """Keep only rows whose graph code matches *graph_code*.

    Unlike :func:`_restrict_id_to_anchor`, this filters **all** rows (ID and
    OOD alike), ensuring mechanism-shift families with different graph anchors
    are not mixed in summary panels that use a single representative anchor.
    """
    return subset[
        subset["DatasetKey"].map(lambda dk: graph_code_of(dk) == graph_code)
    ].copy()


def _restrict_id_to_er_linear(subset: pd.DataFrame) -> pd.DataFrame:
    """Keep only the ``id_linear_er*`` dataset as the ID baseline.

    Legacy helper kept for backward compatibility; prefers the more
    general :func:`_restrict_id_to_anchor` for new code.
    """
    return _restrict_id_to_anchor(subset, mech_key="linear", graph_code=None)


# ── Mechanism-shift multi-panel helpers (one panel per graph anchor) ───

_MECH_SHIFT_GRAPH_ANCHORS: list[str] = ["er20", "er60", "sf2"]
"""Graph anchors used for mechanism-shift panels."""


def _mech_shift_label(dataset_key: str) -> str:
    """Short mechanism label for mechanism-shift panels.

    OOD-mechanism datasets return the OOD mechanism name (e.g. ``"Periodic"``).
    ID datasets return the ID mechanism with ``(ID)`` suffix.
    """
    from causal_meta.analysis.utils import MECH_DESCRIPTION_MAP

    dk = dataset_key.lower()
    body = re.sub(r"_d\d+_n\d+$", "", dk)

    if dk.startswith("ood_mech_"):
        # Strip prefix and graph anchor to get mechanism code
        remainder = body[len("ood_mech_") :]
        # Remove trailing graph anchor
        for code in sorted(GRAPH_ANCHOR_LABELS.keys(), key=len, reverse=True):
            if remainder.endswith(f"_{code}"):
                remainder = remainder[: -len(code) - 1]
                break
        return MECH_DESCRIPTION_MAP.get(remainder, remainder.replace("_", " ").title())

    # ID datasets — show mechanism name
    for mech_key in ("neuralnet", "gpcde", "linear"):
        if f"_{mech_key}" in body:
            return f"{MECH_DESCRIPTION_MAP.get(mech_key, mech_key)} (ID)"

    return dataset_key


def _generate_mech_shift_panels(
    subset: pd.DataFrame,
    metric_name: str,
    axis_title: str,
    output_path: Path,
) -> pd.DataFrame:
    """Generate a multi-panel mechanism shift figure, one row per graph anchor.

    Each panel holds the graph anchor constant (e.g. ER-20) and shows
    the ID baseline plus OOD mechanisms evaluated on that graph.
    """
    subset = subset.copy()
    # Tag each OOD-mech row with its graph anchor
    subset["_graph_anchor"] = subset["DatasetKey"].map(mech_shift_graph_anchor)
    # Tag each ID row with its graph code (to match to anchors)
    subset.loc[subset["AxisCategory"] == "id", "_graph_anchor"] = subset.loc[
        subset["AxisCategory"] == "id", "DatasetKey"
    ].map(graph_code_of)

    anchors = [
        a for a in _MECH_SHIFT_GRAPH_ANCHORS if a in subset["_graph_anchor"].unique()
    ]
    if not anchors:
        return pd.DataFrame()

    n_panels = len(anchors)
    models = list(PAPER_MODEL_LABELS.values())
    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.0 * n_panels, 4.5), sharey=True, squeeze=False
    )

    all_agg: list[pd.DataFrame] = []

    for panel_idx, anchor in enumerate(anchors):
        ax = axes[0, panel_idx]

        # For each anchor panel, keep OOD-mech families with this anchor
        # plus ID families matching this graph code
        ood_mask = (subset["AxisCategory"] == "mechanism") & (
            subset["_graph_anchor"] == anchor
        )
        # For ID, find the matching graph code. For ER anchors, any ID
        # with the same ER sparsity works. We want one ID baseline per anchor.
        # Use the first ID mechanism that has data (usually linear).
        id_mask = (subset["AxisCategory"] == "id") & (subset["_graph_anchor"] == anchor)
        panel_data = subset[ood_mask | id_mask]

        if panel_data.empty:
            ax.set_visible(False)
            continue

        agg = (
            panel_data.groupby(
                ["Model", "DatasetKey", "Dataset", "AxisCategory"], dropna=False
            )["Value"]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )
        agg["DatasetLabel"] = agg["DatasetKey"].map(_mech_shift_label)
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

        ax.set_xticks(x_base)
        ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=8)
        anchor_label = GRAPH_ANCHOR_LABELS.get(anchor, anchor.upper())
        ax.set_title(f"Graph: {anchor_label}", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if panel_idx == 0:
            ylabel = (
                r"Normalized $\mathbb{E}$-SID $\downarrow$"
                if metric_name == "ne-sid"
                else r"$\mathbb{E}$-SID $\downarrow$"
            )
            ax.set_ylabel(ylabel, fontsize=11)

        # Grey ID region
        id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")
        if 0 < id_count < len(datasets):
            ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
            ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

    # Shared legend on top
    handles, labels = axes[0, 0].get_legend_handles_labels()
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

    return pd.concat(all_agg, ignore_index=True) if all_agg else pd.DataFrame()


# ── Noise-shift multi-panel helpers (one panel per anchor) ─────────────

_NOISE_SHIFT_ANCHORS: list[tuple[str, str]] = [
    ("linear", "er20"),
    ("neuralnet", "sf2"),
]
"""(mechanism, graph_code) anchors used for noise-shift panels."""


def _noise_shift_label(dataset_key: str) -> str:
    """Short noise label for noise-shift panels.

    Returns ``"Laplace"``, ``"Uniform"``, or ``"Gaussian (ID)"``.
    """
    dk = dataset_key.lower()
    body = re.sub(r"_d\d+_n\d+$", "", dk)

    if dk.startswith("ood_noise_"):
        remainder = body[len("ood_noise_") :]
        # First token is the noise type
        noise_type = remainder.split("_")[0]
        return noise_type.title()

    # ID baseline
    return "Gaussian (ID)"


def _generate_noise_shift_panels(
    subset: pd.DataFrame,
    metric_name: str,
    axis_title: str,
    output_path: Path,
) -> pd.DataFrame:
    """Generate a multi-panel noise shift figure, one row per anchor.

    Each panel holds mechanism and graph constant (e.g. Linear/ER-20) and
    shows the Gaussian ID baseline plus Laplace and Uniform noise.
    """
    subset = subset.copy()
    # Tag each noise row with its anchor
    subset["_noise_anchor"] = subset["DatasetKey"].map(noise_shift_anchor)
    # Tag each ID row with its (mech, graph_code)
    id_mask = subset["AxisCategory"] == "id"
    subset.loc[id_mask, "_noise_anchor"] = subset.loc[id_mask, "DatasetKey"].map(
        lambda dk: (
            (id_mechanism_of(dk), graph_code_of(dk))
            if id_mechanism_of(dk) is not None
            else None
        )
    )

    present_anchors = [
        a
        for a in _NOISE_SHIFT_ANCHORS
        if a in set(subset["_noise_anchor"].dropna().tolist())
    ]
    if not present_anchors:
        return pd.DataFrame()

    n_panels = len(present_anchors)
    models = list(PAPER_MODEL_LABELS.values())
    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.0 * n_panels, 4.5), sharey=True, squeeze=False
    )

    all_agg: list[pd.DataFrame] = []

    for panel_idx, anchor in enumerate(present_anchors):
        ax = axes[0, panel_idx]
        panel_data = subset[subset["_noise_anchor"] == anchor]

        if panel_data.empty:
            ax.set_visible(False)
            continue

        agg = (
            panel_data.groupby(
                ["Model", "DatasetKey", "Dataset", "AxisCategory"], dropna=False
            )["Value"]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )
        agg["DatasetLabel"] = agg["DatasetKey"].map(_noise_shift_label)
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

        ax.set_xticks(x_base)
        ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=8)
        mech_label = ID_MECHANISM_LABELS.get(anchor[0], anchor[0])
        graph_label = GRAPH_ANCHOR_LABELS.get(anchor[1], anchor[1].upper())
        ax.set_title(f"{mech_label} / {graph_label}", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if panel_idx == 0:
            ylabel = (
                r"Normalized $\mathbb{E}$-SID $\downarrow$"
                if metric_name == "ne-sid"
                else r"$\mathbb{E}$-SID $\downarrow$"
            )
            ax.set_ylabel(ylabel, fontsize=11)

        # Grey ID region
        id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")
        if 0 < id_count < len(datasets):
            ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
            ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

    # Shared legend on top
    handles, labels = axes[0, 0].get_legend_handles_labels()
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

    return pd.concat(all_agg, ignore_index=True) if all_agg else pd.DataFrame()


def _compound_shift_label(dataset_key: str) -> str:
    """Short label for compound-shift panels.

    OOD-compound datasets show only the OOD mechanism label because the OOD graph
    is fixed by the panel title. ID representatives show their graph/mechanism
    anchor with an ``(ID)`` suffix.
    """
    dk = dataset_key.lower()
    body = re.sub(r"_d\d+_n\d+$", "", dk)

    if dk.startswith("ood_both_"):
        remainder = body[len("ood_both_") :]
        graph_code = graph_code_of(dk)
        if graph_code is not None and remainder.startswith(f"{graph_code}_"):
            remainder = remainder[len(graph_code) + 1 :]
        return MECH_DESCRIPTION_MAP.get(remainder, remainder.replace("_", " ").title())

    mech_key = id_mechanism_of(dk)
    graph_code = graph_code_of(dk)
    if mech_key is None or graph_code is None:
        return dataset_key
    mech_label = ID_MECHANISM_LABELS.get(mech_key, mech_key)
    graph_label = GRAPH_ANCHOR_LABELS.get(graph_code, graph_code.upper())
    return f"{graph_label} / {mech_label} (ID)"


def _compound_shift_sort_key(dataset_key: str) -> tuple[int, int, str]:
    """Sort ID representatives first, then OOD mechanisms in a stable order."""
    dk = dataset_key.lower()
    mech_key = id_mechanism_of(dk)
    graph_code = graph_code_of(dk)
    anchor = (mech_key, graph_code)
    if dk.startswith("id_") and anchor in _COMPOUND_ID_REPRESENTATIVES:
        return (0, _COMPOUND_ID_REPRESENTATIVES.index(anchor), dk)

    if dk.startswith("ood_both_"):
        body = re.sub(r"_d\d+_n\d+$", "", dk)
        remainder = body[len("ood_both_") :]
        ood_graph = graph_code_of(dk)
        if ood_graph is not None and remainder.startswith(f"{ood_graph}_"):
            remainder = remainder[len(ood_graph) + 1 :]
        return (1, _COMPOUND_OOD_MECH_ORDER.get(remainder, 999), dk)

    return (2, 999, dk)


def _generate_compound_shift_panels(
    subset: pd.DataFrame,
    metric_name: str,
    axis_title: str,
    output_path: Path,
    *,
    stress_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate a multi-panel compound-shift figure, one column per OOD graph.

    Each panel fixes the OOD graph topology and compares different OOD
    mechanisms against the same small set of representative ID anchors.

    When *stress_df* is provided the extreme stress-test families are appended
    after a second vertical divider, showing the ID → compound → stress
    progression in a single figure.
    """
    subset = subset.copy()
    id_mask = subset["AxisCategory"] == "id"
    keep_id = id_mask & subset["DatasetKey"].map(
        lambda dk: (
            (id_mechanism_of(dk), graph_code_of(dk)) in _COMPOUND_ID_REPRESENTATIVES
        )
    )
    subset = subset[keep_id | ~id_mask].copy()

    present_graphs = [
        graph_code
        for graph_code in _COMPOUND_OOD_GRAPHS
        if (
            (subset["AxisCategory"] == "compound")
            & subset["DatasetKey"].map(lambda dk: graph_code_of(dk) == graph_code)
        ).any()
    ]
    if not present_graphs:
        return pd.DataFrame()

    # Build a lookup from graph_code → stress-test family key
    stress_by_graph: dict[str, pd.DataFrame] = {}
    if stress_df is not None and not stress_df.empty:
        for fam_key in _EXTREME_FAMILIES:
            gc = graph_code_of(fam_key)
            if gc is not None:
                fam_rows = stress_df[stress_df["DatasetKey"] == fam_key]
                if not fam_rows.empty:
                    stress_by_graph[gc] = fam_rows

    n_panels = len(present_graphs)
    models = list(PAPER_MODEL_LABELS.values())
    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6 * n_panels, 5), sharey=True, squeeze=False
    )

    all_agg: list[pd.DataFrame] = []

    for panel_idx, graph_code in enumerate(present_graphs):
        ax = axes[0, panel_idx]
        ood_mask = (subset["AxisCategory"] == "compound") & subset["DatasetKey"].map(
            lambda dk: graph_code_of(dk) == graph_code
        )
        panel_data = subset[ood_mask | (subset["AxisCategory"] == "id")].copy()
        if panel_data.empty:
            ax.set_visible(False)
            continue

        agg = (
            panel_data.groupby(
                ["Model", "DatasetKey", "Dataset", "AxisCategory"], dropna=False
            )["Value"]
            .agg(Mean="mean", SEM=metric_sem)
            .reset_index()
        )
        agg["DatasetLabel"] = agg["DatasetKey"].map(_compound_shift_label)
        agg["FixedConcept"] = GRAPH_ANCHOR_LABELS.get(graph_code, graph_code.upper())
        agg["SortKey"] = agg["DatasetKey"].map(_compound_shift_sort_key)
        agg = agg.sort_values("SortKey").drop(columns=["SortKey"])

        # ── Append stress-test rows if present for this graph ─────────
        stress_agg: pd.DataFrame | None = None
        if graph_code in stress_by_graph:
            stress_panel = stress_by_graph[graph_code]
            stress_agg = (
                stress_panel.groupby(["Model", "DatasetKey"], dropna=False)["Value"]
                .agg(Mean="mean", SEM=metric_sem)
                .reset_index()
            )
            stress_agg["AxisCategory"] = "stress"
            stress_agg["Dataset"] = stress_agg["DatasetKey"]
            stress_agg["DatasetLabel"] = stress_agg["DatasetKey"].map(
                lambda dk: _EXTREME_LABELS.get(dk, dk)
            )
            stress_agg["FixedConcept"] = GRAPH_ANCHOR_LABELS.get(
                graph_code, graph_code.upper()
            )
            agg = pd.concat([agg, stress_agg], ignore_index=True)

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

        ax.set_xticks(x_base)
        ax.set_xticklabels(datasets, rotation=25, ha="right", fontsize=9)
        ax.set_title(
            f"Graph: {GRAPH_ANCHOR_LABELS.get(graph_code, graph_code.upper())}",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        if panel_idx == 0:
            ylabel = (
                r"Normalized $\mathbb{E}$-SID $\downarrow$"
                if metric_name == "ne-sid"
                else r"$\mathbb{E}$-SID $\downarrow$"
            )
            ax.set_ylabel(ylabel, fontsize=12)

        # Grey ID region
        id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")
        if 0 < id_count < len(datasets):
            ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
            ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

        # Stress-test region (light red tint + second divider)
        if stress_agg is not None and not stress_agg.empty:
            stress_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "stress")
            if stress_count > 0:
                stress_start = n_datasets - stress_count
                ax.axvspan(
                    stress_start - 0.5,
                    n_datasets - 0.5,
                    color="#ffe0e0",
                    alpha=0.35,
                    zorder=0,
                )
                ax.axvline(
                    stress_start - 0.5,
                    color="#cc4444",
                    linestyle=":",
                    linewidth=1.0,
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
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

    return pd.concat(all_agg, ignore_index=True) if all_agg else pd.DataFrame()


def generate_results_anchor_table(
    raw_df: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    subset = raw_df[
        raw_df["AxisCategory"].eq("id")
        & raw_df["Metric"].isin(
            [
                "ne-sid",
                "ne-shd",
                "e-edgef1",
                "valid_dag_pct",
                "inference_time_s",
            ]
        )
    ].copy()

    grouped = (
        subset.groupby(["Model", "Metric"], dropna=False)["Value"]
        .agg(Mean="mean", SEM=metric_sem)
        .reset_index()
    )

    metric_specs = [
        ("ne-sid", False),
        ("ne-shd", False),
        ("e-edgef1", True),
        ("valid_dag_pct", True),
        ("inference_time_s", False),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Speed--robustness anchor on the in-distribution families. Values report task-level means and standard errors aggregated over all in-distribution tasks.}",
        r"\label{tab:results_anchor}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Normalized $\mathbb{E}$-SID $\downarrow$} & \textbf{Normalized $\mathbb{E}$-SHD $\downarrow$} & \textbf{$\mathbb{E}$-Edge F1 $\uparrow$} & \textbf{Valid DAG Samples (\%) $\uparrow$} & \textbf{Runtime / dataset $\downarrow$} \\",
        r"\midrule",
    ]

    models = list(PAPER_MODEL_LABELS.values())
    for model in models:
        model_rows = grouped[grouped["Model"] == model]
        cells: list[str] = []
        for metric_name, higher_is_better in metric_specs:
            metric_rows = grouped[grouped["Metric"] == metric_name]
            if metric_rows.empty:
                best_model = None
            elif higher_is_better:
                best_model = str(
                    metric_rows.sort_values("Mean", ascending=False).iloc[0]["Model"]
                )
            else:
                best_model = str(
                    metric_rows.sort_values("Mean", ascending=True).iloc[0]["Model"]
                )
            row = model_rows[model_rows["Metric"] == metric_name]
            if row.empty:
                cells.append("-")
                continue
            mean = float(row.iloc[0]["Mean"])
            sem = float(row.iloc[0]["SEM"])
            cells.append(
                _bold_if_best(format_value(mean, sem), is_best=(best_model == model))
            )
        lines.append(f"{model} & " + " & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    output_path.write_text("\n".join(lines) + "\n")
    return grouped


# ── RQ2: Degradation heatmap ──────────────────────────────────────────

_DEGRADATION_SHIFT_AXES: dict[str, str] = {
    "graph": "Graph",
    "mechanism": "Mechanism",
    "noise": "Noise",
    "compound": "Compound",
    "nodes": "Nodes",
    "samples": "Samples",
}
"""Shift axes included in the RQ2 degradation heatmap."""


def _degradation_id_subset(
    full_df: pd.DataFrame,
    shift_key: str,
) -> pd.DataFrame:
    """Return the ID rows that form the correct baseline for *shift_key*.

    Each shift axis uses a different anchored ID subset so that the
    degradation ratio isolates only the varied dimension:

    * **graph** — ID families with Linear mechanism (graph is the only
      variable in the graph-shift figure).
    * **mechanism** — ID families on ER-20 (mechanism is the only variable).
    * **noise** — all fixed-size ID families (both anchors participate).
    * **compound** — the three representative ID anchor families.
    * **nodes** — fixed-size ID families on the two transfer anchors.
    * **samples** — same as *nodes*.
    """
    id_data = full_df[full_df["AxisCategory"] == "id"].copy()

    if shift_key == "graph":
        return id_data[
            id_data["DatasetKey"].map(lambda dk: id_mechanism_of(dk) == "linear")
        ]
    if shift_key == "mechanism":
        return _restrict_to_graph_anchor(id_data, "er20")
    if shift_key == "compound":
        return id_data[
            id_data["DatasetKey"].map(
                lambda dk: (
                    (id_mechanism_of(dk), graph_code_of(dk))
                    in _COMPOUND_ID_REPRESENTATIVES
                )
            )
        ]
    if shift_key in ("nodes", "samples"):
        # Transfer anchors: (linear, er20), (neuralnet, sf2)
        _TRANSFER_ANCHORS = {("linear", "er20"), ("neuralnet", "sf2")}
        return id_data[
            id_data["DatasetKey"].map(
                lambda dk: (id_mechanism_of(dk), graph_code_of(dk)) in _TRANSFER_ANCHORS
            )
        ]
    # noise and any other: all fixed-size ID
    return id_data


def generate_degradation_heatmap(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
) -> pd.DataFrame:
    """Generate a heatmap showing per-model degradation ratio across shift axes.

    Each cell shows ``mean(OOD ne-SID) / mean(ID ne-SID)`` for a given model
    and shift axis.  The ID baseline is *anchored per axis* so that each ratio
    isolates only the varied distributional dimension.

    A ratio of 1.0 means no degradation; higher values indicate worse OOD
    performance relative to the relevant ID baseline.

    The colour scale encodes normalized degradation so models and axes can
    be compared at a glance.

    Args:
        raw_df: Long-format raw task DataFrame.
        output_path: Path for the output PDF figure.

    Returns:
        Pivot table (models x shift axes) of degradation ratios.
    """
    import matplotlib.colors as mcolors

    metric_name = "ne-sid"
    subset = raw_df[raw_df["Metric"].eq(metric_name)].copy()

    # For fixed-size axes, restrict to d=20 / n=500.
    # For transfer axes (nodes, samples), keep all sizes.
    fixed_subset = subset[is_fixed_size_task_frame(subset)]

    if fixed_subset.empty:
        log.warning("No ne-sid data for degradation heatmap; skipping.")
        return pd.DataFrame()

    models = list(PAPER_MODEL_LABELS.values())

    rows: list[dict[str, object]] = []
    for shift_key, shift_label in _DEGRADATION_SHIFT_AXES.items():
        # Choose the right data pool for this axis
        if shift_key in ("nodes", "samples"):
            pool = subset  # transfer data is NOT fixed-size
        else:
            pool = fixed_subset

        # Per-axis anchored ID baseline
        id_data = _degradation_id_subset(fixed_subset, shift_key)
        id_means = id_data.groupby("Model")["Value"].mean().to_dict()

        ood_data = pool[pool["AxisCategory"] == shift_key]
        ood_means = ood_data.groupby("Model")["Value"].mean().to_dict()
        for model in models:
            id_val = id_means.get(model)
            ood_val = ood_means.get(model)
            if id_val is not None and ood_val is not None and id_val > 0:
                ratio = ood_val / id_val
            else:
                ratio = float("nan")
            rows.append({"Model": model, "Shift": shift_label, "Ratio": ratio})

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        log.warning("No data for degradation heatmap; skipping.")
        return pd.DataFrame()

    pivot = result_df.pivot(index="Model", columns="Shift", values="Ratio")
    # Reorder rows and columns
    pivot = pivot.reindex(index=[m for m in models if m in pivot.index])
    pivot = pivot.reindex(
        columns=[v for v in _DEGRADATION_SHIFT_AXES.values() if v in pivot.columns]
    )

    fig, ax = plt.subplots(figsize=(8, 3.5))

    # Diverging colormap around 1.0 (no degradation)
    vmin = max(0.5, float(pivot.min().min()) - 0.1) if not pivot.empty else 0.5
    vmax = max(float(pivot.max().max()) + 0.1, 1.5) if not pivot.empty else 2.0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    cmap = plt.cm.RdYlGn_r  # red = worse (high ratio), green = good (low ratio)

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)

    # Annotate cells with ratio values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isnan(val):
                text = "--"
            else:
                text = f"{val:.2f}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white" if val > (vmin + vmax) / 2 else "black",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    cbar.set_label("OOD / ID ratio", fontsize=10)

    ax.set_title(
        "Normalized E-SID Degradation Ratio (OOD / ID)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return pivot


def generate_shift_figure(
    raw_df: pd.DataFrame, *, shift_axis: str, output_path: Path
) -> pd.DataFrame:
    axis_cat, axis_title = _SHIFT_AXIS_SPECS[shift_axis]
    metric_name = "ne-sid" if shift_axis == "compound" else "e-sid"
    subset = raw_df[
        raw_df["Metric"].eq(metric_name) & raw_df["AxisCategory"].isin(["id", axis_cat])
    ].copy()
    subset = subset[is_fixed_size_task_frame(subset)]

    # ── Graph shift: multi-panel by mechanism family ──────────────────
    if shift_axis == "graph":
        return _generate_graph_shift_panels(
            subset, metric_name, axis_title, output_path
        )

    # ── Mechanism shift: multi-panel by graph anchor ──────────────────
    if shift_axis == "mechanism":
        return _generate_mech_shift_panels(subset, metric_name, axis_title, output_path)

    # ── Noise shift: multi-panel by (mechanism, graph) anchor ─────────
    if shift_axis == "noise":
        return _generate_noise_shift_panels(
            subset, metric_name, axis_title, output_path
        )

    # ── Compound shift: multi-panel by fixed OOD graph ────────────────
    if shift_axis == "compound":
        return _generate_compound_shift_panels(
            subset,
            metric_name,
            axis_title,
            output_path,
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
    models = list(PAPER_MODEL_LABELS.values())
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
    return agg


def generate_compound_and_stress_figure(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
) -> pd.DataFrame:
    """Generate a merged compound-shift + stress-test figure.

    Each panel shows a single OOD graph topology with three zones:
    ID baselines (grey) → fixed-size compound OOD → extreme stress-test
    families (light red).

    Args:
        raw_df: Long-format raw task DataFrame.
        output_path: Path for the output PDF figure.

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

    return _generate_compound_shift_panels(
        compound_subset,
        metric_name,
        "Compound Shift",
        output_path,
        stress_df=stress_subset,
    )


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
    models = [m for m in PAPER_MODEL_LABELS.values() if m in wide["Model"].unique()]

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
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(
        r"\caption{Mean SHD error decomposition (false positive, false negative,"
        r" and reversed edges) per model under each OOD shift axis. Values are"
        r" averaged over all OOD families in the respective shift axis"
        r" (graph, mechanism, noise, and compound).}"
    )
    lines.append(r"\label{tab:error_decomposition}")

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
                        row_str += rf" & ${mean_val:.1f} \pm {sem_val:.1f}$"
        row_str += r" \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    output_path.write_text("\n".join(lines) + "\n")
    log.info("Saved error decomposition table to %s", output_path)
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

    Returns:
        Aggregated DataFrame used for the table.
    """
    metric_keys = [m for m, _, _ in _SUMMARY_TABLE_METRICS]
    subset = raw_df[raw_df["Metric"].isin(metric_keys)].copy()
    if subset.empty:
        log.warning("No data for cross-axis summary table; skipping.")
        return pd.DataFrame()

    fixed = subset[is_fixed_size_task_frame(subset)]
    models = [m for m in PAPER_MODEL_LABELS.values() if m in subset["Model"].unique()]
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

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(
        r"\caption{Cross-axis performance summary. Each cell reports the"
        r" task-level mean $\pm$ SEM for the indicated metric and shift axis."
        r" Per-column best values (excluding Random) are \textbf{bolded}.}"
    )
    lines.append(r"\label{tab:cross_axis_summary}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    # Column spec: Model + n_sub per axis
    col_spec = "l" + ("|" + "r" * n_sub) * n_axes
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: axis group headers
    header1 = r"\textbf{Model}"
    for _, axis_label in axes_list:
        header1 += rf" & \multicolumn{{{n_sub}}}{{c}}{{\textbf{{{axis_label}}}}}"
    header1 += r" \\"
    lines.append(header1)

    # Cmidrules
    cmidrules = ""
    col_start = 2
    for _ in axes_list:
        col_end = col_start + n_sub - 1
        cmidrules += rf"\cmidrule(lr){{{col_start}-{col_end}}} "
        col_start = col_end + 1
    lines.append(cmidrules)

    # Header row 2: metric sub-column labels
    header2 = ""
    for _ in axes_list:
        for _, mlabel, _ in _SUMMARY_TABLE_METRICS:
            header2 += rf" & \textbf{{{mlabel}}}"
    header2 += r" \\"
    lines.append(header2)
    lines.append(r"\midrule")

    # Data rows
    for model in models:
        row_str = model
        for _, axis_label in axes_list:
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
    lines.append(r"\end{table}")

    output_path.write_text("\n".join(lines) + "\n")
    log.info("Saved cross-axis summary table to %s", output_path)
    return combined
