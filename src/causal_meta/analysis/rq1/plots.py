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
        n_panels, 1, figsize=(7.5, 3.8 * n_panels), sharex=True, squeeze=False
    )

    all_agg: list[pd.DataFrame] = []

    for panel_idx, (mech_key, mech_label) in enumerate(mech_families):
        ax = axes[panel_idx, 0]
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
        if panel_idx == n_panels - 1:
            ax.set_xticklabels(datasets, rotation=25, ha="right", fontsize=9)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.set_title(mech_label, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
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
            fontsize=10,
            frameon=False,
        )

    fig.suptitle(axis_title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
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
        n_panels, 1, figsize=(7.5, 3.8 * n_panels), sharex=True, squeeze=False
    )

    all_agg: list[pd.DataFrame] = []

    for panel_idx, anchor in enumerate(anchors):
        ax = axes[panel_idx, 0]

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
        if panel_idx == n_panels - 1:
            ax.set_xticklabels(datasets, rotation=25, ha="right", fontsize=9)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        anchor_label = GRAPH_ANCHOR_LABELS.get(anchor, anchor.upper())
        ax.set_title(f"Graph: {anchor_label}", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
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
            fontsize=10,
            frameon=False,
        )

    fig.suptitle(axis_title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
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
        n_panels, 1, figsize=(7.5, 3.8 * n_panels), sharex=True, squeeze=False
    )

    all_agg: list[pd.DataFrame] = []

    for panel_idx, anchor in enumerate(present_anchors):
        ax = axes[panel_idx, 0]
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
        if panel_idx == n_panels - 1:
            ax.set_xticklabels(datasets, rotation=25, ha="right", fontsize=9)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        mech_label = ID_MECHANISM_LABELS.get(anchor[0], anchor[0])
        graph_label = GRAPH_ANCHOR_LABELS.get(anchor[1], anchor[1].upper())
        ax.set_title(f"{mech_label} / {graph_label}", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
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
            fontsize=10,
            frameon=False,
        )

    fig.suptitle(axis_title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
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
) -> pd.DataFrame:
    """Generate a multi-panel compound-shift figure, one column per OOD graph.

    Each panel fixes the OOD graph topology and compares different OOD
    mechanisms against the same small set of representative ID anchors.
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

        id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")
        if 0 < id_count < len(datasets):
            ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
            ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

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
            subset, metric_name, axis_title, output_path
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

    from causal_meta.analysis.rq3.diagnostics import _discover_artifacts
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
    fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 5), squeeze=False)

    all_agg_rows: list[pd.DataFrame] = []

    for panel_idx, shift_key in enumerate(shift_keys):
        ax = axes[0, panel_idx]
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
        y=1.07,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    combined = (
        pd.concat(all_agg_rows, ignore_index=True) if all_agg_rows else pd.DataFrame()
    )
    return combined


# ── Error decomposition (FP / FN / Reversed) shift figure ─────────────

_ERROR_COMPONENT_COLORS: dict[str, str] = {
    "False Positive": "#d62728",
    "False Negative": "#1f77b4",
    "Reversed": "#ff7f0e",
}

_ERROR_DECOMP_SHIFT_SPECS: dict[str, tuple[str, str]] = {
    "graph": ("graph", "Graph Shift"),
    "mechanism": ("mechanism", "Mechanism Shift"),
    "compound": ("compound", "Compound Shift"),
}


def generate_error_decomposition_figure(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
) -> pd.DataFrame:
    """Generate a multi-panel stacked-bar figure decomposing SHD into FP/FN/reversed.

    Layout: one row per shift axis, one column per model.  Each bar shows the
    mean error decomposition for a dataset family, stacked as false-positive
    (red), false-negative (blue), and reversed (orange).

    Args:
        raw_df: Long-format raw task DataFrame (must contain ``fp_count``,
            ``fn_count``, ``reversed_count``).
        output_path: Path for the output PDF figure.

    Returns:
        Aggregated DataFrame used for plotting.
    """
    needed = {"fp_count", "fn_count", "reversed_count"}
    available = set(raw_df["Metric"].unique()) if not raw_df.empty else set()
    if not needed.issubset(available):
        log.warning(
            "Missing error decomposition metrics %s; skipping.",
            needed - available,
        )
        return pd.DataFrame()

    error_df = raw_df[raw_df["Metric"].isin(needed)].copy()
    error_df = error_df[is_fixed_size_task_frame(error_df)]

    if error_df.empty:
        log.warning("No fixed-size error decomposition data; skipping.")
        return pd.DataFrame()

    # Pivot to wide: one row per (Model, DatasetKey, TaskIdx)
    from causal_meta.analysis.rq1.failure_modes import _pivot_raw_wide

    wide = _pivot_raw_wide(error_df)
    if wide.empty:
        return pd.DataFrame()

    for col in needed:
        if col not in wide.columns:
            log.warning("Column %s not found after pivot; skipping error decomp.", col)
            return pd.DataFrame()

    wide["AxisCategory"] = wide["DatasetKey"].map(axis_category)

    # Compute dataset label
    from causal_meta.analysis.utils import map_dataset_description

    wide["Dataset_desc"] = wide["DatasetKey"].map(map_dataset_description)
    wide["DatasetLabel"] = wide.apply(
        lambda row: thesis_dataset_label(
            str(row["DatasetKey"]), str(row["Dataset_desc"])
        ),
        axis=1,
    )

    shift_keys = list(_ERROR_DECOMP_SHIFT_SPECS.keys())
    models = [m for m in PAPER_MODEL_LABELS.values() if m in wide["Model"].unique()]
    n_shifts = len(shift_keys)
    n_models = len(models)

    if n_models == 0:
        return pd.DataFrame()

    fig, axes = plt.subplots(
        n_shifts,
        n_models,
        figsize=(max(6, 3) * n_models, 4.5 * n_shifts),
        squeeze=False,
        sharey="row",
    )

    all_agg: list[pd.DataFrame] = []

    for row_idx, shift_key in enumerate(shift_keys):
        axis_cat, axis_title = _ERROR_DECOMP_SHIFT_SPECS[shift_key]

        panel = wide[wide["AxisCategory"].isin(["id", axis_cat])].copy()

        # ── Isolate the shift axis ────────────────────────────────────
        if shift_key == "graph":
            panel = panel[
                panel["DatasetKey"].map(lambda dk: id_mechanism_of(dk) == "linear")
            ].copy()
            # Use short graph-topology labels
            panel["DatasetLabel"] = panel["DatasetKey"].map(_graph_shift_label)
        elif shift_key == "mechanism":
            # Use ER-20 as representative graph anchor.
            # Filter both ID and OOD-mech to ER-20 to avoid mixing anchors.
            panel = _restrict_to_graph_anchor(panel, "er20")
        elif shift_key == "compound":
            # Restrict ID to the same 3 representatives as the main figure
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

        if panel.empty:
            for col_idx in range(n_models):
                axes[row_idx, col_idx].set_visible(False)
            continue

        # Aggregate: mean FP / FN / reversed per (Model, DatasetLabel, AxisCategory)
        agg = (
            panel.groupby(["Model", "DatasetKey", "DatasetLabel", "AxisCategory"])[
                list(needed)
            ]
            .mean()
            .reset_index()
        )
        agg["_sort"] = agg["AxisCategory"].map({"id": 0}).fillna(1)
        agg = agg.sort_values(["_sort", "DatasetLabel"]).drop(columns=["_sort"])
        all_agg.append(agg)

        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            model_agg = agg[agg["Model"] == model]
            datasets = list(model_agg["DatasetLabel"].unique())
            n_ds = len(datasets)

            if n_ds == 0:
                ax.set_visible(False)
                continue

            x = np.arange(n_ds)
            bar_width = 0.65

            bottoms = np.zeros(n_ds)
            for comp_metric, comp_label in [
                ("fn_count", "False Negative"),
                ("fp_count", "False Positive"),
                ("reversed_count", "Reversed"),
            ]:
                heights = np.array(
                    [
                        float(
                            model_agg[model_agg["DatasetLabel"] == ds][
                                comp_metric
                            ].values[0]
                        )
                        if ds in model_agg["DatasetLabel"].values
                        else 0.0
                        for ds in datasets
                    ]
                )
                ax.bar(
                    x,
                    heights,
                    bar_width,
                    bottom=bottoms,
                    color=_ERROR_COMPONENT_COLORS[comp_label],
                    label=comp_label if (row_idx == 0 and col_idx == 0) else None,
                    edgecolor="white",
                    linewidth=0.5,
                )
                bottoms += heights

            ax.set_xticks(x)
            ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel("Mean Edge Errors", fontsize=10)
            if row_idx == 0:
                ax.set_title(model, fontsize=12, fontweight="bold")
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)

            # Grey ID region
            axis_lookup = dict(
                zip(model_agg["DatasetLabel"], model_agg["AxisCategory"])
            )
            id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")
            if 0 < id_count < n_ds:
                ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
                ax.axvline(
                    id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0
                )

        # Row label on the far-left axis
        if shift_key == "graph":
            row_label = f"{axis_title} (Linear)"
        elif shift_key == "mechanism":
            row_label = f"{axis_title} (ER-20)"
        else:
            row_label = axis_title
        axes[row_idx, 0].annotate(
            row_label,
            xy=(-0.35, 0.5),
            xycoords="axes fraction",
            fontsize=11,
            fontweight="bold",
            rotation=90,
            ha="center",
            va="center",
        )

    # Shared legend on top
    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in labels:
                    handles.append(hh)
                    labels.append(ll)
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
        "SHD Error Decomposition Under Distribution Shift",
        fontsize=14,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    combined = pd.concat(all_agg, ignore_index=True) if all_agg else pd.DataFrame()
    return combined
