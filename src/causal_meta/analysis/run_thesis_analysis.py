from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_meta.analysis.failure_modes import (
    classify_failure_modes,
    failure_mode_fractions,
)
from causal_meta.analysis.ood_detection import (
    compute_ood_detection_metrics,
    compute_selective_prediction,
)
from causal_meta.analysis.plots.results import (
    generate_event_probability_bar,
    generate_failure_mode_bar,
    generate_per_model_failure_mode_bar,
    generate_posterior_diagnostic_violins,
    generate_selective_prediction_pareto,
)
from causal_meta.analysis.posterior_diagnostics import (
    run_posterior_diagnostics_from_runs,
)
from causal_meta.analysis.tables.results import (
    generate_distance_regression_table,
    generate_robustness_table,
)
from causal_meta.analysis.utils import (
    AMORTISED_MODELS,
    EmptyAnalysisDataError,
    MODEL_COLORS,
    MODEL_MARKERS,
    PAPER_MODEL_LABELS,
    load_raw_task_dataframe,
    load_runs_dataframe,
    map_dataset_description,
)

log = logging.getLogger(__name__)

EXPECTED_MODEL_DIRS: tuple[str, ...] = tuple(PAPER_MODEL_LABELS.keys())


def _model_color(model: str) -> str:
    """Return the canonical colour for *model*, falling back to grey."""
    return MODEL_COLORS.get(model, "#555555")


OUTPUT_SUBDIRS: tuple[str, ...] = (
    "figures",
    "tables",
    "data",
    "snippets",
    "provenance",
)


@dataclass(frozen=True)
class SelectedRun:
    """Curated thesis run selected for one benchmarked model."""

    model_dir: str
    run_dir: Path
    run_id: str
    run_name: str
    model_name: str


class ThesisRunSelectionError(FileNotFoundError):
    """Raised when the curated thesis run layout is invalid."""


def _paper_model_label(model_key: str) -> str:
    model_key_norm = model_key.lower()
    for key, label in PAPER_MODEL_LABELS.items():
        if model_key_norm == key or model_key_norm.startswith(f"{key}_"):
            return label
    return model_key


def _axis_category(dataset_key: str) -> str:
    dataset_key_norm = dataset_key.lower()
    if dataset_key_norm.startswith("id_") or dataset_key_norm == "id_test":
        return "id"
    if dataset_key_norm.startswith("ood_graph_"):
        return "graph"
    if dataset_key_norm.startswith("ood_mech_"):
        return "mechanism"
    if dataset_key_norm.startswith("ood_noise_"):
        return "noise"
    if dataset_key_norm.startswith("ood_both_"):
        return "compound"
    if dataset_key_norm.startswith("ood_nodes_"):
        return "nodes"
    if dataset_key_norm.startswith("ood_samples_"):
        return "samples"
    return "other"


def _is_fixed_size_task_frame(df: pd.DataFrame) -> pd.Series:
    return df["NNodes"].eq(20) & df["SamplesPerTask"].eq(500)


def _thesis_dataset_label(dataset_key: str, dataset_label: str) -> str:
    label = dataset_label
    for prefix in ("ID ", "OOD-G ", "OOD-M ", "OOD-N ", "OOD-Both "):
        if label.startswith(prefix):
            label = label[len(prefix) :]
            break
    label = re.sub(r"\s*\(d=\d+, n=\d+\)$", "", label)
    if dataset_key.startswith("ood_nodes_"):
        match = re.search(r"_d(\d+)_n\d+$", dataset_key)
        if match is not None:
            return match.group(1)
    if dataset_key.startswith("ood_samples_"):
        match = re.search(r"_d\d+_n(\d+)$", dataset_key)
        if match is not None:
            return match.group(1)
    return label


def _metric_sem(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    sem = float(values.sem(ddof=1))
    return 0.0 if not np.isfinite(sem) else sem


def _format_value(mean: float, sem: float) -> str:
    return rf"${mean:.3f} \pm {sem:.3f}$"


def _bold_if_best(value: str, *, is_best: bool) -> str:
    if not is_best:
        return value
    return r"\textbf{" + value + "}"


def _read_metrics_payload(run_dir: Path) -> Mapping[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise ThesisRunSelectionError(
            f"Missing metrics.json in curated run directory: {run_dir}"
        )
    with open(metrics_path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise ThesisRunSelectionError(f"Malformed metrics.json in {run_dir}")
    return payload


def resolve_thesis_run_directories(input_root: Path) -> list[SelectedRun]:
    """Resolve the curated one-run-per-model thesis layout."""

    selected: list[SelectedRun] = []
    for model_dir in EXPECTED_MODEL_DIRS:
        run_dir = input_root / model_dir
        if not run_dir.exists():
            raise ThesisRunSelectionError(
                f"Missing curated model directory '{run_dir}'. Expected one folder per "
                "model under experiments/thesis_runs/."
            )
        payload = _read_metrics_payload(run_dir)
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        run_id = str(metadata.get("run_id", run_dir.name))
        run_name = str(metadata.get("run_name", run_id))
        model_name = str(metadata.get("model_name", model_dir)).strip() or model_dir
        if model_dir not in model_name.lower():
            raise ThesisRunSelectionError(
                f"Curated run '{run_dir}' reports model_name='{model_name}', which "
                f"does not match the expected model folder '{model_dir}'."
            )
        selected.append(
            SelectedRun(
                model_dir=model_dir,
                run_dir=run_dir.resolve(),
                run_id=run_id,
                run_name=run_name,
                model_name=model_name,
            )
        )
    return selected


def _prepare_generated_workspace(thesis_root: Path) -> Path:
    thesis_root.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="generated_tmp_", dir=thesis_root))
    for subdir in OUTPUT_SUBDIRS:
        (temp_root / subdir).mkdir(parents=True, exist_ok=True)
    return temp_root


def _finalize_generated_workspace(temp_root: Path, thesis_root: Path) -> Path:
    final_root = thesis_root / "generated"
    backup_root = thesis_root / "generated_backup"
    if backup_root.exists():
        shutil.rmtree(backup_root)
    if final_root.exists():
        final_root.replace(backup_root)
    temp_root.replace(final_root)
    if backup_root.exists():
        shutil.rmtree(backup_root)
    return final_root


def _write_json(output_path: Path, payload: Mapping[str, Any]) -> None:
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _escape_tex(value: str) -> str:
    for char, repl in (
        ("\\", r"\textbackslash{}"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        value = value.replace(char, repl)
    return value


def _write_results_macros(
    selected_runs: Sequence[SelectedRun], output_path: Path
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "% Auto-generated by run_thesis_analysis.py. Do not edit by hand.",
        rf"\providecommand{{\ResultsGeneratedAt}}{{{timestamp}}}",
    ]
    for run in selected_runs:
        macro_name = "ResultsRunId" + run.model_dir.capitalize()
        lines.append(rf"\providecommand{{\{macro_name}}}{{{_escape_tex(run.run_id)}}}")
    output_path.write_text("\n".join(lines) + "\n")


def _prepare_summary_dataframe(run_dirs: Sequence[Path]) -> pd.DataFrame:
    summary_df = load_runs_dataframe(run_dirs, translate_names=False)
    if summary_df.empty:
        raise EmptyAnalysisDataError("No summary metrics found in curated thesis runs.")
    summary_df = summary_df.copy()
    summary_df["Model"] = summary_df["ModelKey"].map(_paper_model_label)
    summary_df["Dataset"] = summary_df["DatasetKey"].map(map_dataset_description)
    summary_df["Dataset"] = summary_df.apply(
        lambda row: _thesis_dataset_label(str(row["DatasetKey"]), str(row["Dataset"])),
        axis=1,
    )
    summary_df["AxisCategory"] = summary_df["DatasetKey"].map(_axis_category)
    return summary_df


def _prepare_raw_dataframe(run_dirs: Sequence[Path]) -> pd.DataFrame:
    metrics = [
        "e-sid",
        "e-shd",
        "e-edgef1",
        "ne-sid",
        "ne-shd",
        "graph_nll_per_edge",
        "edge_entropy",
        "ece",
        "inference_time_s",
        "sparsity_ratio",
        "skeleton_f1",
        "orientation_accuracy",
        "valid_dag_pct",
    ]
    raw_df = load_raw_task_dataframe(
        run_dirs,
        metrics=metrics,
        translate_names=False,
        require_per_task=True,
        skip_non_per_task=False,
    )
    if raw_df.empty:
        raise EmptyAnalysisDataError(
            "No per-task raw metrics found in curated thesis runs."
        )
    raw_df = raw_df.copy()
    raw_df["Model"] = raw_df["ModelKey"].map(_paper_model_label)
    raw_df["Dataset"] = raw_df["DatasetKey"].map(map_dataset_description)
    raw_df["Dataset"] = raw_df.apply(
        lambda row: _thesis_dataset_label(str(row["DatasetKey"]), str(row["Dataset"])),
        axis=1,
    )
    raw_df["AxisCategory"] = raw_df["DatasetKey"].map(_axis_category)
    return raw_df


def generate_results_anchor_table(
    raw_df: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    """Generate the in-distribution speed--robustness anchor table."""

    subset = raw_df[
        raw_df["AxisCategory"].eq("id")
        & raw_df["Metric"].isin(
            ["ne-sid", "ne-shd", "e-edgef1", "valid_dag_pct", "inference_time_s"]
        )
    ].copy()
    if subset.empty:
        raise EmptyAnalysisDataError(
            "No ID raw metrics available for the anchor table."
        )

    grouped = (
        subset.groupby(["Model", "Metric"], dropna=False)["Value"]
        .agg(Mean="mean", SEM=_metric_sem)
        .reset_index()
    )

    metric_specs = [
        ("ne-sid", r"Normalized \mathbb{E}-SID", False),
        ("ne-shd", r"Normalized \mathbb{E}-SHD", False),
        ("e-edgef1", r"\mathbb{E}-Edge F1", True),
        ("valid_dag_pct", r"Valid DAG (\%)", True),
        ("inference_time_s", r"Runtime / dataset", False),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Speed--robustness anchor on the in-distribution families. Values report task-level means and standard errors aggregated over all in-distribution tasks.}",
        r"\label{tab:results_anchor}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Normalized $\mathbb{E}$-SID $\downarrow$} & \textbf{Normalized $\mathbb{E}$-SHD $\downarrow$} & \textbf{$\mathbb{E}$-Edge F1 $\uparrow$} & \textbf{Valid DAG (\%) $\uparrow$} & \textbf{Runtime / dataset $\downarrow$} \\",
        r"\midrule",
    ]

    models = list(PAPER_MODEL_LABELS.values())
    for model in models:
        model_rows = grouped[grouped["Model"] == model]
        cells: list[str] = []
        for metric_name, _, higher_is_better in metric_specs:
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
            formatted = _format_value(mean, sem)
            cells.append(_bold_if_best(formatted, is_best=(best_model == model)))
        lines.append(f"{model} & " + " & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    output_path.write_text("\n".join(lines) + "\n")
    return grouped


def _bar_positions(
    n_groups: int, n_series: int, width: float = 0.8
) -> tuple[np.ndarray, float]:
    x_base = np.arange(n_groups)
    bar_width = width / max(n_series, 1)
    return x_base, bar_width


def generate_fixed_ood_figure(raw_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Generate grouped degradation bars for graph/mechanism/compound shift."""

    subset = raw_df[
        raw_df["Metric"].isin(["e-sid", "e-edgef1"])
        & raw_df["AxisCategory"].isin(["id", "graph", "mechanism", "compound"])
    ].copy()
    if subset.empty:
        raise EmptyAnalysisDataError("No fixed-size OOD raw metrics available.")

    id_baselines = (
        subset[subset["AxisCategory"] == "id"]
        .groupby(["Model", "Metric"], dropna=False)["Value"]
        .mean()
        .rename("IDBaseline")
        .reset_index()
    )
    ood_subset = subset[subset["AxisCategory"] != "id"].merge(
        id_baselines,
        on=["Model", "Metric"],
        how="left",
    )
    ood_subset["Degradation"] = np.where(
        ood_subset["Metric"].eq("e-edgef1"),
        ood_subset["IDBaseline"] - ood_subset["Value"],
        ood_subset["Value"] - ood_subset["IDBaseline"],
    )

    agg = (
        ood_subset.groupby(["Model", "AxisCategory", "Metric"], dropna=False)[
            "Degradation"
        ]
        .agg(Mean="mean", SEM=_metric_sem)
        .reset_index()
    )

    category_order = ["graph", "mechanism", "compound"]
    category_labels = ["Graph", "Mechanism", "Compound"]
    models = list(PAPER_MODEL_LABELS.values())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    metric_specs = [
        ("e-sid", r"$\Delta \mathbb{E}$-SID (OOD - ID) $\uparrow$ = worse"),
        ("e-edgef1", r"$\Delta \mathbb{E}$-Edge F1 (ID - OOD) $\uparrow$ = worse"),
    ]

    for axis_idx, (metric_name, ylabel) in enumerate(metric_specs):
        ax = axes[0, axis_idx]
        x_base, bar_width = _bar_positions(len(category_order), len(models))
        for model_idx, model in enumerate(models):
            model_metric = agg[(agg["Model"] == model) & (agg["Metric"] == metric_name)]
            means: list[float] = []
            sems: list[float] = []
            for category in category_order:
                row = model_metric[model_metric["AxisCategory"] == category]
                means.append(
                    float(row.iloc[0]["Mean"]) if not row.empty else float("nan")
                )
                sems.append(float(row.iloc[0]["SEM"]) if not row.empty else 0.0)
            x_pos = x_base + model_idx * bar_width
            ax.bar(
                x_pos,
                means,
                width=bar_width,
                color=_model_color(model),
                label=model,
                yerr=sems,
                capsize=3,
                alpha=0.9,
            )

        ax.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
        ax.set_xticks(x_base + bar_width * (len(models) - 1) / 2)
        ax.set_xticklabels(category_labels)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[0, 0].legend(title="Model", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return agg


# ── Shift figures: one PDF per distribution-shift axis ──────────────────

_SHIFT_AXIS_SPECS: dict[str, tuple[str, str]] = {
    "graph": ("graph", "Graph Shift"),
    "mechanism": ("mechanism", "Mechanism Shift"),
    "noise": ("noise", "Noise Shift"),
    "compound": ("compound", "Compound Shift"),
}
"""Mapping from shift key → (AxisCategory value, human-readable title)."""


def generate_shift_figure(
    raw_df: pd.DataFrame,
    *,
    shift_axis: str,
    output_path: Path,
) -> pd.DataFrame:
    """Generate an error-bar plot for a single distribution-shift axis.

    For each shift axis (graph, mechanism, noise, compound) this produces one
    PDF with x-axis = dataset, hue = model, y-axis = E-SID. Fixed-size thesis
    plots are restricted to the $N_G=20$, $N=500$ regime.

    Args:
        raw_df: Per-task raw DataFrame with ``AxisCategory`` and ``Dataset``.
        shift_axis: One of ``"graph"``, ``"mechanism"``, ``"noise"``,
            ``"compound"``.
        output_path: Where to write the figure.

    Returns:
        The aggregated DataFrame used for plotting.
    """
    if shift_axis not in _SHIFT_AXIS_SPECS:
        raise ValueError(
            f"Unknown shift_axis '{shift_axis}'. "
            f"Expected one of {sorted(_SHIFT_AXIS_SPECS)}."
        )
    axis_cat, axis_title = _SHIFT_AXIS_SPECS[shift_axis]

    metric_name = "ne-sid" if shift_axis == "compound" else "e-sid"
    subset = raw_df[
        raw_df["Metric"].eq(metric_name) & raw_df["AxisCategory"].isin(["id", axis_cat])
    ].copy()
    subset = subset[_is_fixed_size_task_frame(subset)]
    if subset.empty:
        raise EmptyAnalysisDataError(
            f"No raw {metric_name} data for shift axis '{shift_axis}'."
        )

    agg = (
        subset.groupby(
            ["Model", "DatasetKey", "Dataset", "AxisCategory"], dropna=False
        )["Value"]
        .agg(Mean="mean", SEM=_metric_sem)
        .reset_index()
    )

    agg["DatasetLabel"] = agg.apply(
        lambda row: _thesis_dataset_label(str(row["DatasetKey"]), str(row["Dataset"])),
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
            color = _model_color(model)
            marker = MODEL_MARKERS.get(model, "o")
            ax.errorbar(
                xs,
                means,
                yerr=sems,
                fmt=marker,
                label=model,
                color=color,
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
    ax.set_title(axis_title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Model", fontsize=9, loc="best")

    id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")
    if 0 < id_count < len(datasets):
        ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
        ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)
        ymax = ax.get_ylim()[1]
        ax.text((id_count - 1) / 2, ymax, "ID", ha="center", va="bottom", fontsize=10)
        ax.text(
            (id_count + len(datasets) - 1) / 2,
            ymax,
            "OOD",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return agg


def generate_transfer_figure(
    raw_df: pd.DataFrame,
    *,
    axis: str,
    output_path: Path,
) -> pd.DataFrame:
    """Generate the node-count or sample-count transfer figure."""

    if axis == "nodes":
        axis_categories = {"id", "nodes"}
        x_col = "NNodes"
        metric_specs = [
            ("ne-sid", "Normalized E-SID", False),
            ("ne-shd", "Normalized E-SHD", False),
            ("e-edgef1", "E-Edge F1", True),
        ]
        xlabel = "Target node count"
    elif axis == "samples":
        axis_categories = {"id", "samples"}
        x_col = "SamplesPerTask"
        metric_specs = [
            ("ne-sid", "Normalized E-SID", False),
            ("ne-shd", "Normalized E-SHD", False),
            ("e-edgef1", "E-Edge F1", True),
        ]
        xlabel = "Observational samples per task"
    else:
        raise ValueError(f"Unknown transfer axis '{axis}'.")

    needed_metrics = [metric for metric, _, _ in metric_specs]
    subset = raw_df[
        raw_df["AxisCategory"].isin(axis_categories)
        & raw_df["Metric"].isin(needed_metrics)
    ].copy()
    subset = subset[subset[x_col].notna()]
    if subset.empty:
        raise EmptyAnalysisDataError(f"No transfer data available for axis '{axis}'.")

    agg = (
        subset.groupby(["Model", x_col, "Metric"], dropna=False)["Value"]
        .agg(Mean="mean", SEM=_metric_sem)
        .reset_index()
    )
    agg[x_col] = agg[x_col].astype(int)

    models = list(PAPER_MODEL_LABELS.values())
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(15.5, 4.8), squeeze=False)
    for axis_idx, (metric_name, ylabel, higher_is_better) in enumerate(metric_specs):
        ax = axes[0, axis_idx]
        metric_df = agg[agg["Metric"] == metric_name]
        for model_idx, model in enumerate(models):
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


# ── Uncertainty scatter: SID vs uncertainty per model ──────────────────


# Colours per OOD category for scatter plots.
_OOD_CATEGORY_COLORS: dict[str, str] = {
    "ID": "#2ca02c",
    "OOD-Graph": "#d62728",
    "OOD-Mech": "#9467bd",
    "OOD-Noise": "#8c564b",
    "OOD-Both": "#e377c2",
    "OOD": "#17becf",
}


def generate_uncertainty_scatter(
    raw_df: pd.DataFrame,
    *,
    score_metric: str = "edge_entropy",
    output_path: Path,
) -> pd.DataFrame:
    """Scatter plot of E-SID vs an uncertainty score, one subplot per model.

    Each point is one dataset (aggregated across tasks). Points are coloured
    by OOD category. A Spearman correlation
    coefficient is annotated on each subplot.

    Args:
        raw_df: Per-task raw DataFrame with ``AxisCategory``, ``Dataset``,
            ``DatasetKey``, ``Model`` columns.
        score_metric: The uncertainty metric to plot on the x-axis
            (``"edge_entropy"`` or ``"graph_nll_per_edge"``).
        output_path: Where to write the figure.

    Returns:
        The aggregated DataFrame used for plotting.
    """
    from scipy.stats import spearmanr

    from causal_meta.analysis.failure_modes import ood_category

    needed = {"e-sid", score_metric}
    subset = raw_df[raw_df["Metric"].isin(needed)].copy()
    if subset.empty:
        raise EmptyAnalysisDataError(
            f"No data for uncertainty scatter (need {needed})."
        )

    # Aggregate per (Model, DatasetKey, Metric)
    agg = (
        subset.groupby(["Model", "DatasetKey", "Dataset", "Metric"], dropna=False)[
            "Value"
        ]
        .mean()
        .reset_index()
    )

    # Pivot so each row is (Model, DatasetKey) with columns for e-sid and score_metric
    pivot = agg.pivot_table(
        index=["Model", "DatasetKey", "Dataset"],
        columns="Metric",
        values="Value",
    ).reset_index()
    pivot.columns.name = None

    if "e-sid" not in pivot.columns or score_metric not in pivot.columns:
        raise EmptyAnalysisDataError(
            f"Pivot missing required columns for uncertainty scatter."
        )

    pivot = pivot.dropna(subset=["e-sid", score_metric])
    if pivot.empty:
        raise EmptyAnalysisDataError(
            "No overlapping (E-SID, uncertainty) data after aggregation."
        )

    # Add OOD category for colouring
    pivot["OODCategory"] = pivot["DatasetKey"].map(
        lambda k: ood_category(k, binary=False)
    )

    models = [m for m in PAPER_MODEL_LABELS.values() if m in pivot["Model"].unique()]
    n_models = len(models)
    if n_models == 0:
        raise EmptyAnalysisDataError("No models with uncertainty scatter data.")

    fig, axes = plt.subplots(1, n_models, figsize=(6.2 * n_models, 5.8), squeeze=False)

    score_label = (
        "Edge Entropy" if score_metric == "edge_entropy" else "Graph NLL / edge"
    )

    all_categories = sorted(pivot["OODCategory"].unique())

    for ax_idx, model in enumerate(models):
        ax = axes[0, ax_idx]
        model_df = pivot[pivot["Model"] == model]

        # Plot each OOD category with its colour
        for cat in all_categories:
            cat_df = model_df[model_df["OODCategory"] == cat]
            if cat_df.empty:
                continue
            color = _OOD_CATEGORY_COLORS.get(cat, "#aaaaaa")
            ax.scatter(
                cat_df[score_metric],
                cat_df["e-sid"],
                c=color,
                label=cat,
                s=60,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )

        # Spearman correlation
        if len(model_df) >= 3:
            rho, p_val = spearmanr(model_df[score_metric], model_df["e-sid"])
            if np.isfinite(rho):
                p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                ax.annotate(
                    f"Spearman $\\rho$={rho:.2f}\n({p_str})",
                    xy=(0.03, 0.97),
                    xycoords="axes fraction",
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                )

        ax.set_xlabel(score_label, fontsize=11)
        ax.set_ylabel(r"$\mathbb{E}$-SID", fontsize=11)
        ax.set_title(model, fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        if ax_idx == 0:
            ax.legend(
                title="Category",
                fontsize=8,
                title_fontsize=9,
                loc="lower right",
            )

    fig.tight_layout(pad=1.0, w_pad=1.0)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return pivot


def generate_ece_summary_table(raw_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Generate a compact ECE summary split by ID and OOD."""

    subset = raw_df[raw_df["Metric"].eq("ece")].copy()
    if subset.empty:
        raise EmptyAnalysisDataError("No ECE data available for calibration summary.")

    subset["Split"] = np.where(subset["AxisCategory"].eq("id"), "ID", "OOD")
    split_agg = (
        subset.groupby(["Model", "Split"], dropna=False)["Value"]
        .agg(Mean="mean", SEM=_metric_sem)
        .reset_index()
    )
    overall_agg = (
        subset.groupby(["Model"], dropna=False)["Value"]
        .agg(Mean="mean", SEM=_metric_sem)
        .reset_index()
    )
    overall_agg["Split"] = "Overall"

    combined = pd.concat([split_agg, overall_agg], ignore_index=True)
    split_order = ["ID", "OOD", "Overall"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Expected calibration error (ECE) of posterior edge confidence. Lower is better. ID and OOD splits summarize whether edge-confidence calibration degrades under shift.}",
        r"\label{tab:ece_summary}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{ID ECE $\downarrow$} & \textbf{OOD ECE $\downarrow$} & \textbf{Overall ECE $\downarrow$} \\",
        r"\midrule",
    ]

    best_by_split: dict[str, float] = {}
    for split in split_order:
        split_vals = combined[combined["Split"] == split]["Mean"].dropna()
        best_by_split[split] = (
            float(split_vals.min()) if not split_vals.empty else float("inf")
        )

    for model in list(PAPER_MODEL_LABELS.values()):
        model_rows = combined[combined["Model"] == model]
        cells: list[str] = []
        for split in split_order:
            row = model_rows[model_rows["Split"] == split]
            if row.empty:
                cells.append("-")
                continue
            mean = float(row.iloc[0]["Mean"])
            sem = float(row.iloc[0]["SEM"])
            cell = _format_value(mean, sem)
            if abs(mean - best_by_split[split]) < 1e-6:
                cell = _bold_if_best(cell, is_best=True)
            cells.append(cell)
        lines.append(f"{model} & " + " & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    output_path.write_text("\n".join(lines) + "\n")
    return combined


def generate_ood_detection_summary_table(
    raw_df: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    """Generate one combined OOD detection table for both uncertainty scores."""

    edge_entropy_df = compute_ood_detection_metrics(raw_df, score_metric="edge_entropy")
    graph_nll_df = compute_ood_detection_metrics(
        raw_df, score_metric="graph_nll_per_edge"
    )
    if edge_entropy_df.empty and graph_nll_df.empty:
        raise EmptyAnalysisDataError("No OOD detection metrics could be computed.")

    frames: list[pd.DataFrame] = []
    for score_name, detection_df in (
        ("edge_entropy", edge_entropy_df),
        ("graph_nll_per_edge", graph_nll_df),
    ):
        if detection_df.empty:
            continue
        renamed = detection_df.rename(
            columns={
                "AUROC": f"{score_name}_AUROC",
                "AUPRC": f"{score_name}_AUPRC",
                "N_ID": f"{score_name}_N_ID",
                "N_OOD": f"{score_name}_N_OOD",
            }
        )
        frames.append(renamed.drop(columns=["ScoreMetric"], errors="ignore"))

    if not frames:
        raise EmptyAnalysisDataError("No OOD detection frames after filtering.")

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on=["RunID", "Model"], how="outer")
    combined = combined.sort_values("Model")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{ID--OOD detection performance using posterior uncertainty scores. Higher AUROC/AUPRC is better.}",
        r"\label{tab:ood_detection}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Entropy AUROC} & \textbf{Entropy AUPRC} & \textbf{Graph NLL / edge AUROC} & \textbf{Graph NLL / edge AUPRC} \\",
        r"\midrule",
    ]

    # Find per-column best (highest) for bolding
    metric_cols = [
        "edge_entropy_AUROC",
        "edge_entropy_AUPRC",
        "graph_nll_per_edge_AUROC",
        "graph_nll_per_edge_AUPRC",
    ]
    col_best: dict[str, float] = {}
    for col in metric_cols:
        if col in combined.columns:
            vals = combined[col].dropna()
            col_best[col] = float(vals.max()) if not vals.empty else float("-inf")

    def _fmt_cell(value: float, col: str) -> str:
        cell = f"{value:.3f}"
        if np.isfinite(value) and abs(value - col_best.get(col, float("-inf"))) < 1e-6:
            cell = r"\textbf{" + cell + "}"
        return cell

    for _, row in combined.iterrows():
        entropy_auroc = float(row.get("edge_entropy_AUROC", float("nan")))
        entropy_auprc = float(row.get("edge_entropy_AUPRC", float("nan")))
        graph_auroc = float(row.get("graph_nll_per_edge_AUROC", float("nan")))
        graph_auprc = float(row.get("graph_nll_per_edge_AUPRC", float("nan")))
        lines.append(
            f"{row['Model']} & "
            f"{_fmt_cell(entropy_auroc, 'edge_entropy_AUROC')} & "
            f"{_fmt_cell(entropy_auprc, 'edge_entropy_AUPRC')} & "
            f"{_fmt_cell(graph_auroc, 'graph_nll_per_edge_AUROC')} & "
            f"{_fmt_cell(graph_auprc, 'graph_nll_per_edge_AUPRC')}" + r" \\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    output_path.write_text("\n".join(lines) + "\n")
    return combined


def _write_selected_runs(
    selected_runs: Sequence[SelectedRun], output_path: Path
) -> None:
    payload = {
        run.model_dir: {
            "run_dir": str(run.run_dir),
            "run_id": run.run_id,
            "run_name": run.run_name,
            "model_name": run.model_name,
        }
        for run in selected_runs
    }
    _write_json(output_path, payload)


def run_thesis_analysis(
    *,
    input_root: Path,
    thesis_root: Path,
    strict: bool = True,
) -> Path:
    """Run the curated thesis analysis pipeline and rebuild generated outputs."""

    selected_runs = resolve_thesis_run_directories(input_root)
    run_dirs = [run.run_dir for run in selected_runs]
    temp_root = _prepare_generated_workspace(thesis_root)
    generated_files: list[str] = []

    try:
        summary_df = _prepare_summary_dataframe(run_dirs)
        raw_df = _prepare_raw_dataframe(run_dirs)

        data_dir = temp_root / "data"
        tables_dir = temp_root / "tables"
        figures_dir = temp_root / "figures"
        snippets_dir = temp_root / "snippets"
        provenance_dir = temp_root / "provenance"

        summary_df.to_csv(data_dir / "summary_metrics.csv", index=False)
        raw_df.to_csv(data_dir / "raw_task_metrics.csv", index=False)
        generated_files.extend(
            [
                "data/summary_metrics.csv",
                "data/raw_task_metrics.csv",
            ]
        )

        anchor_df = generate_results_anchor_table(
            raw_df, tables_dir / "results_anchor.tex"
        )
        anchor_df.to_csv(data_dir / "results_anchor.csv", index=False)
        generated_files.extend(
            [
                "tables/results_anchor.tex",
                "data/results_anchor.csv",
            ]
        )

        # ── Per-shift-axis figures (graph, mechanism, noise, compound) ─
        for shift_key in _SHIFT_AXIS_SPECS:
            fig_name = f"shift_{shift_key}.pdf"
            csv_name = f"shift_{shift_key}.csv"
            try:
                shift_df = generate_shift_figure(
                    raw_df,
                    shift_axis=shift_key,
                    output_path=figures_dir / fig_name,
                )
                shift_df.to_csv(data_dir / csv_name, index=False)
                generated_files.extend([f"figures/{fig_name}", f"data/{csv_name}"])
            except EmptyAnalysisDataError:
                if strict:
                    raise
                log.warning("Shift figure for '%s' skipped (no data).", shift_key)

        # ── Uncertainty scatter plots ──────────────────────────────────
        for score_metric in ("edge_entropy", "graph_nll_per_edge"):
            fig_name = f"uncertainty_scatter_{score_metric}.pdf"
            csv_name = f"uncertainty_scatter_{score_metric}.csv"
            try:
                scatter_df = generate_uncertainty_scatter(
                    raw_df,
                    score_metric=score_metric,
                    output_path=figures_dir / fig_name,
                )
                scatter_df.to_csv(data_dir / csv_name, index=False)
                generated_files.extend([f"figures/{fig_name}", f"data/{csv_name}"])
            except EmptyAnalysisDataError:
                if strict:
                    raise
                log.warning(
                    "Uncertainty scatter for '%s' skipped (no data).",
                    score_metric,
                )

        node_transfer_df = generate_transfer_figure(
            raw_df, axis="nodes", output_path=figures_dir / "node_transfer.pdf"
        )
        node_transfer_df.to_csv(data_dir / "node_transfer.csv", index=False)
        generated_files.extend(
            [
                "figures/node_transfer.pdf",
                "data/node_transfer.csv",
            ]
        )

        sample_transfer_df = generate_transfer_figure(
            raw_df, axis="samples", output_path=figures_dir / "sample_transfer.pdf"
        )
        sample_transfer_df.to_csv(data_dir / "sample_transfer.csv", index=False)
        generated_files.extend(
            [
                "figures/sample_transfer.pdf",
                "data/sample_transfer.csv",
            ]
        )

        ood_detection_df = generate_ood_detection_summary_table(
            raw_df, tables_dir / "ood_detection.tex"
        )
        ood_detection_df.to_csv(data_dir / "ood_detection.csv", index=False)
        generated_files.extend(
            [
                "tables/ood_detection.tex",
                "data/ood_detection.csv",
            ]
        )

        ece_df = generate_ece_summary_table(raw_df, tables_dir / "ece_summary.tex")
        ece_df.to_csv(data_dir / "ece_summary.csv", index=False)
        generated_files.extend(
            [
                "tables/ece_summary.tex",
                "data/ece_summary.csv",
            ]
        )

        selective_df = compute_selective_prediction(raw_df)
        if not selective_df.empty:
            generate_selective_prediction_pareto(
                selective_df, figures_dir / "selective_prediction.pdf"
            )
            selective_df.to_csv(data_dir / "selective_prediction.csv", index=False)
            generated_files.extend(
                [
                    "figures/selective_prediction.pdf",
                    "data/selective_prediction.csv",
                ]
            )

        generate_robustness_table(summary_df, tables_dir / "fixed_ood_appendix.tex")
        generated_files.append("tables/fixed_ood_appendix.tex")

        try:
            generate_distance_regression_table(
                summary_df, tables_dir / "distance_regression.tex"
            )
            generated_files.append("tables/distance_regression.tex")
        except Exception:
            if strict:
                raise
            log.warning("Distance regression table generation failed.", exc_info=True)

        try:
            failure_df = raw_df[
                raw_df["Metric"].isin(
                    ["sparsity_ratio", "skeleton_f1", "orientation_accuracy"]
                )
            ].copy()
            classified = classify_failure_modes(failure_df)
            if not classified.empty:
                fractions = failure_mode_fractions(classified)
                generate_failure_mode_bar(fractions, figures_dir / "failure_modes.pdf")
                fractions.to_csv(data_dir / "failure_modes.csv", index=False)
                generated_files.extend(
                    [
                        "figures/failure_modes.pdf",
                        "data/failure_modes.csv",
                    ]
                )
                # Per-model greyscale failure mode bars for amortised models
                for model_name in sorted(AMORTISED_MODELS):
                    safe_name = model_name.lower().replace("-", "_")
                    fig_name = f"failure_modes_{safe_name}.pdf"
                    generate_per_model_failure_mode_bar(
                        fractions,
                        model=model_name,
                        output_path=figures_dir / fig_name,
                    )
                    generated_files.append(f"figures/{fig_name}")
        except Exception:
            if strict:
                raise
            log.warning("Failure-mode analysis failed.", exc_info=True)

        try:
            posterior_df = run_posterior_diagnostics_from_runs(run_dirs)
            if not posterior_df.empty:
                posterior_df = posterior_df.copy()
                posterior_df["Model"] = posterior_df["Model"].map(_paper_model_label)
                generate_event_probability_bar(
                    posterior_df, figures_dir / "event_probabilities.pdf"
                )
                generate_posterior_diagnostic_violins(
                    posterior_df, figures_dir / "posterior_diagnostics.pdf"
                )
                generated_files.extend(
                    [
                        "figures/event_probabilities.pdf",
                        "figures/posterior_diagnostics.pdf",
                    ]
                )
        except Exception:
            if strict:
                raise
            log.warning("Posterior diagnostics failed.", exc_info=True)

        _write_selected_runs(selected_runs, provenance_dir / "selected_runs.json")
        _write_results_macros(selected_runs, snippets_dir / "results_macros.tex")
        generated_files.extend(
            [
                "provenance/selected_runs.json",
                "snippets/results_macros.tex",
            ]
        )

        has_mock_runs = any(
            "mock" in run.run_name.lower() or "mock" in run.run_id.lower()
            for run in selected_runs
        )
        if has_mock_runs:
            log.warning(
                "Selected thesis runs include mock/provisional outputs; treat generated artifacts as provisional."
            )

        analysis_report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_root": str(input_root.resolve()),
            "thesis_root": str(thesis_root.resolve()),
            "strict": bool(strict),
            "has_mock_runs": has_mock_runs,
            "generated_files": sorted(generated_files),
        }
        _write_json(provenance_dir / "analysis_report.json", analysis_report)
        generated_files.append("provenance/analysis_report.json")

        final_root = _finalize_generated_workspace(temp_root, thesis_root)
        log.info("Generated thesis artifacts under %s", final_root)
        return final_root
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild paper/final_thesis/generated from curated runs under "
            "experiments/thesis_runs/."
        )
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="experiments/thesis_runs",
        help="Curated one-run-per-model input root (default: experiments/thesis_runs).",
    )
    parser.add_argument(
        "--thesis-root",
        type=str,
        default="paper/final_thesis",
        help="Thesis repository root containing the generated/ folder.",
    )
    parser.add_argument(
        "--best-effort",
        action="store_true",
        help="Continue when optional diagnostics fail instead of aborting.",
    )
    args = parser.parse_args()

    run_thesis_analysis(
        input_root=Path(args.input_root),
        thesis_root=Path(args.thesis_root),
        strict=not bool(args.best_effort),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except (EmptyAnalysisDataError, FileNotFoundError, ThesisRunSelectionError) as exc:
        log.error("Thesis analysis failed: %s", exc)
        raise SystemExit(1) from exc
