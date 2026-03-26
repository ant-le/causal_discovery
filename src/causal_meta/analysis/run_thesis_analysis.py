from __future__ import annotations

import argparse
import json
import logging
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
    EmptyAnalysisDataError,
    PAPER_MODEL_LABELS,
    load_raw_task_dataframe,
    load_runs_dataframe,
    map_dataset_description,
)

log = logging.getLogger(__name__)

EXPECTED_MODEL_DIRS: tuple[str, ...] = tuple(PAPER_MODEL_LABELS.keys())

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
    if dataset_key_norm.startswith("ood_both_"):
        return "compound"
    if dataset_key_norm.startswith("ood_nodes_"):
        return "nodes"
    if dataset_key_norm.startswith("ood_samples_"):
        return "samples"
    return "other"


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
        "inference_time_s",
        "sparsity_ratio",
        "skeleton_f1",
        "orientation_accuracy",
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
    raw_df["AxisCategory"] = raw_df["DatasetKey"].map(_axis_category)
    return raw_df


def generate_results_anchor_table(
    raw_df: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    """Generate the in-distribution speed--robustness anchor table."""

    subset = raw_df[
        raw_df["AxisCategory"].eq("id")
        & raw_df["Metric"].isin(["ne-sid", "ne-shd", "e-edgef1", "inference_time_s"])
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
        ("inference_time_s", r"Runtime / dataset", False),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Speed--robustness anchor on the in-distribution families. Values report task-level means and standard errors aggregated over all in-distribution tasks.}",
        r"\label{tab:results_anchor}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Normalized $\mathbb{E}$-SID $\downarrow$} & \textbf{Normalized $\mathbb{E}$-SHD $\downarrow$} & \textbf{$\mathbb{E}$-Edge F1 $\uparrow$} & \textbf{Runtime / dataset $\downarrow$} \\",
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
    cmap = plt.get_cmap("tab10")

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
                color=cmap(model_idx),
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
            ("e-edgef1", "E-Edge F1", True),
            ("graph_nll_per_edge", "Graph NLL / edge", False),
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
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(16, 4.8), squeeze=False)
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
                color=cmap(model_idx),
                marker="o",
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

    for _, row in combined.iterrows():
        entropy_auroc = float(row.get("edge_entropy_AUROC", float("nan")))
        entropy_auprc = float(row.get("edge_entropy_AUPRC", float("nan")))
        graph_auroc = float(row.get("graph_nll_per_edge_AUROC", float("nan")))
        graph_auprc = float(row.get("graph_nll_per_edge_AUPRC", float("nan")))
        lines.append(
            f"{row['Model']} & {entropy_auroc:.3f} & {entropy_auprc:.3f} & "
            f"{graph_auroc:.3f} & {graph_auprc:.3f}" + r" \\"
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

        fixed_ood_df = generate_fixed_ood_figure(
            raw_df, figures_dir / "fixed_ood_degradation.pdf"
        )
        fixed_ood_df.to_csv(data_dir / "fixed_ood_degradation.csv", index=False)
        generated_files.extend(
            [
                "figures/fixed_ood_degradation.pdf",
                "data/fixed_ood_degradation.csv",
            ]
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

        analysis_report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_root": str(input_root.resolve()),
            "thesis_root": str(thesis_root.resolve()),
            "strict": bool(strict),
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
