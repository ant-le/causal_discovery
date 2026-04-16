from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_meta.analysis.common.thesis import format_value, metric_sem
from causal_meta.analysis.diagnostics.failure_modes import ood_category
from causal_meta.analysis.uncertainty.ood_detection import compute_ood_detection_metrics
from causal_meta.analysis.utils import (
    EmptyAnalysisDataError,
    MODEL_COLORS,
    MODEL_MARKERS,
    PAPER_MODEL_LABELS,
)


def _model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#555555")


def _bold_if_best(value: str, *, is_best: bool) -> str:
    return r"\textbf{" + value + "}" if is_best else value


def generate_uncertainty_scatter(
    raw_df: pd.DataFrame, *, score_metric: str, output_path: Path
) -> pd.DataFrame:
    from scipy.stats import spearmanr

    needed = {"ne-sid", score_metric}
    subset = raw_df[raw_df["Metric"].isin(needed)].copy()
    agg = (
        subset.groupby(["Model", "DatasetKey", "Dataset", "Metric"], dropna=False)[
            "Value"
        ]
        .mean()
        .reset_index()
    )
    pivot = agg.pivot_table(
        index=["Model", "DatasetKey", "Dataset"], columns="Metric", values="Value"
    ).reset_index()
    pivot.columns.name = None
    if score_metric not in pivot.columns or "ne-sid" not in pivot.columns:
        raise EmptyAnalysisDataError(
            f"Missing required columns for uncertainty scatter: {score_metric}."
        )
    pivot = pivot.dropna(subset=["ne-sid", score_metric])
    if pivot.empty:
        raise EmptyAnalysisDataError("No uncertainty scatter data available.")
    pivot["OODCategory"] = pivot["DatasetKey"].map(
        lambda k: ood_category(k, binary=False)
    )
    models = [m for m in PAPER_MODEL_LABELS.values() if m in pivot["Model"].unique()]
    if not models:
        raise EmptyAnalysisDataError("No models with uncertainty scatter data.")
    fig, axes = plt.subplots(
        1, len(models), figsize=(6.2 * len(models), 5.8), squeeze=False
    )
    score_label = (
        "Edge Entropy" if score_metric == "edge_entropy" else "Graph NLL / edge"
    )
    category_colors = {
        "ID": "#2ca02c",
        "OOD-Graph": "#d62728",
        "OOD-Mech": "#9467bd",
        "OOD-Noise": "#8c564b",
        "OOD-Both": "#e377c2",
        "OOD-Nodes": "#ff7f0e",
        "OOD-Samples": "#1f77b4",
        "OOD": "#17becf",
    }
    all_categories = sorted(pivot["OODCategory"].unique())
    for ax_idx, model in enumerate(models):
        ax = axes[0, ax_idx]
        model_df = pivot[pivot["Model"] == model]
        for cat in all_categories:
            cat_df = model_df[model_df["OODCategory"] == cat]
            if cat_df.empty:
                continue
            ax.scatter(
                cat_df[score_metric],
                cat_df["ne-sid"],
                c=category_colors.get(cat, "#aaaaaa"),
                label=cat,
                s=60,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
            )
        # Grey reference line at ne-SID = 0 (perfect causal discovery).
        ax.axhline(y=0, color="grey", linestyle="--", linewidth=1, alpha=0.6)
        if len(model_df) >= 3:
            rho, p_val = spearmanr(model_df[score_metric], model_df["ne-sid"])
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
        ax.set_xlabel(score_label)
        ax.set_ylabel(r"Normalized $\mathbb{E}$-SID")
        ax.set_title(model, fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)

    # Shared legend on top.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Category",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(labels),
            fontsize=9,
            frameon=False,
        )

    suptitle = (
        "Edge Entropy vs. Structural Error"
        if score_metric == "edge_entropy"
        else "Graph NLL / Edge vs. Structural Error"
    )
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return pivot


def generate_ece_summary_table(raw_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    subset = raw_df[raw_df["Metric"].eq("ece")].copy()
    if subset.empty:
        raise EmptyAnalysisDataError("No ECE data available for calibration summary.")
    subset["Split"] = np.where(subset["AxisCategory"].eq("id"), "ID", "OOD")
    split_agg = (
        subset.groupby(["Model", "Split"], dropna=False)["Value"]
        .agg(Mean="mean", SEM=metric_sem)
        .reset_index()
    )
    overall_agg = (
        subset.groupby(["Model"], dropna=False)["Value"]
        .agg(Mean="mean", SEM=metric_sem)
        .reset_index()
    )
    overall_agg["Split"] = "Overall"
    combined = pd.concat([split_agg, overall_agg], ignore_index=True)
    split_order = ["ID", "OOD", "Overall"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Expected calibration error (ECE) of posterior edge confidence. Lower is better. ID and OOD splits summarize whether edge-confidence calibration degrades under shift.}",
        r"\label{tab:ece_summary}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{ID ECE $\downarrow$} & \textbf{OOD ECE $\downarrow$} & \textbf{Overall ECE $\downarrow$} \\",
        r"\midrule",
    ]
    best_by_split = {
        split: float(combined[combined["Split"] == split]["Mean"].dropna().min())
        for split in split_order
    }
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
            cell = format_value(mean, sem)
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
    edge_entropy_df = compute_ood_detection_metrics(raw_df, score_metric="edge_entropy")
    graph_nll_df = compute_ood_detection_metrics(
        raw_df, score_metric="graph_nll_per_edge"
    )
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
        raise EmptyAnalysisDataError("No OOD detection metrics could be computed.")
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on=["RunID", "Model"], how="outer")
    combined = combined.sort_values("Model")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{ID--OOD detection performance using posterior uncertainty scores. Higher AUROC/AUPRC is better.}",
        r"\label{tab:ood_detection}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Entropy AUROC} & \textbf{Entropy AUPRC} & \textbf{Graph NLL / edge AUROC} & \textbf{Graph NLL / edge AUPRC} \\",
        r"\midrule",
    ]
    metric_cols = [
        "edge_entropy_AUROC",
        "edge_entropy_AUPRC",
        "graph_nll_per_edge_AUROC",
        "graph_nll_per_edge_AUPRC",
    ]
    col_best = {
        col: float(combined[col].dropna().max())
        for col in metric_cols
        if col in combined.columns
    }

    def _fmt_cell(value: float, col: str) -> str:
        cell = f"{value:.3f}"
        return (
            r"\textbf{" + cell + "}"
            if np.isfinite(value)
            and abs(value - col_best.get(col, float("-inf"))) < 1e-6
            else cell
        )

    for _, row in combined.iterrows():
        lines.append(
            f"{row['Model']} & {_fmt_cell(float(row.get('edge_entropy_AUROC', float('nan'))), 'edge_entropy_AUROC')} & {_fmt_cell(float(row.get('edge_entropy_AUPRC', float('nan'))), 'edge_entropy_AUPRC')} & {_fmt_cell(float(row.get('graph_nll_per_edge_AUROC', float('nan'))), 'graph_nll_per_edge_AUROC')} & {_fmt_cell(float(row.get('graph_nll_per_edge_AUPRC', float('nan'))), 'graph_nll_per_edge_AUPRC')}"
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    output_path.write_text("\n".join(lines) + "\n")
    return combined
