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
    save_figure_data,
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
    save_figure_data(output_path, pivot)
    return pivot


# ── Core models used in the combined scatter (no Random) ───────────
_CORE_MODELS_ORDERED: tuple[str, ...] = ("AviCi", "BCNP", "DiBS", "BayesDAG")

_CATEGORY_COLORS: dict[str, str] = {
    "ID": "#2ca02c",
    "OOD-Graph": "#d62728",
    "OOD-Mech": "#9467bd",
    "OOD-Noise": "#8c564b",
    "OOD-Both": "#e377c2",
    "OOD-Nodes": "#ff7f0e",
    "OOD-Samples": "#1f77b4",
    "OOD": "#17becf",
}


def _add_ideal_tracking_line(
    ax: plt.Axes,
    x_vals: pd.Series,
    y_vals: pd.Series,
    *,
    linear: bool = False,
) -> None:
    """Draw a reference curve showing the ideal tracking direction.

    When *linear* is ``False`` (default, for entropy vs SID), the reference
    is a convex power-law ``y = a * x^p`` (p=2), reflecting the causal-
    cascade amplification between entropy and SID.

    When *linear* is ``True`` (for GraphNLL vs SID), the reference is a
    straight line ``y = a * x``, because NLL divergence scales approximately
    linearly with structural error.
    """
    x_clean = x_vals.dropna()
    y_clean = y_vals.dropna()
    if x_clean.empty or y_clean.empty:
        return
    x_hi = float(np.percentile(x_clean, 90))
    y_hi = float(np.percentile(y_clean, 90))
    if x_hi <= 0 or y_hi <= 0:
        return

    x_max = float(x_clean.max()) * 1.05
    xs = np.linspace(0, x_max, 100)

    if linear:
        a = y_hi / x_hi
        ys = a * xs
    else:
        p = 2.0
        a = y_hi / (x_hi**p)
        ys = a * xs**p

    ax.plot(
        xs,
        ys,
        color="#888888",
        linestyle=":",
        linewidth=1.3,
        alpha=0.7,
        zorder=0,
        label="_ideal",  # hidden from legend
    )


def generate_uncertainty_scatter_combined(
    raw_df: pd.DataFrame,
    *,
    output_path: Path,
) -> pd.DataFrame:
    """2x4 combined scatter: top row = edge entropy, bottom = GraphNLL.

    Omits the Random baseline and adds an ideal-tracking reference line
    to each panel.
    """
    from scipy.stats import spearmanr

    score_metrics = ("edge_entropy", "graph_nll_per_edge")
    score_labels = ("Edge Entropy", "Graph NLL / edge")

    needed = {"ne-sid", *score_metrics}
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
    for col in ("ne-sid", *score_metrics):
        if col not in pivot.columns:
            raise EmptyAnalysisDataError(
                f"Missing required column for combined scatter: {col}."
            )
    pivot = pivot.dropna(subset=["ne-sid"])
    if pivot.empty:
        raise EmptyAnalysisDataError("No uncertainty scatter data available.")
    pivot["OODCategory"] = pivot["DatasetKey"].map(
        lambda k: ood_category(k, binary=False)
    )

    models = [m for m in _CORE_MODELS_ORDERED if m in pivot["Model"].unique()]
    if not models:
        raise EmptyAnalysisDataError("No core models found for combined scatter.")

    n_models = len(models)
    fig, axes = plt.subplots(
        2,
        n_models,
        figsize=(5.0 * n_models, 9.5),
        squeeze=False,
    )

    all_categories = sorted(pivot["OODCategory"].unique())

    for row_idx, (score_metric, score_label) in enumerate(
        zip(score_metrics, score_labels)
    ):
        row_pivot = pivot.dropna(subset=[score_metric])
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            model_df = row_pivot[row_pivot["Model"] == model]

            for cat in all_categories:
                cat_df = model_df[model_df["OODCategory"] == cat]
                if cat_df.empty:
                    continue
                ax.scatter(
                    cat_df[score_metric],
                    cat_df["ne-sid"],
                    c=_CATEGORY_COLORS.get(cat, "#aaaaaa"),
                    label=cat if row_idx == 0 else "_nolegend_",
                    s=50,
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.4,
                )

            # Ideal-tracking reference line.
            # Convex for entropy (row 0), linear for GraphNLL (row 1).
            _add_ideal_tracking_line(
                ax,
                model_df[score_metric],
                model_df["ne-sid"],
                linear=(row_idx == 1),
            )

            # Spearman annotation.
            if len(model_df) >= 3:
                rho, p_val = spearmanr(model_df[score_metric], model_df["ne-sid"])
                if np.isfinite(rho):
                    p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                    ax.annotate(
                        f"$\\rho$={rho:.2f} ({p_str})",
                        xy=(0.03, 0.97),
                        xycoords="axes fraction",
                        fontsize=8,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8),
                    )

            ax.set_xlabel(score_label, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(
                    r"Normalized $\mathbb{E}$-SID",
                    fontsize=9,
                )
            else:
                ax.set_ylabel("")
            if row_idx == 0:
                ax.set_title(model, fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.35)

    # Shared legend on top.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Filter out internal labels.
    keep = [(h, l) for h, l in zip(handles, labels) if not l.startswith("_")]
    if keep:
        fig.legend(
            [h for h, _ in keep],
            [l for _, l in keep],
            title="Category",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(keep),
            fontsize=8,
            frameon=False,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_figure_data(output_path, pivot)
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
        r"\begin{table}[h]",
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
        r"\begin{table}[h]",
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
