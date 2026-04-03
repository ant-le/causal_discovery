from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_meta.analysis.common.thesis import (
    format_value,
    is_fixed_size_task_frame,
    metric_sem,
    thesis_dataset_label,
)
from causal_meta.analysis.utils import MODEL_COLORS, MODEL_MARKERS, PAPER_MODEL_LABELS


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
                "threshold_valid_dag_pct",
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
        ("threshold_valid_dag_pct", True),
        ("inference_time_s", False),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Speed--robustness anchor on the in-distribution families. Values report task-level means and standard errors aggregated over all in-distribution tasks.}",
        r"\label{tab:results_anchor}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Normalized $\mathbb{E}$-SID $\downarrow$} & \textbf{Normalized $\mathbb{E}$-SHD $\downarrow$} & \textbf{$\mathbb{E}$-Edge F1 $\uparrow$} & \textbf{Valid DAG Samples (\%) $\uparrow$} & \textbf{Thresholded DAG (\%) $\uparrow$} & \textbf{Runtime / dataset $\downarrow$} \\",
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
    ax.set_title(axis_title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Model", fontsize=9, loc="best")

    id_count = sum(1 for ds in datasets if axis_lookup.get(ds) == "id")
    if 0 < id_count < len(datasets):
        ax.axvspan(-0.5, id_count - 0.5, color="#f2f2f2", alpha=0.35, zorder=0)
        ax.axvline(id_count - 0.5, color="#999999", linestyle=":", linewidth=1.0)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return agg
