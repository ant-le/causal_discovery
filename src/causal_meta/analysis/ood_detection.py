"""OOD detection and selective prediction analysis.

Treats the problem as binary classification (ID vs OOD) using posterior
uncertainty measures (``edge_entropy``, ``graph_nll``) as detection scores.
Computes AUROC / AUPRC per model and generates selective prediction
(accuracy-vs-coverage) Pareto frontiers.

All functions consume the per-task raw DataFrame produced by
:func:`~causal_meta.analysis.utils.load_raw_task_dataframe`.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────


def _is_ood(dataset_key: str) -> bool:
    """Return True if the dataset key is OOD (not in-distribution)."""
    dk = dataset_key.lower()
    return not (dk.startswith("id_") or dk == "id_test")


def _pivot_raw_wide(
    raw_df: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Pivot raw long-format to wide (one row per task) for selected metrics."""
    if raw_df.empty or "Metric" not in raw_df.columns:
        return pd.DataFrame()

    subset = raw_df[raw_df["Metric"].isin(metrics)].copy()
    if subset.empty:
        return pd.DataFrame()

    id_cols = [
        c
        for c in (
            "RunID",
            "Model",
            "ModelKey",
            "DatasetKey",
            "Dataset",
            "TaskIdx",
            "GraphType",
            "MechType",
            "NNodes",
            "SparsityParam",
            "SpectralDist",
            "KLDegreeDist",
        )
        if c in subset.columns
    ]

    wide = subset.pivot_table(
        index=id_cols,
        columns="Metric",
        values="Value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


# ── OOD Detection AUROC/AUPRC ──────────────────────────────────────────


def _roc_auc_manual(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC without sklearn dependency.

    Uses the Wilcoxon–Mann–Whitney statistic (equivalent to AUC).
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    # Count concordant pairs
    n_concordant = 0
    n_tied = 0
    for p in pos:
        n_concordant += int(np.sum(neg < p))
        n_tied += int(np.sum(neg == p))

    auc = (n_concordant + 0.5 * n_tied) / (len(pos) * len(neg))
    return float(auc)


def _precision_recall_auc_manual(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute area under the Precision-Recall curve without sklearn.

    Uses the trapezoidal rule on sorted thresholds.
    """
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")

    # Sort by descending score
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    tp_cumsum = np.cumsum(sorted_labels)
    n_predicted = np.arange(1, len(sorted_labels) + 1)
    precisions = tp_cumsum / n_predicted
    recalls = tp_cumsum / labels.sum()

    # Prepend (recall=0, precision=1) for area calculation
    precisions = np.concatenate([[1.0], precisions])
    recalls = np.concatenate([[0.0], recalls])

    # Trapezoidal AUC (np.trapezoid replaces np.trapz in NumPy 2.0+)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if _trapz is None:
        raise RuntimeError("No trapezoidal integration function found in numpy.")
    auprc = float(_trapz(precisions, recalls))
    return abs(auprc)  # sign depends on recall ordering


def compute_ood_detection_metrics(
    raw_df: pd.DataFrame,
    score_metric: str = "edge_entropy",
) -> pd.DataFrame:
    """Compute OOD detection AUROC and AUPRC per model.

    Args:
        raw_df: Long-format per-task DataFrame from ``load_raw_task_dataframe``.
        score_metric: Which metric to use as OOD detection score.
            Higher score should indicate more likely OOD.

    Returns:
        DataFrame with columns ``Model``, ``ScoreMetric``, ``AUROC``, ``AUPRC``,
        ``N_ID``, ``N_OOD``.
    """
    if raw_df.empty or "Metric" not in raw_df.columns:
        return pd.DataFrame()

    # We also need an accuracy metric (e-shd) to pair with the score
    wide = _pivot_raw_wide(raw_df, [score_metric])
    if wide.empty or score_metric not in wide.columns:
        log.warning("Score metric '%s' not found in raw data; skipping.", score_metric)
        return pd.DataFrame()

    if "DatasetKey" not in wide.columns or "Model" not in wide.columns:
        return pd.DataFrame()

    wide["is_ood"] = wide["DatasetKey"].apply(_is_ood).astype(int)

    rows: list[dict[str, object]] = []
    for model in sorted(wide["Model"].unique()):
        mdf = wide[wide["Model"] == model].dropna(subset=[score_metric])
        labels = mdf["is_ood"].to_numpy()
        scores = mdf[score_metric].to_numpy(dtype=float)

        n_id = int((labels == 0).sum())
        n_ood = int((labels == 1).sum())

        if n_id == 0 or n_ood == 0:
            continue

        auroc = _roc_auc_manual(labels, scores)
        auprc = _precision_recall_auc_manual(labels, scores)

        rows.append(
            {
                "Model": model,
                "ScoreMetric": score_metric,
                "AUROC": round(auroc, 4),
                "AUPRC": round(auprc, 4),
                "N_ID": n_id,
                "N_OOD": n_ood,
            }
        )

    return pd.DataFrame(rows)


# ── Selective Prediction ────────────────────────────────────────────────


def compute_selective_prediction(
    raw_df: pd.DataFrame,
    score_metric: str = "edge_entropy",
    accuracy_metric: str = "e-shd",
    n_thresholds: int = 50,
) -> pd.DataFrame:
    """Compute selective prediction curves: accuracy vs coverage at varying thresholds.

    Policy: reject predictions where ``score_metric > threshold``.

    Args:
        raw_df: Long-format per-task DataFrame.
        score_metric: Uncertainty metric (higher = less trustworthy).
        accuracy_metric: Performance metric to evaluate accepted predictions.
        n_thresholds: Number of threshold values to sweep.

    Returns:
        DataFrame with columns ``Model``, ``Threshold``, ``Coverage``,
        ``MeanAccuracy``, ``N_Accepted``.
    """
    if raw_df.empty or "Metric" not in raw_df.columns:
        return pd.DataFrame()

    wide = _pivot_raw_wide(raw_df, [score_metric, accuracy_metric])
    needed = {score_metric, accuracy_metric}
    if wide.empty or not needed.issubset(wide.columns):
        log.warning(
            "Missing metrics for selective prediction (%s); skipping.",
            needed - set(wide.columns) if not wide.empty else needed,
        )
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for model in sorted(wide["Model"].unique()):
        mdf = wide[wide["Model"] == model].dropna(
            subset=[score_metric, accuracy_metric]
        )
        if mdf.empty:
            continue

        scores = mdf[score_metric].to_numpy(dtype=float)
        accuracy = mdf[accuracy_metric].to_numpy(dtype=float)
        total = len(scores)

        lo, hi = float(scores.min()), float(scores.max())
        if lo == hi:
            thresholds = np.array([lo])
        else:
            thresholds = np.linspace(lo, hi, n_thresholds)

        for t in thresholds:
            mask = scores <= t
            n_accepted = int(mask.sum())
            if n_accepted == 0:
                continue
            mean_acc = float(accuracy[mask].mean())
            coverage = n_accepted / total

            rows.append(
                {
                    "Model": model,
                    "Threshold": round(float(t), 6),
                    "Coverage": round(coverage, 6),
                    "MeanAccuracy": round(mean_acc, 6),
                    "N_Accepted": n_accepted,
                }
            )

    return pd.DataFrame(rows)


# ── OOD Detection Table ────────────────────────────────────────────────


def generate_ood_detection_table(
    detection_df: pd.DataFrame,
    output_path: "str | __import__('pathlib').Path",
) -> None:
    """Write a LaTeX table summarising OOD detection AUROC/AUPRC per model.

    Args:
        detection_df: Output of :func:`compute_ood_detection_metrics`.
        output_path: Where to write the ``.tex`` file.
    """
    from pathlib import Path

    output_path = Path(output_path)

    if detection_df.empty:
        log.warning("Empty detection DataFrame; skipping OOD detection table.")
        return

    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{OOD detection performance using posterior uncertainty as a"
        r" discriminator. ID tasks are negative, OOD tasks are positive.}"
    )
    lines.append(r"\label{tab:ood_detection}")
    lines.append(r"\begin{tabular}{l c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{Score} & \textbf{AUROC} & \textbf{AUPRC}"
        r" & \textbf{$N_{\mathrm{ID}}$} & \textbf{$N_{\mathrm{OOD}}$} \\"
    )
    lines.append(r"\midrule")

    for _, row in detection_df.iterrows():
        model_tex = str(row["Model"]).replace("_", r"\_")
        lines.append(
            f"{model_tex} & {row['ScoreMetric']}"
            f" & {row['AUROC']:.3f} & {row['AUPRC']:.3f}"
            f" & {row['N_ID']} & {row['N_OOD']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    log.info("Saved OOD detection table to %s", output_path)
