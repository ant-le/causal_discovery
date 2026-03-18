from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def generate_robustness_table(df: pd.DataFrame, output_path: Path) -> None:
    """
    Generates a LaTeX table comparing Models across Datasets for e-SHD and e-SID.
    """
    # Filter for relevant metrics
    metrics = ["e-shd", "e-sid"]
    subset = df[df["Metric"].isin(metrics)].copy()

    # Clean dataset names for LaTeX
    subset["Dataset"] = subset["Dataset"].str.replace("_", r"\_")
    subset["Model"] = subset["Model"].str.replace("_", r"\_")

    models = sorted(subset["Model"].unique())
    datasets = sorted(subset["Dataset"].unique())

    # Start building LaTeX
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Robustness Analysis: Structural metrics ($\mathbb{E}$-SHD and $\mathbb{E}$-SID) across in-distribution and OOD datasets. Lower is better.}"
    )
    lines.append(r"\label{tab:robustness}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    # Column definition: Dataset + (2 metrics * N models)
    # We group by Metric -> Model
    # Columns: Dataset | SHD (Model1...ModelN) | SID (Model1...ModelN)
    col_def = "l" + ("c" * len(models)) + "|" + ("c" * len(models))
    lines.append(r"\begin{tabular}{" + col_def + "}")
    lines.append(r"\toprule")

    # Header Row 1: Metrics
    header1 = r"\multirow{2}{*}{\textbf{Dataset}}"
    header1 += (
        r" & \multicolumn{"
        + str(len(models))
        + r"}{c}{\textbf{$\mathbb{E}$-SHD} $\downarrow$}"
    )
    header1 += (
        r" & \multicolumn{"
        + str(len(models))
        + r"}{c}{\textbf{$\mathbb{E}$-SID} $\downarrow$}"
    )
    header1 += r" \\"
    lines.append(header1)

    # Header Row 2: Models
    header2 = ""
    # For SHD
    header2 += r" & " + " & ".join(
        [r"\textbf{" + m.replace("_", r"\_") + "}" for m in models]
    )
    # For SID
    header2 += r" & " + " & ".join(
        [r"\textbf{" + m.replace("_", r"\_") + "}" for m in models]
    )
    header2 += r" \\"

    # Midrule spanning the metric groups
    # SHD columns: 2 to 1+len(models)
    # SID columns: 2+len(models) to 1+2*len(models)
    lines.append(
        r"\cmidrule(lr){2-"
        + str(1 + len(models))
        + r"} \cmidrule(lr){"
        + str(2 + len(models))
        + r"-"
        + str(1 + 2 * len(models))
        + r"}"
    )
    lines.append(header2)
    lines.append(r"\midrule")

    # Data Rows
    for ds in datasets:
        row_str = f"{ds}"

        # Collect values to find best (lowest) per metric per row
        shd_vals = []
        sid_vals = []

        # Pass 1: Get values to identify min
        for m in models:
            # SHD
            try:
                val = subset[
                    (subset["Dataset"] == ds)
                    & (subset["Model"] == m)
                    & (subset["Metric"] == "e-shd")
                ]["Mean"].iloc[0]
                shd_vals.append(val)
            except:
                shd_vals.append(float("inf"))

            # SID
            try:
                val = subset[
                    (subset["Dataset"] == ds)
                    & (subset["Model"] == m)
                    & (subset["Metric"] == "e-sid")
                ]["Mean"].iloc[0]
                sid_vals.append(val)
            except:
                sid_vals.append(float("inf"))

        min_shd = min(shd_vals)
        min_sid = min(sid_vals)

        # Pass 2: Build String
        # SHD Columns
        for i, m in enumerate(models):
            try:
                mean = subset[
                    (subset["Dataset"] == ds)
                    & (subset["Model"] == m)
                    & (subset["Metric"] == "e-shd")
                ]["Mean"].iloc[0]
                sem = subset[
                    (subset["Dataset"] == ds)
                    & (subset["Model"] == m)
                    & (subset["Metric"] == "e-shd")
                ]["SEM"].iloc[0]

                cell = f"${mean:.1f} \pm {sem:.1f}$"
                if abs(mean - min_shd) < 0.001:  # Best value
                    cell = r"\textbf{" + cell + "}"
                row_str += f" & {cell}"
            except:
                row_str += " & -"

        # SID Columns
        for i, m in enumerate(models):
            try:
                mean = subset[
                    (subset["Dataset"] == ds)
                    & (subset["Model"] == m)
                    & (subset["Metric"] == "e-sid")
                ]["Mean"].iloc[0]
                sem = subset[
                    (subset["Dataset"] == ds)
                    & (subset["Model"] == m)
                    & (subset["Metric"] == "e-sid")
                ]["SEM"].iloc[0]

                cell = f"${mean:.1f} \pm {sem:.1f}$"
                if abs(mean - min_sid) < 0.001:  # Best value
                    cell = r"\textbf{" + cell + "}"
                row_str += f" & {cell}"
            except:
                row_str += " & -"

        row_str += r" \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ── E.5  Multi-distance regression table (S5) ──────────────────────────


def _ood_category(dataset_key: str) -> str:
    dk = dataset_key.lower()
    if dk.startswith("id_") or dk == "id_test":
        return "ID"
    return "OOD"


def generate_distance_regression_table(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Regress E-SID degradation on (spectral distance, KL degree distance) via OLS.

    Produces a LaTeX table with R^2, coefficients, and p-values per model.
    Answers: *which type of distributional shift predicts degradation most strongly?*
    """
    # Require scipy for OLS
    try:
        from scipy import stats as sp_stats  # noqa: F811
    except ImportError:
        log.warning("scipy not available; skipping distance regression table.")
        return

    # Filter to E-SID metric only
    if df.empty or "Metric" not in df.columns:
        log.warning("Empty or column-less DataFrame for regression table; skipping.")
        return
    sid_df = df[df["Metric"] == "e-sid"].copy()
    if sid_df.empty:
        log.warning("No E-SID data for regression table; skipping.")
        return

    needed_cols = {"SpectralDist", "KLDegreeDist", "Model", "DatasetKey", "Mean"}
    if not needed_cols.issubset(sid_df.columns):
        log.warning("Missing enrichment columns for regression table; skipping.")
        return

    sid_df["OODCategory"] = sid_df["DatasetKey"].apply(_ood_category)

    # Compute per-model ID baseline
    id_means = (
        sid_df[sid_df["OODCategory"] == "ID"]
        .groupby("Model")["Mean"]
        .mean()
        .rename("id_baseline")
    )
    sid_df = sid_df.merge(id_means, on="Model", how="left")
    sid_df["degradation"] = sid_df["Mean"] - sid_df["id_baseline"]
    sid_df = sid_df.dropna(subset=["SpectralDist", "KLDegreeDist", "degradation"])

    models = sorted(sid_df["Model"].unique())
    if not models:
        log.warning("No valid models for regression; skipping.")
        return

    # Build LaTeX
    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{OLS regression of $\mathbb{E}$-SID degradation on distributional"
        r" distance measures. Degradation is computed relative to each model's mean"
        r" ID performance.}"
    )
    lines.append(r"\label{tab:distance_regression}")
    lines.append(r"\begin{tabular}{l c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{$R^2$} & \textbf{$\beta_{\mathrm{spectral}}$}"
        r" & \textbf{$p_{\mathrm{spectral}}$} & \textbf{$\beta_{\mathrm{KL}}$}"
        r" & \textbf{$p_{\mathrm{KL}}$} \\"
    )
    lines.append(r"\midrule")

    for model in models:
        mdf = sid_df[sid_df["Model"] == model]
        if len(mdf) < 3:
            lines.append(f"{model} & \\multicolumn{{5}}{{c}}{{insufficient data}} \\\\")
            continue

        y = mdf["degradation"].to_numpy(dtype=float)
        x_spectral = mdf["SpectralDist"].to_numpy(dtype=float)
        x_kl = mdf["KLDegreeDist"].to_numpy(dtype=float)

        # Simple bivariate OLS via normal equations
        X = np.column_stack([np.ones(len(y)), x_spectral, x_kl])
        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            lines.append(f"{model} & \\multicolumn{{5}}{{c}}{{singular}} \\\\")
            continue

        y_hat = X @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # p-values from t-test on coefficients
        n = len(y)
        k = X.shape[1]
        if n > k:
            mse = ss_res / (n - k)
            try:
                cov = mse * np.linalg.inv(X.T @ X)
                se = np.sqrt(np.diag(cov))
                t_vals = beta / se
                p_vals = [2.0 * float(sp_stats.t.sf(abs(t), df=n - k)) for t in t_vals]
            except np.linalg.LinAlgError:
                p_vals = [float("nan")] * k
        else:
            p_vals = [float("nan")] * k

        def _fmt_p(p: float) -> str:
            if np.isnan(p):
                return "--"
            if p < 0.001:
                return f"$<$0.001"
            return f"{p:.3f}"

        model_tex = model.replace("_", r"\_")
        lines.append(
            f"{model_tex} & {r2:.3f}"
            f" & {beta[1]:.2f} & {_fmt_p(p_vals[1])}"
            f" & {beta[2]:.2f} & {_fmt_p(p_vals[2])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    log.info("Saved distance regression table to %s", output_path)
