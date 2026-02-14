from __future__ import annotations

import logging
from pathlib import Path

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
    lines.append(r"\caption{Robustness Analysis: Structural metrics ($\mathbb{E}$-SHD and $\mathbb{E}$-SID) across in-distribution and OOD datasets. Lower is better.}")
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
    header1 += r" & \multicolumn{" + str(len(models)) + r"}{c}{\textbf{$\mathbb{E}$-SHD} $\downarrow$}"
    header1 += r" & \multicolumn{" + str(len(models)) + r"}{c}{\textbf{$\mathbb{E}$-SID} $\downarrow$}"
    header1 += r" \\"
    lines.append(header1)
    
    # Header Row 2: Models
    header2 = ""
    # For SHD
    header2 += r" & " + " & ".join([r"\textbf{" + m.replace("_", r"\_") + "}" for m in models])
    # For SID
    header2 += r" & " + " & ".join([r"\textbf{" + m.replace("_", r"\_") + "}" for m in models])
    header2 += r" \\"
    
    # Midrule spanning the metric groups
    # SHD columns: 2 to 1+len(models)
    # SID columns: 2+len(models) to 1+2*len(models)
    lines.append(r"\cmidrule(lr){2-" + str(1+len(models)) + r"} \cmidrule(lr){" + str(2+len(models)) + r"-" + str(1+2*len(models)) + r"}")
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
                val = subset[(subset["Dataset"] == ds) & (subset["Model"] == m) & (subset["Metric"] == "e-shd")]["Mean"].iloc[0]
                shd_vals.append(val)
            except:
                shd_vals.append(float('inf'))
                
            # SID
            try:
                val = subset[(subset["Dataset"] == ds) & (subset["Model"] == m) & (subset["Metric"] == "e-sid")]["Mean"].iloc[0]
                sid_vals.append(val)
            except:
                sid_vals.append(float('inf'))

        min_shd = min(shd_vals)
        min_sid = min(sid_vals)
        
        # Pass 2: Build String
        # SHD Columns
        for i, m in enumerate(models):
            try:
                mean = subset[(subset["Dataset"] == ds) & (subset["Model"] == m) & (subset["Metric"] == "e-shd")]["Mean"].iloc[0]
                sem = subset[(subset["Dataset"] == ds) & (subset["Model"] == m) & (subset["Metric"] == "e-shd")]["SEM"].iloc[0]
                
                cell = f"${mean:.1f} \pm {sem:.1f}$"
                if abs(mean - min_shd) < 0.001: # Best value
                    cell = r"\textbf{" + cell + "}"
                row_str += f" & {cell}"
            except:
                 row_str += " & -"

        # SID Columns
        for i, m in enumerate(models):
            try:
                mean = subset[(subset["Dataset"] == ds) & (subset["Model"] == m) & (subset["Metric"] == "e-sid")]["Mean"].iloc[0]
                sem = subset[(subset["Dataset"] == ds) & (subset["Model"] == m) & (subset["Metric"] == "e-sid")]["SEM"].iloc[0]
                
                cell = f"${mean:.1f} \pm {sem:.1f}$"
                if abs(mean - min_sid) < 0.001: # Best value
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
