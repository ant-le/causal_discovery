from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def load_results(root_dir: str) -> pd.DataFrame:
    """Load per-model result files into a flat DataFrame.

    Args:
        root_dir: Root directory containing run outputs.

    Returns:
        DataFrame with flattened metrics and metadata inferred from path.
    """
    root_path = Path(root_dir)
    result_files = glob.glob(
        str(root_path / "**" / "results" / "*.json"), recursive=True
    )

    data = []

    for file_path in result_files:
        if Path(file_path).name == "aggregated.json":
            continue
        try:
            with open(file_path, "r") as f:
                content = json.load(f)

            summary = content.get("summary", content)
            model_name = Path(file_path).stem

            # Infer run ID from path (best-effort)
            path_parts = Path(file_path).parts

            # Simple heuristic: try to find the numeric folder (job id)
            run_id = "unknown"
            for part in reversed(path_parts):
                if part.isdigit():
                    run_id = part
                    break

            for dataset_name, metrics in summary.items():
                row: dict[str, Any] = {
                    "run_id": run_id,
                    "model": model_name,
                    "dataset": dataset_name,
                    "path": str(Path(file_path).parent),
                }
                # Flatten metrics
                for metric_key, metric_value in metrics.items():
                    row[metric_key] = metric_value
                data.append(row)

        except Exception as e:
            log.error(f"Failed to load {file_path}: {e}")

    return pd.DataFrame(data)


def aggregate_results(
    df: pd.DataFrame, group_by: list[str] = ["dataset"]
) -> pd.DataFrame:
    """
    Aggregates results by dataset.
    Computes Mean and SEM for all numeric columns.
    """
    if df.empty:
        return df

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude run_id if it happened to be numeric
    numeric_cols = [c for c in numeric_cols if c not in group_by and c != "run_id"]

    # Group
    grouped = df.groupby(group_by)[numeric_cols]

    means = grouped.mean()

    def calc_sem(series: pd.Series) -> float:
        return series.std(ddof=1) / np.sqrt(len(series)) if len(series) > 1 else 0.0

    # Calculate SEM manually to ensure DataFrame structure
    sems = grouped.agg(calc_sem)

    # Rename columns
    means.columns = [f"{c}_mean" for c in means.columns]
    sems.columns = [f"{c}_sem" for c in sems.columns]

    result = pd.concat([means, sems], axis=1)

    # Add count
    counts = grouped.size().to_frame(name="count")
    result = pd.concat([result, counts], axis=1)

    return result


def latex_table(
    agg_df: pd.DataFrame, metrics: list[str] = ["e-shd", "e-edgef1", "inil"]
) -> str:
    """
    Generates a simple LaTeX table body from the aggregated DataFrame.
    """
    lines = []

    # Header
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{l" + "c" * len(metrics) + "}")
    lines.append(r"\toprule")
    header = (
        "Dataset & "
        + " & ".join([m.upper().replace("_", "-") for m in metrics])
        + " \\"
    )
    lines.append(header)
    lines.append(r"\midrule")

    for idx, row in agg_df.iterrows():
        dataset_name = str(idx).replace("_", r"\_")

        row_str = f"{dataset_name}"

        for metric in metrics:
            mean_key = (
                f"{metric}_mean" if f"{metric}_mean" in row else f"{metric}_mean"
            )  # Check suffix
            sem_key = f"{metric}_sem"

            # Handle mismatch in keys if metrics don't exactly match column names (e.g. e-shd vs e_shd)
            # Try converting hyphens to underscores
            alt_mean = metric.replace("-", "_") + "_mean"
            alt_sem = metric.replace("-", "_") + "_sem"

            val_mean = row.get(mean_key, row.get(alt_mean, np.nan))
            val_sem = row.get(sem_key, row.get(alt_sem, 0.0))

            if pd.isna(val_mean):
                row_str += " & N/A"
            else:
                row_str += f" & ${val_mean:.3f} \pm {val_sem:.3f}$"

        row_str += " \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Aggregated Results}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing experiment results",
    )
    parser.add_argument(
        "--output", type=str, default="results_summary.tex", help="Output LaTeX file"
    )
    args = parser.parse_args()

    print(f"Loading results from {args.root}...")
    df = load_results(args.root)

    if df.empty:
        print("No results found.")
    else:
        print(f"Found {len(df)} runs.")
        agg = aggregate_results(df)
        print(agg)

        tex = latex_table(agg)

        with open(args.output, "w") as f:
            f.write(tex)
        print(f"LaTeX table saved to {args.output}")
