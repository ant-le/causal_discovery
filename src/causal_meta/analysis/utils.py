from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


class EmptyOverviewError(RuntimeError):
    """Raised when an overview file exists but contains no usable rows."""


# -----------------------------------------------------------------------------
# Presentation mappings
# -----------------------------------------------------------------------------

# Run directory / overview.json keys -> display names
MODEL_NAME_MAP: dict[str, str] = {
    "avici": "AviCi",
    # Backward compat: older runs logged smoke/full-suffixed model ids.
    "avici_smoke": "AviCi",
    "avici_full": "AviCi",
    "bcnp": "BCNP",
    "bcnp_smoke": "BCNP",
    "bcnp_full": "BCNP",
    "dibs": "DiBS",
    "random": "Random",
    "random_smoke": "Random",
    "bayesdag": "BayesDAG",
}

# Dataset keys -> human-readable labels
DATASET_DESCRIPTION_MAP: dict[str, str] = {
    "id_test": "ID",
    "ood_logistic_map": "Logistic Map",
    "ood_periodic": "Periodic",
    "ood_pnl_tanh": "Post-Nonlinear (tanh)",
    "ood_square": "Square",
    "ood_sbm_strong": "SBM (strong)",
}


def map_model_name(raw_model: str) -> str:
    """Map a run/model key to a display name."""
    if raw_model in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[raw_model]
    # Common normalization: strip known suffixes
    for suffix in ("_smoke",):
        if raw_model.endswith(suffix):
            base = raw_model[: -len(suffix)]
            return MODEL_NAME_MAP.get(base, base)
    return raw_model


def map_dataset_description(raw_dataset: str) -> str:
    """Map a dataset key to a human-readable description."""
    return DATASET_DESCRIPTION_MAP.get(raw_dataset, raw_dataset)


def load_overview_json(file_path: str, *, translate_names: bool = True) -> pd.DataFrame:
    """
    Loads the overview.json file and converts it into a long-format DataFrame.

    Args:
        file_path: Path to the overview.json file.

    Returns:
        DataFrame with columns: ['Model', 'Dataset', 'Metric', 'Mean', 'SEM', 'Std']
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    rows: list[dict[str, object]] = []
    for model_key, datasets in data.items():
        model_name = map_model_name(model_key) if translate_names else model_key
        for dataset_key, metrics in datasets.items():
            dataset_name = (
                map_dataset_description(dataset_key) if translate_names else dataset_key
            )
            # Identify unique base metrics (removing _mean, _sem, _std suffixes)
            base_metrics = set()
            for key in metrics.keys():
                if key.endswith("_mean"):
                    base_metrics.add(key[:-5])
                elif key.endswith("_sem"):
                    base_metrics.add(key[:-4])
                elif key.endswith("_std"):
                    base_metrics.add(key[:-4])

            for metric in base_metrics:
                row = {
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "ModelKey": model_key,
                    "DatasetKey": dataset_key,
                    "Metric": metric,
                    "Mean": metrics.get(f"{metric}_mean", float("nan")),
                    "SEM": metrics.get(f"{metric}_sem", 0.0),
                    "Std": metrics.get(f"{metric}_std", 0.0),
                }
                rows.append(row)

    return pd.DataFrame(rows)


def generate_all_artifacts(
    overview_path: Path,
    output_dir: Path,
) -> None:
    """Load an overview JSON and generate all standard analysis artifacts.

    This is the shared implementation used by both the CLI entry point
    (``run_analysis.py``) and the Hydra task runner (``runners/tasks/analysis``).

    Args:
        overview_path: Path to the ``overview.json`` file.
        output_dir: Directory where generated figures and tables are written.

    Raises:
        FileNotFoundError: If *overview_path* does not exist.
        EmptyOverviewError: If the loaded DataFrame is empty.
    """
    from causal_meta.analysis.plots.results import (
        generate_performance_figure,
        generate_structural_figure,
    )
    from causal_meta.analysis.tables.results import generate_robustness_table

    if not overview_path.exists():
        raise FileNotFoundError(
            f"Missing overview file: {overview_path}. "
            "Generate overview.json first, then rerun this command."
        )

    log.info(f"Loading overview from {overview_path}")
    df = load_overview_json(str(overview_path))
    if df.empty:
        raise EmptyOverviewError(f"No data found in {overview_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Writing artifacts to {output_dir}")

    generate_structural_figure(df, output_dir / "structural_metrics.png")
    generate_performance_figure(df, output_dir / "performance_metrics.png")
    generate_robustness_table(df, output_dir / "robustness_table.tex")
