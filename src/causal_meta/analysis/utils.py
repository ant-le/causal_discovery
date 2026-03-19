from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

log = logging.getLogger(__name__)


class EmptyAnalysisDataError(RuntimeError):
    """Raised when selected runs contain no usable analysis rows."""


class RunSelectionError(ValueError):
    """Raised when run selection arguments are invalid."""


# Run/model keys -> display names
MODEL_NAME_MAP: dict[str, str] = {
    "avici": "AviCi",
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
    # ── ID: ER × {Linear, MLP, GP} ────────────────────────────────────
    "id_linear_er20": "ID Linear ER-20",
    "id_linear_er40": "ID Linear ER-40",
    "id_linear_er60": "ID Linear ER-60",
    "id_neuralnet_er20": "ID MLP ER-20",
    "id_neuralnet_er40": "ID MLP ER-40",
    "id_neuralnet_er60": "ID MLP ER-60",
    "id_gpcde_er20": "ID GP ER-20",
    "id_gpcde_er40": "ID GP ER-40",
    "id_gpcde_er60": "ID GP ER-60",
    # ── ID: SF × {Linear, MLP, GP} ────────────────────────────────────
    "id_linear_sf1": "ID Linear SF-1",
    "id_linear_sf2": "ID Linear SF-2",
    "id_linear_sf3": "ID Linear SF-3",
    "id_neuralnet_sf1": "ID MLP SF-1",
    "id_neuralnet_sf2": "ID MLP SF-2",
    "id_neuralnet_sf3": "ID MLP SF-3",
    "id_gpcde_sf1": "ID GP SF-1",
    "id_gpcde_sf2": "ID GP SF-2",
    "id_gpcde_sf3": "ID GP SF-3",
    # ── OOD Graph: SBM × {Linear, MLP, GP} ────────────────────────────
    "ood_graph_sbm_linear": "OOD-G SBM Linear",
    "ood_graph_sbm_neuralnet": "OOD-G SBM MLP",
    "ood_graph_sbm_gpcde": "OOD-G SBM GP",
    # ── OOD Mechanism: ER-40 × {Periodic, Square, LogisticMap, PNL} ───
    "ood_mech_periodic_er40": "OOD-M Periodic",
    "ood_mech_square_er40": "OOD-M Square",
    "ood_mech_logistic_map_er40": "OOD-M Logistic Map",
    "ood_mech_pnl_tanh_er40": "OOD-M PNL (tanh)",
    # ── OOD Both: SBM × Periodic ──────────────────────────────────────
    "ood_both_sbm_periodic": "OOD-Both SBM Periodic",
}


def map_model_name(raw_model: str) -> str:
    """Map a model key to a display name."""
    if raw_model in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[raw_model]
    for suffix in ("_smoke",):
        if raw_model.endswith(suffix):
            base = raw_model[: -len(suffix)]
            return MODEL_NAME_MAP.get(base, base)
    return raw_model


def map_dataset_description(raw_dataset: str) -> str:
    """Map a dataset key to a human-readable description."""
    return DATASET_DESCRIPTION_MAP.get(raw_dataset, raw_dataset)


def resolve_run_directories(
    *,
    runs_root: Path | None,
    run_ids: Sequence[str] | None = None,
    run_dirs: Sequence[Path] | None = None,
) -> list[Path]:
    """Resolve run directories from explicit IDs/paths or by discovery.

    Args:
        runs_root: Root directory under which run IDs are resolved.
        run_ids: Optional run IDs (directory names under ``runs_root``).
        run_dirs: Optional explicit run directories.

    Returns:
        A de-duplicated list of absolute run directories containing ``metrics.json``.

    Raises:
        RunSelectionError: If selection arguments are inconsistent.
        FileNotFoundError: If no valid runs are found.
    """
    selected_dirs: list[Path] = []

    if run_dirs:
        selected_dirs.extend(Path(run_dir) for run_dir in run_dirs)

    if run_ids:
        if runs_root is None:
            raise RunSelectionError("runs_root is required when run_ids are provided.")
        selected_dirs.extend(Path(runs_root) / str(run_id) for run_id in run_ids)

    if not selected_dirs:
        if runs_root is None:
            raise RunSelectionError(
                "Provide run_ids/run_dirs, or provide runs_root for automatic discovery."
            )
        selected_dirs = sorted(
            path.parent for path in Path(runs_root).rglob("metrics.json")
        )

    if not selected_dirs:
        raise FileNotFoundError("No run directories with metrics.json were found.")

    resolved: list[Path] = []
    seen: set[str] = set()
    for run_dir in selected_dirs:
        absolute_run_dir = Path(run_dir).expanduser().resolve()
        metrics_path = absolute_run_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Missing metrics.json for run '{absolute_run_dir}'. "
                f"Expected file: {metrics_path}"
            )
        key = str(absolute_run_dir)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(absolute_run_dir)

    return resolved


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_base_metrics(metrics: Mapping[str, Any]) -> set[str]:
    base_metrics: set[str] = set()
    for key, value in metrics.items():
        if key.endswith("_mean"):
            base_metrics.add(key[:-5])
        elif key.endswith("_sem"):
            base_metrics.add(key[:-4])
        elif key.endswith("_std"):
            base_metrics.add(key[:-4])
        elif not key.startswith("_") and isinstance(value, (int, float)):
            base_metrics.add(str(key))
    return base_metrics


def _infer_model_key(run_id: str) -> str:
    run_id_norm = run_id.lower()
    candidates = sorted(MODEL_NAME_MAP.keys(), key=len, reverse=True)
    for candidate in candidates:
        candidate_norm = candidate.lower()
        if (
            run_id_norm == candidate_norm
            or run_id_norm.endswith(f"_{candidate_norm}")
            or f"_{candidate_norm}_" in run_id_norm
        ):
            return candidate
    return run_id


def load_runs_dataframe(
    run_dirs: Sequence[Path], *, translate_names: bool = True
) -> pd.DataFrame:
    """Load selected run metrics into a normalized long-format DataFrame.

    Args:
        run_dirs: Run directories containing ``metrics.json``.
        translate_names: Whether to map model/dataset keys to display names.

    Returns:
        DataFrame with rows per (run, dataset, metric).
    """
    rows: list[dict[str, object]] = []

    for run_dir in run_dirs:
        metrics_path = Path(run_dir) / "metrics.json"
        with open(metrics_path, "r") as f:
            payload = json.load(f)

        if not isinstance(payload, Mapping):
            log.warning("Skipping malformed metrics payload at %s", metrics_path)
            continue

        metadata = _as_mapping(payload.get("metadata"))
        run_id = str(metadata.get("run_id", Path(run_dir).name))
        run_name = str(metadata.get("run_name", run_id))

        model_key_raw = metadata.get("model_name")
        model_key = (
            str(model_key_raw)
            if model_key_raw is not None and str(model_key_raw).strip()
            else _infer_model_key(run_id)
        )
        model_name = map_model_name(model_key) if translate_names else model_key

        summary = _as_mapping(payload.get("summary"))
        fam_meta = _as_mapping(payload.get("family_metadata"))
        distances = _as_mapping(payload.get("distances"))

        for dataset_key, dataset_metrics_any in summary.items():
            dataset_key_str = str(dataset_key)
            dataset_metrics = _as_mapping(dataset_metrics_any)
            if not dataset_metrics:
                continue

            dataset_name = (
                map_dataset_description(dataset_key_str)
                if translate_names
                else dataset_key_str
            )

            # Enrich with family metadata (Phase C)
            ds_fam = _as_mapping(fam_meta.get(dataset_key_str))
            graph_type = str(ds_fam.get("graph_type", "")) if ds_fam else ""
            mech_type = str(ds_fam.get("mech_type", "")) if ds_fam else ""
            n_nodes = ds_fam.get("n_nodes")
            sparsity_param = ds_fam.get("sparsity_param")

            # Enrich with distributional distances (Phase D)
            ds_dist = _as_mapping(distances.get(dataset_key_str))
            spectral_dist = _to_float(ds_dist.get("spectral"), float("nan"))
            kl_degree_dist = _to_float(ds_dist.get("kl_degree"), float("nan"))

            for metric in sorted(_extract_base_metrics(dataset_metrics)):
                mean_raw = dataset_metrics.get(f"{metric}_mean")
                sem_raw = dataset_metrics.get(f"{metric}_sem", 0.0)
                std_raw = dataset_metrics.get(f"{metric}_std", 0.0)
                if mean_raw is None:
                    mean_raw = dataset_metrics.get(metric, float("nan"))

                rows.append(
                    {
                        "RunID": run_id,
                        "RunName": run_name,
                        "RunDir": str(run_dir),
                        "Model": model_name,
                        "Dataset": dataset_name,
                        "ModelKey": model_key,
                        "DatasetKey": dataset_key_str,
                        "Metric": metric,
                        "Mean": _to_float(mean_raw, float("nan")),
                        "SEM": _to_float(sem_raw, 0.0),
                        "Std": _to_float(std_raw, 0.0),
                        "GraphType": graph_type,
                        "MechType": mech_type,
                        "NNodes": int(n_nodes) if n_nodes is not None else None,
                        "SparsityParam": (
                            float(sparsity_param)
                            if sparsity_param is not None
                            else None
                        ),
                        "SpectralDist": spectral_dist,
                        "KLDegreeDist": kl_degree_dist,
                    }
                )

    return pd.DataFrame(rows)


def load_raw_task_dataframe(
    run_dirs: Sequence[Path],
    metrics: Sequence[str] | None = None,
    *,
    translate_names: bool = True,
) -> pd.DataFrame:
    """Load per-task raw metric values into a long-format DataFrame.

    Unlike :func:`load_runs_task_dataframe`, this reads the ``"raw"`` block from
    each ``metrics.json`` and returns **one row per (run, dataset, task, metric)**.

    Args:
        run_dirs: Run directories containing ``metrics.json``.
        metrics: If given, only load these metric keys.  ``None`` loads all.
        translate_names: Whether to map model/dataset keys to display names.

    Returns:
        DataFrame with columns ``RunID``, ``Model``, ``ModelKey``, ``DatasetKey``,
        ``Dataset``, ``TaskIdx``, ``Metric``, ``Value``, plus enrichment columns
        ``GraphType``, ``MechType``, ``NNodes``, ``SparsityParam``,
        ``SpectralDist``, ``KLDegreeDist``.
    """
    rows: list[dict[str, object]] = []

    for run_dir in run_dirs:
        metrics_path = Path(run_dir) / "metrics.json"
        with open(metrics_path, "r") as f:
            payload = json.load(f)

        if not isinstance(payload, Mapping):
            continue

        metadata = _as_mapping(payload.get("metadata"))
        run_id = str(metadata.get("run_id", Path(run_dir).name))

        model_key_raw = metadata.get("model_name")
        model_key = (
            str(model_key_raw)
            if model_key_raw is not None and str(model_key_raw).strip()
            else _infer_model_key(run_id)
        )
        model_name = map_model_name(model_key) if translate_names else model_key

        raw = _as_mapping(payload.get("raw"))
        fam_meta = _as_mapping(payload.get("family_metadata"))
        distances = _as_mapping(payload.get("distances"))

        for dataset_key, raw_metrics_any in raw.items():
            dataset_key_str = str(dataset_key)
            raw_metrics = _as_mapping(raw_metrics_any)
            if not raw_metrics:
                continue

            dataset_name = (
                map_dataset_description(dataset_key_str)
                if translate_names
                else dataset_key_str
            )

            # Enrichment (Phase C/D)
            ds_fam = _as_mapping(fam_meta.get(dataset_key_str))
            graph_type = str(ds_fam.get("graph_type", "")) if ds_fam else ""
            mech_type = str(ds_fam.get("mech_type", "")) if ds_fam else ""
            n_nodes = ds_fam.get("n_nodes")
            sparsity_param = ds_fam.get("sparsity_param")

            ds_dist = _as_mapping(distances.get(dataset_key_str))
            spectral_dist = _to_float(ds_dist.get("spectral"), float("nan"))
            kl_degree_dist = _to_float(ds_dist.get("kl_degree"), float("nan"))

            for metric_name, values_any in raw_metrics.items():
                if metrics is not None and metric_name not in metrics:
                    continue
                if not isinstance(values_any, list):
                    continue
                # Skip prefixed duplicates (e.g. "dataset_key/metric")
                if "/" in metric_name:
                    continue

                for task_idx, val in enumerate(values_any):
                    rows.append(
                        {
                            "RunID": run_id,
                            "Model": model_name,
                            "ModelKey": model_key,
                            "DatasetKey": dataset_key_str,
                            "Dataset": dataset_name,
                            "TaskIdx": task_idx,
                            "Metric": metric_name,
                            "Value": _to_float(val, float("nan")),
                            "GraphType": graph_type,
                            "MechType": mech_type,
                            "NNodes": (int(n_nodes) if n_nodes is not None else None),
                            "SparsityParam": (
                                float(sparsity_param)
                                if sparsity_param is not None
                                else None
                            ),
                            "SpectralDist": spectral_dist,
                            "KLDegreeDist": kl_degree_dist,
                        }
                    )

    return pd.DataFrame(rows)


def generate_all_artifacts_from_runs(
    run_dirs: Sequence[Path],
    output_dir: Path,
) -> pd.DataFrame:
    """Generate all standard plots/tables from selected run directories.

    Args:
        run_dirs: Selected run directories containing ``metrics.json``.
        output_dir: Directory where figures and tables are written.

    Returns:
        The normalized DataFrame used to create outputs.

    Raises:
        EmptyAnalysisDataError: If selected runs contain no usable rows.
    """
    from causal_meta.analysis.plots.results import (
        generate_calibration_scatter,
        generate_density_stratified_figure,
        generate_distance_degradation_scatter,
        generate_entropy_histogram,
        generate_event_probability_bar,
        generate_failure_mode_bar,
        generate_performance_figure,
        generate_posterior_diagnostic_violins,
        generate_selective_prediction_pareto,
        generate_structural_figure,
    )
    from causal_meta.analysis.tables.results import (
        generate_distance_regression_table,
        generate_robustness_table,
    )
    from causal_meta.analysis.failure_modes import (
        classify_failure_modes,
        failure_mode_fractions,
    )
    from causal_meta.analysis.ood_detection import (
        compute_ood_detection_metrics,
        compute_selective_prediction,
        generate_ood_detection_table,
    )
    from causal_meta.analysis.posterior_diagnostics import (
        run_posterior_diagnostics_from_runs,
    )

    df = load_runs_dataframe(run_dirs)
    if df.empty:
        raise EmptyAnalysisDataError("No analysis rows found in selected run metrics.")

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing artifacts to %s", output_dir)

    # Original figures
    generate_structural_figure(df, output_dir / "structural_metrics.png")
    generate_performance_figure(df, output_dir / "performance_metrics.png")
    generate_robustness_table(df, output_dir / "robustness_table.tex")

    # Phase E additions
    generate_calibration_scatter(df, output_dir / "calibration_scatter.png")
    generate_distance_degradation_scatter(df, output_dir / "distance_degradation.png")
    generate_density_stratified_figure(df, output_dir / "density_stratified.png")
    generate_distance_regression_table(df, output_dir / "distance_regression.tex")

    # Phase F: Failure mode analysis (needs raw per-task data)
    raw_df = load_raw_task_dataframe(
        run_dirs,
        metrics=["sparsity_ratio", "skeleton_f1", "orientation_accuracy"],
    )
    if not raw_df.empty:
        classified = classify_failure_modes(raw_df)
        if not classified.empty:
            fractions = failure_mode_fractions(classified)
            generate_failure_mode_bar(fractions, output_dir / "failure_modes.png")

    # Phase G: OOD detection analysis (needs raw per-task data)
    ood_raw_df = load_raw_task_dataframe(
        run_dirs,
        metrics=["edge_entropy", "graph_nll", "e-shd"],
    )
    if not ood_raw_df.empty:
        # Entropy histogram (E.3a)
        generate_entropy_histogram(ood_raw_df, output_dir / "entropy_histogram.png")

        # OOD detection AUROC/AUPRC table
        for score in ("edge_entropy", "graph_nll"):
            detection_df = compute_ood_detection_metrics(ood_raw_df, score_metric=score)
            if not detection_df.empty:
                generate_ood_detection_table(
                    detection_df,
                    output_dir / f"ood_detection_{score}.tex",
                )

        # Selective prediction Pareto curve (E.3b)
        pareto_df = compute_selective_prediction(ood_raw_df)
        generate_selective_prediction_pareto(
            pareto_df, output_dir / "selective_prediction.png"
        )

    # Phase F (updated): Posterior failure diagnostics from .pt artifacts
    try:
        posterior_df = run_posterior_diagnostics_from_runs(run_dirs)
        if not posterior_df.empty:
            generate_event_probability_bar(
                posterior_df, output_dir / "event_probabilities.png"
            )
            generate_posterior_diagnostic_violins(
                posterior_df, output_dir / "posterior_diagnostics.png"
            )
            log.info("Generated posterior diagnostics from %d tasks", len(posterior_df))
        else:
            log.info("No inference artifacts found; skipping posterior diagnostics.")
    except Exception:
        log.warning(
            "Posterior diagnostics failed; continuing with remaining artifacts.",
            exc_info=True,
        )

    return df
