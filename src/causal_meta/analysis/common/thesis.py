from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from causal_meta.analysis.utils import (
    EmptyAnalysisDataError,
    PAPER_MODEL_LABELS,
    load_raw_task_dataframe,
    load_runs_dataframe,
    map_dataset_description,
)

EXPECTED_MODEL_DIRS: tuple[str, ...] = tuple(PAPER_MODEL_LABELS.keys())
OUTPUT_SUBDIRS: tuple[str, ...] = (
    "figures",
    "tables",
    "data",
    "snippets",
    "appendix",
    "provenance",
)
REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIGS_ROOT = REPO_ROOT / "src" / "causal_meta" / "configs"


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


def paper_model_label(model_key: str) -> str:
    model_key_norm = model_key.lower()
    for key, label in PAPER_MODEL_LABELS.items():
        if model_key_norm == key or model_key_norm.startswith(f"{key}_"):
            return label
    return model_key


def axis_category(dataset_key: str) -> str:
    dataset_key_norm = dataset_key.lower()
    if dataset_key_norm.startswith("id_") or dataset_key_norm == "id_test":
        return "id"
    if dataset_key_norm.startswith("ood_graph_"):
        return "graph"
    if dataset_key_norm.startswith("ood_mech_"):
        return "mechanism"
    if dataset_key_norm.startswith("ood_noise_"):
        return "noise"
    if dataset_key_norm.startswith("ood_both_"):
        return "compound"
    if dataset_key_norm.startswith("ood_nodes_"):
        return "nodes"
    if dataset_key_norm.startswith("ood_samples_"):
        return "samples"
    return "other"


# ── Mechanism and graph-family extraction ──────────────────────────────

ID_MECHANISM_LABELS: dict[str, str] = {
    "linear": "Linear",
    "neuralnet": "MLP",
    "gpcde": "GP",
}
"""Display names for the three in-distribution mechanism families."""


def id_mechanism_of(dataset_key: str) -> str | None:
    """Return the in-distribution mechanism family, or *None* for OOD-mechanism datasets.

    For datasets whose functional mechanism is one of the three ID families
    (``linear``, ``neuralnet``, ``gpcde``) this returns the canonical key.
    For OOD-mechanism and compound-shift datasets the mechanism is
    out-of-distribution and the function returns ``None``.
    """
    dk = dataset_key.lower()
    body = re.sub(r"_d\d+_n\d+$", "", dk)
    # OOD-mechanism and compound datasets have no ID mechanism
    if body.startswith("ood_mech_") or body.startswith("ood_both_"):
        return None
    for mech in ("neuralnet", "gpcde", "linear"):  # longest first
        if f"_{mech}" in body:
            return mech
    return None


def graph_family_of(dataset_key: str) -> str | None:
    """Return the broad graph topology family (``'er'``, ``'sf'``, ``'sbm'``, ``'ws'``, ``'grg'``)."""
    dk = dataset_key.lower()
    body = re.sub(r"_d\d+_n\d+$", "", dk)
    for graph in ("sbm", "ws", "grg"):
        if f"_{graph}" in body:
            return graph
    if re.search(r"_er\d+", body):
        return "er"
    if re.search(r"_sf\d+", body):
        return "sf"
    return None


def graph_code_of(dataset_key: str) -> str | None:
    """Return the specific graph code (e.g. ``'er20'``, ``'sf2'``, ``'sbm'``) from a dataset key.

    Unlike :func:`graph_family_of`, this returns the full code including
    any numeric suffix (``er20`` instead of ``er``).
    """
    dk = dataset_key.lower()
    body = re.sub(r"_d\d+_n\d+$", "", dk)
    # OOD graphs (no numeric suffix)
    for graph in ("sbm", "ws", "grg"):
        if f"_{graph}" in body:
            return graph
    # ER/SF with numeric suffix
    m = re.search(r"_(er\d+|sf\d+)", body)
    return m.group(1) if m else None


def mech_shift_graph_anchor(dataset_key: str) -> str | None:
    """Return the graph anchor code for a mechanism-shift dataset.

    For keys like ``ood_mech_periodic_er20_d20_n500`` returns ``'er20'``.
    For non-mechanism-shift datasets, returns ``None``.
    """
    dk = dataset_key.lower()
    if not dk.startswith("ood_mech_"):
        return None
    return graph_code_of(dk)


def noise_shift_anchor(dataset_key: str) -> tuple[str, str] | None:
    """Return the ``(mechanism, graph_code)`` anchor for a noise-shift dataset.

    For keys like ``ood_noise_laplace_linear_er20_d20_n500`` returns
    ``('linear', 'er20')``.  For non-noise-shift datasets, returns ``None``.
    """
    dk = dataset_key.lower()
    if not dk.startswith("ood_noise_"):
        return None
    mech = id_mechanism_of(dk)
    graph = graph_code_of(dk)
    if mech is None or graph is None:
        return None
    return (mech, graph)


def transfer_anchor(dataset_key: str) -> tuple[str, str] | None:
    """Return the ``(mechanism, graph_code)`` anchor for a node/sample transfer dataset.

    For keys like ``ood_nodes_linear_er20_d40_n500`` returns ``('linear', 'er20')``.
    For non-transfer datasets, returns ``None``.
    """
    dk = dataset_key.lower()
    if not (dk.startswith("ood_nodes_") or dk.startswith("ood_samples_")):
        return None
    mech = id_mechanism_of(dk)
    graph = graph_code_of(dk)
    if mech is None or graph is None:
        return None
    return (mech, graph)


# ── Anchor display labels ─────────────────────────────────────────────

GRAPH_ANCHOR_LABELS: dict[str, str] = {
    "er20": "ER-20",
    "er40": "ER-40",
    "er60": "ER-60",
    "sf1": "SF-1",
    "sf2": "SF-2",
    "sf3": "SF-3",
    "sbm": "SBM",
    "ws": "WS",
    "grg": "GRG",
}
"""Display names for graph anchor codes."""

TRANSFER_ANCHOR_LABELS: dict[tuple[str, str], str] = {
    ("linear", "er20"): "ER-20 × Linear",
    ("neuralnet", "sf2"): "SF-2 × MLP",
}
"""Display names for the two transfer ladder anchors."""


def is_fixed_size_task_frame(df: pd.DataFrame) -> pd.Series:
    return df["NNodes"].eq(20) & df["SamplesPerTask"].eq(500)


def thesis_dataset_label(dataset_key: str, dataset_label: str) -> str:
    label = dataset_label
    # Longest prefixes first so "OOD-Graph " is tried before "OOD-G ".
    for prefix in (
        "OOD-Graph ",
        "OOD-Mech ",
        "OOD-Noise ",
        "OOD-Both ",
        "OOD-G ",
        "OOD-M ",
        "OOD-N ",
        "ID ",
    ):
        if label.startswith(prefix):
            label = label[len(prefix) :]
            break
    label = re.sub(r"\s*\(d=\d+, n=\d+\)$", "", label)
    if dataset_key.startswith("ood_nodes_"):
        match = re.search(r"_d(\d+)_n\d+$", dataset_key)
        if match is not None:
            return match.group(1)
    if dataset_key.startswith("ood_samples_"):
        match = re.search(r"_d\d+_n(\d+)$", dataset_key)
        if match is not None:
            return match.group(1)
    return label


def metric_sem(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    sem = float(values.sem(ddof=1))
    return 0.0 if not pd.notna(sem) else sem


def format_value(mean: float, sem: float) -> str:
    return rf"${mean:.3f} \pm {sem:.3f}$"


def read_metrics_payload(run_dir: Path) -> Mapping[str, Any]:
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
    selected: list[SelectedRun] = []
    for model_dir in EXPECTED_MODEL_DIRS:
        run_dir = input_root / model_dir
        if not run_dir.exists():
            raise ThesisRunSelectionError(
                f"Missing curated model directory '{run_dir}'. Expected one folder per "
                "model under experiments/thesis_runs/."
            )
        payload = read_metrics_payload(run_dir)
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


def prepare_generated_workspace(thesis_root: Path) -> Path:
    thesis_root.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="generated_tmp_", dir=thesis_root))
    for subdir in OUTPUT_SUBDIRS:
        (temp_root / subdir).mkdir(parents=True, exist_ok=True)
    return temp_root


def finalize_generated_workspace(temp_root: Path, thesis_root: Path) -> Path:
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


def write_json(output_path: Path, payload: Mapping[str, Any]) -> None:
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def escape_tex(value: str) -> str:
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


def write_results_macros(
    selected_runs: Sequence[SelectedRun], output_path: Path
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "% Auto-generated by run_thesis_analysis.py. Do not edit by hand.",
        rf"\providecommand{{\ResultsGeneratedAt}}{{{timestamp}}}",
    ]
    for run in selected_runs:
        macro_name = "ResultsRunId" + run.model_dir.capitalize()
        lines.append(rf"\providecommand{{\{macro_name}}}{{{escape_tex(run.run_id)}}}")
    output_path.write_text("\n".join(lines) + "\n")


def prepare_summary_dataframe(run_dirs: Sequence[Path]) -> pd.DataFrame:
    summary_df = load_runs_dataframe(run_dirs, translate_names=False)
    if summary_df.empty:
        raise EmptyAnalysisDataError("No summary metrics found in curated thesis runs.")
    summary_df = summary_df.copy()
    summary_df["Model"] = summary_df["ModelKey"].map(paper_model_label)
    summary_df["Dataset"] = summary_df["DatasetKey"].map(map_dataset_description)
    summary_df["Dataset"] = summary_df.apply(
        lambda row: thesis_dataset_label(str(row["DatasetKey"]), str(row["Dataset"])),
        axis=1,
    )
    summary_df["AxisCategory"] = summary_df["DatasetKey"].map(axis_category)
    return summary_df


def prepare_raw_dataframe(run_dirs: Sequence[Path]) -> pd.DataFrame:
    metrics = [
        "e-sid",
        "e-shd",
        "e-edgef1",
        "ne-sid",
        "ne-shd",
        "graph_nll_per_edge",
        "edge_entropy",
        "ece",
        "inference_time_s",
        "sparsity_ratio",
        "skeleton_f1",
        "orientation_accuracy",
        "valid_dag_pct",
        "threshold_valid_dag_pct",
        "fp_count",
        "fn_count",
        "reversed_count",
        "correct_count",
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
    raw_df["Model"] = raw_df["ModelKey"].map(paper_model_label)
    raw_df["Dataset"] = raw_df["DatasetKey"].map(map_dataset_description)
    raw_df["Dataset"] = raw_df.apply(
        lambda row: thesis_dataset_label(str(row["DatasetKey"]), str(row["Dataset"])),
        axis=1,
    )
    raw_df["AxisCategory"] = raw_df["DatasetKey"].map(axis_category)
    return raw_df


def write_selected_runs(
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
    write_json(output_path, payload)
