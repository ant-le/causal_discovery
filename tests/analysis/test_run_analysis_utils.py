from __future__ import annotations

import json
from pathlib import Path

from causal_meta.analysis.utils import (
    generate_all_artifacts_from_runs,
    load_runs_dataframe,
    resolve_run_directories,
)


def _write_metrics(
    run_dir: Path,
    *,
    run_id: str,
    run_name: str,
    model_name: str,
    dataset_key: str,
    mean_offset: float,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "run_id": run_id,
            "run_name": run_name,
            "model_name": model_name,
            "output_dir": str(run_dir),
        },
        "summary": {
            dataset_key: {
                "e-shd_mean": 10.0 + mean_offset,
                "e-shd_sem": 1.0,
                "e-shd_std": 2.0,
                "e-sid_mean": 20.0 + mean_offset,
                "e-sid_sem": 2.0,
                "e-sid_std": 3.0,
                "auc_mean": 0.7,
                "auc_sem": 0.01,
                "auc_std": 0.02,
            }
        },
        "raw": {dataset_key: {"e-shd": [10.0 + mean_offset]}},
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2))


def test_resolve_run_directories_from_ids(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "rq1_20260317_avici"
    _write_metrics(
        run_dir,
        run_id="rq1_20260317_avici",
        run_name="rq1",
        model_name="avici",
        dataset_key="id_test",
        mean_offset=0.0,
    )

    resolved = resolve_run_directories(runs_root=runs_root, run_ids=[run_dir.name])
    assert resolved == [run_dir.resolve()]


def test_resolve_run_directories_discovers_metrics_files(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_a = runs_root / "run_a"
    run_b = runs_root / "nested" / "run_b"
    _write_metrics(
        run_a,
        run_id="run_a",
        run_name="A",
        model_name="bcnp",
        dataset_key="id_test",
        mean_offset=0.0,
    )
    _write_metrics(
        run_b,
        run_id="run_b",
        run_name="B",
        model_name="dibs",
        dataset_key="ood_periodic",
        mean_offset=2.0,
    )

    resolved = resolve_run_directories(runs_root=runs_root)
    resolved_set = {path.resolve() for path in resolved}
    assert resolved_set == {run_a.resolve(), run_b.resolve()}


def test_load_runs_dataframe_maps_model_and_dataset_names(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "rq1_20260317_avici"
    _write_metrics(
        run_dir,
        run_id="rq1_20260317_avici",
        run_name="rq1",
        model_name="avici",
        dataset_key="id_test",
        mean_offset=0.0,
    )

    df = load_runs_dataframe([run_dir])
    assert not df.empty

    shd_rows = df[df["Metric"] == "e-shd"]
    assert not shd_rows.empty
    first = shd_rows.iloc[0]
    assert first["RunID"] == "rq1_20260317_avici"
    assert first["Model"] == "AviCi"
    assert first["Dataset"] == "ID"
    assert float(first["Mean"]) == 10.0


def test_generate_all_artifacts_from_runs_writes_expected_files(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_a = runs_root / "rq1_20260317_avici"
    run_b = runs_root / "rq1_20260317_bcnp"
    _write_metrics(
        run_a,
        run_id="rq1_20260317_avici",
        run_name="rq1",
        model_name="avici",
        dataset_key="id_test",
        mean_offset=0.0,
    )
    _write_metrics(
        run_b,
        run_id="rq1_20260317_bcnp",
        run_name="rq1",
        model_name="bcnp",
        dataset_key="ood_periodic",
        mean_offset=1.0,
    )

    output_dir = tmp_path / "graphics"
    generate_all_artifacts_from_runs([run_a, run_b], output_dir)

    assert (output_dir / "structural_metrics.png").exists()
    assert (output_dir / "performance_metrics.png").exists()
    assert (output_dir / "robustness_table.tex").exists()
