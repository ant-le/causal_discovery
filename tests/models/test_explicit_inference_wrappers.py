import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from causal_meta.models.bayesdag.model import BayesDAGModel
from causal_meta.models.dibs.model import DiBSModel


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def test_dibs_wrapper_dependency_contract() -> None:
    model = DiBSModel(num_nodes=3)
    x = torch.zeros(1, 4, 3)

    if _module_available("dibs") and _module_available("jax"):
        # Installed path: import contract should succeed without running expensive sampling.
        jax, jnp, dibs_cls, make_target = model._require_dibs()
        assert jax is not None
        assert jnp is not None
        assert dibs_cls is not None
        assert callable(make_target)
    else:
        # Missing dependency path: sample should fail with actionable message.
        with pytest.raises(RuntimeError, match="dibs-lib"):
            _ = model.sample(x)


def test_dibs_external_python_expands_user_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home_dir = tmp_path / "home"
    python_path = home_dir / ".venv-dibs" / "bin" / "python"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("#!/usr/bin/env python3\n")

    monkeypatch.setenv("HOME", str(home_dir))

    model = DiBSModel(
        num_nodes=3,
        external_process=True,
        external_python="~/.venv-dibs/bin/python",
    )

    assert model._resolve_external_python() == str(python_path.resolve())


def test_dibs_external_python_missing_path_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist" / "python"
    model = DiBSModel(
        num_nodes=3,
        external_process=True,
        external_python=str(missing_path),
    )

    with pytest.raises(FileNotFoundError, match="external_python does not exist"):
        _ = model._resolve_external_python()


def test_dibs_external_process_writes_and_reads_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DiBSModel(
        num_nodes=3,
        mode="linear",
        steps=123,
        n_particles=7,
        external_process=True,
    )
    x = torch.zeros(1, 4, 3)

    calls: list[dict[str, object]] = []

    def _fake_run(cmd, check, timeout, env):
        _ = check
        _ = env
        config_index = cmd.index("--config") + 1
        input_index = cmd.index("--input") + 1
        with open(cmd[config_index], "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        input_payload = np.load(cmd[input_index])
        output_path = Path(payload["output"])
        graph_samples = np.ones(
            (int(payload["num_samples"]), 3, 3),
            dtype=np.float32,
        )
        np.savez(output_path, graph_samples=graph_samples)
        calls.append(
            {
                "timeout": timeout,
                "payload": payload,
                "input_shape": tuple(input_payload["data"].shape),
                "python": cmd[0],
            }
        )

    monkeypatch.setattr("causal_meta.models.dibs.model.subprocess.run", _fake_run)

    samples = model.sample(x, num_samples=5)

    assert tuple(samples.shape) == (1, 5, 3, 3)
    assert torch.equal(samples, torch.ones_like(samples))
    assert len(calls) == 1
    payload = calls[0]["payload"]
    assert payload["mode"] == "linear"
    assert payload["steps"] == 123
    assert payload["n_particles"] == 7
    assert payload["num_nodes"] == 3
    assert calls[0]["input_shape"] == (4, 3)


def test_bayesdag_wrapper_requires_dependency() -> None:
    if _module_available("causica"):
        pytest.skip("causica is installed; skipping missing-dependency check.")

    model = BayesDAGModel(num_nodes=3)
    x = torch.zeros(1, 4, 3)

    with pytest.raises(RuntimeError, match="Project-BayesDAG"):
        _ = model.sample(x)


def test_bayesdag_external_python_expands_user_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home_dir = tmp_path / "home"
    python_path = home_dir / ".venv-bayesdag" / "bin" / "python"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("#!/usr/bin/env python3\n")

    monkeypatch.setenv("HOME", str(home_dir))

    model = BayesDAGModel(num_nodes=3, external_python="~/.venv-bayesdag/bin/python")

    assert model._resolve_external_python() == str(python_path.resolve())


def test_bayesdag_external_python_missing_path_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist" / "python"
    model = BayesDAGModel(num_nodes=3, external_python=str(missing_path))

    with pytest.raises(FileNotFoundError, match="external_python does not exist"):
        _ = model._resolve_external_python()
