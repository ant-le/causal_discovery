"""Tests for real-world dataset integration.

These tests verify the config parsing, dataset construction, and output format
of the RealWorldDataset path without requiring network access or large data
files.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict

import pytest
import torch

from causal_meta.datasets.generators.configs import (
    DataModuleConfig,
    FamilyConfig,
    RealWorldFamilyConfig,
)
from causal_meta.datasets.generators.factory import (
    load_data_module_config,
    load_family_config,
)
from causal_meta.datasets.real_world.registry import load_real_world_dataset
from causal_meta.datasets.real_world.sachs import (
    SACHS_EDGES,
    SACHS_VARIABLES,
    _sachs_adjacency,
)
from causal_meta.datasets.torch_datasets import RealWorldDataset


# ---------------------------------------------------------------------------
# Sachs adjacency sanity checks
# ---------------------------------------------------------------------------


def test_sachs_adjacency_shape() -> None:
    adj = _sachs_adjacency()
    assert adj.shape == (11, 11)
    assert adj.dtype == torch.float32


def test_sachs_adjacency_edge_count() -> None:
    adj = _sachs_adjacency()
    assert int(adj.sum().item()) == 17


def test_sachs_adjacency_is_dag() -> None:
    """The Sachs consensus network must be acyclic."""
    adj = _sachs_adjacency()
    # A DAG's adjacency matrix has all eigenvalues with zero real part
    # (it is nilpotent up to permutation).  A simpler check: matrix power
    # A^d should be zero for a d-node DAG.
    d = adj.shape[0]
    power = adj.clone()
    for _ in range(d - 1):
        power = power @ adj
    assert torch.allclose(power, torch.zeros_like(power)), "Sachs DAG has a cycle"


def test_sachs_variable_count() -> None:
    assert len(SACHS_VARIABLES) == 11


def test_sachs_edge_list_length() -> None:
    assert len(SACHS_EDGES) == 17


# ---------------------------------------------------------------------------
# RealWorldFamilyConfig parsing
# ---------------------------------------------------------------------------


def test_load_real_world_family_config_from_dict() -> None:
    raw = {
        "name": "test_rw",
        "type": "real_world",
        "loader": "sachs",
        "n_nodes": 11,
    }
    cfg = load_family_config(raw)
    assert isinstance(cfg, RealWorldFamilyConfig)
    assert cfg.name == "test_rw"
    assert cfg.loader == "sachs"
    assert cfg.n_nodes == 11
    assert cfg.samples_per_task is None


def test_load_real_world_family_config_with_kwargs() -> None:
    raw = {
        "name": "test_syntren",
        "type": "real_world",
        "loader": "syntren",
        "n_nodes": 20,
        "loader_kwargs": {"data_dir": "/tmp/fake"},
    }
    cfg = load_family_config(raw)
    assert isinstance(cfg, RealWorldFamilyConfig)
    assert cfg.loader_kwargs == {"data_dir": "/tmp/fake"}


def test_load_real_world_family_config_validates_name() -> None:
    raw = {
        "name": "",
        "type": "real_world",
        "loader": "sachs",
        "n_nodes": 11,
    }
    with pytest.raises(ValueError, match="non-empty"):
        load_family_config(raw)


def test_load_real_world_missing_loader() -> None:
    raw = {
        "name": "bad",
        "type": "real_world",
        "n_nodes": 5,
    }
    with pytest.raises(ValueError, match="loader"):
        load_family_config(raw)


def test_load_data_module_config_with_real_world_family() -> None:
    """DataModuleConfig should accept a mix of generative and real-world families."""
    raw = {
        "train_family": {
            "name": "train",
            "n_nodes": 5,
            "graph_cfg": {"type": "er", "edge_prob": 0.3},
            "mech_cfg": {"type": "linear"},
        },
        "test_families": {
            "id_test": {
                "name": "id_test",
                "n_nodes": 5,
                "graph_cfg": {"type": "er", "edge_prob": 0.3},
                "mech_cfg": {"type": "linear"},
            },
            "real_sachs": {
                "name": "real_sachs",
                "type": "real_world",
                "loader": "sachs",
                "n_nodes": 11,
            },
        },
        "seeds_test": [100, 101],
        "seeds_val": [200],
    }
    dm_cfg = load_data_module_config(raw)
    assert "id_test" in dm_cfg.test_families
    assert "real_sachs" in dm_cfg.test_families
    assert isinstance(dm_cfg.test_families["real_sachs"], RealWorldFamilyConfig)
    assert isinstance(dm_cfg.test_families["id_test"], FamilyConfig)


# ---------------------------------------------------------------------------
# RealWorldDataset class
# ---------------------------------------------------------------------------


def _make_toy_dataset(
    n_samples: int = 50,
    n_nodes: int = 5,
    n_seeds: int = 3,
) -> RealWorldDataset:
    torch.manual_seed(42)
    data = torch.randn(n_samples, n_nodes)
    adj = torch.zeros(n_nodes, n_nodes)
    adj[0, 1] = 1.0
    adj[1, 2] = 1.0
    return RealWorldDataset(
        name="toy",
        data=data,
        adjacency=adj,
        seeds=list(range(n_seeds)),
    )


def test_real_world_dataset_length() -> None:
    ds = _make_toy_dataset(n_seeds=5)
    assert len(ds) == 5


def test_real_world_dataset_item_format() -> None:
    ds = _make_toy_dataset(n_samples=50, n_nodes=5, n_seeds=2)
    item = ds[0]

    assert set(item.keys()) == {
        "seed",
        "family_name",
        "data",
        "intervention_mask",
        "adjacency",
        "n_nodes",
        "samples_per_task",
    }
    assert item["family_name"] == "toy"
    assert item["n_nodes"] == 5
    assert isinstance(item["data"], torch.Tensor)
    assert item["data"].shape == (50, 5)
    assert item["adjacency"].shape == (5, 5)
    assert item["intervention_mask"].shape == item["data"].shape
    # No interventions — mask should be all zeros.
    assert item["intervention_mask"].sum().item() == 0.0


def test_real_world_dataset_no_resample_returns_full_data() -> None:
    """When samples_per_task matches data size, no resampling occurs."""
    ds = _make_toy_dataset(n_samples=30, n_nodes=4, n_seeds=2)
    item0 = ds[0]
    item1 = ds[1]
    # Both should return identical data (same underlying matrix, no resample).
    assert torch.equal(item0["data"], item1["data"])
    assert item0["data"].shape[0] == 30


def test_real_world_dataset_resample_is_deterministic() -> None:
    """With resampling, identical seeds produce identical subsets."""
    torch.manual_seed(0)
    data = torch.randn(100, 4)
    adj = torch.zeros(4, 4)
    ds1 = RealWorldDataset(
        name="test",
        data=data,
        adjacency=adj,
        seeds=[42, 43],
        samples_per_task=20,
    )
    ds2 = RealWorldDataset(
        name="test",
        data=data,
        adjacency=adj,
        seeds=[42, 43],
        samples_per_task=20,
    )
    assert torch.equal(ds1[0]["data"], ds2[0]["data"])
    assert torch.equal(ds1[1]["data"], ds2[1]["data"])


def test_real_world_dataset_resample_different_seeds_differ() -> None:
    torch.manual_seed(0)
    data = torch.randn(100, 4)
    adj = torch.zeros(4, 4)
    ds = RealWorldDataset(
        name="test",
        data=data,
        adjacency=adj,
        seeds=[42, 43],
        samples_per_task=20,
    )
    # Different seeds should (very likely) produce different subsets.
    assert not torch.equal(ds[0]["data"], ds[1]["data"])


def test_real_world_dataset_validates_shapes() -> None:
    with pytest.raises(ValueError, match="2-D"):
        RealWorldDataset(
            name="bad",
            data=torch.randn(10),
            adjacency=torch.zeros(3, 3),
            seeds=[0],
        )

    with pytest.raises(ValueError, match="square"):
        RealWorldDataset(
            name="bad",
            data=torch.randn(10, 3),
            adjacency=torch.zeros(3, 4),
            seeds=[0],
        )

    with pytest.raises(ValueError, match="variables"):
        RealWorldDataset(
            name="bad",
            data=torch.randn(10, 3),
            adjacency=torch.zeros(5, 5),
            seeds=[0],
        )


# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------


def test_unknown_loader_raises() -> None:
    with pytest.raises(KeyError, match="no_such_loader"):
        load_real_world_dataset("no_such_loader")


# ---------------------------------------------------------------------------
# SynTReN loader — file-not-found path
# ---------------------------------------------------------------------------


def test_syntren_missing_dir_raises() -> None:
    with pytest.raises(FileNotFoundError, match="SynTReN"):
        load_real_world_dataset("syntren", data_dir="/tmp/nonexistent_syntren_dir_xyz")


def test_syntren_npz_roundtrip(tmp_path: Path) -> None:
    """Loading from a .npz file should produce correct tensors."""
    import numpy as np

    d, n = 8, 50
    rng = np.random.default_rng(99)
    data_np = rng.standard_normal((n, d)).astype(np.float32)
    adj_np = np.zeros((d, d), dtype=np.float32)
    adj_np[0, 1] = 1.0

    np.savez(tmp_path / "data.npz", data=data_np, adjacency=adj_np)

    data, adj = load_real_world_dataset("syntren", data_dir=str(tmp_path))
    assert data.shape == (n, d)
    assert adj.shape == (d, d)
    assert data.dtype == torch.float32
    assert adj.dtype == torch.float32
    assert torch.allclose(data, torch.from_numpy(data_np))
