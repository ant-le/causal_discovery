"""Tests for Phase C (family_metadata) and Phase D (distances) in evaluation.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from causal_meta.datasets.generators.configs import (
    ErdosRenyiConfig,
    FamilyConfig,
    GPMechanismConfig,
    LinearMechanismConfig,
    MLPMechanismConfig,
    PeriodicMechanismConfig,
    SBMConfig,
    ScaleFreeConfig,
)
from causal_meta.runners.tasks.evaluation import (
    _build_family_metadata,
    _compute_distances,
    _extract_graph_type,
    _extract_mech_type,
    _extract_sparsity_param,
)


# ── _extract_graph_type ──────────────────────────────────────────────────


class TestExtractGraphType:
    def test_erdos_renyi_config(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=ErdosRenyiConfig(sparsity=0.05),
            mech_cfg=LinearMechanismConfig(),
        )
        assert "er" in _extract_graph_type(fc).lower()

    def test_scale_free_config(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=ScaleFreeConfig(m=2),
            mech_cfg=LinearMechanismConfig(),
        )
        assert "sf" in _extract_graph_type(fc).lower()

    def test_sbm_config(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=SBMConfig(n_blocks=4, p_intra=0.5, p_inter=0.01),
            mech_cfg=LinearMechanismConfig(),
        )
        assert "sbm" in _extract_graph_type(fc).lower()

    def test_unknown_config_returns_class_heuristic_or_unknown(self) -> None:
        """An arbitrary object with no 'type' attr falls back to class name or 'unknown'."""
        mock_graph = MagicMock(spec=[])
        mock_graph.__class__ = type("FooBarConfig", (), {})
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=mock_graph,
            mech_cfg=LinearMechanismConfig(),
        )
        result = _extract_graph_type(fc)
        assert isinstance(result, str)


# ── _extract_mech_type ───────────────────────────────────────────────────


class TestExtractMechType:
    def test_linear(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=ErdosRenyiConfig(),
            mech_cfg=LinearMechanismConfig(),
        )
        assert "linear" in _extract_mech_type(fc).lower()

    def test_mlp(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=ErdosRenyiConfig(),
            mech_cfg=MLPMechanismConfig(),
        )
        assert "mlp" in _extract_mech_type(fc).lower()

    def test_gp(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=ErdosRenyiConfig(),
            mech_cfg=GPMechanismConfig(),
        )
        assert "gp" in _extract_mech_type(fc).lower()

    def test_periodic(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=ErdosRenyiConfig(),
            mech_cfg=PeriodicMechanismConfig(),
        )
        assert "periodic" in _extract_mech_type(fc).lower()


# ── _extract_sparsity_param ──────────────────────────────────────────────


class TestExtractSparsityParam:
    def test_er_with_sparsity(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=ErdosRenyiConfig(sparsity=0.05),
            mech_cfg=LinearMechanismConfig(),
        )
        assert _extract_sparsity_param(fc) == pytest.approx(0.05)

    def test_er_with_edge_prob(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=ErdosRenyiConfig(edge_prob=0.3),
            mech_cfg=LinearMechanismConfig(),
        )
        assert _extract_sparsity_param(fc) == pytest.approx(0.3)

    def test_scale_free_returns_m(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=ScaleFreeConfig(m=3),
            mech_cfg=LinearMechanismConfig(),
        )
        assert _extract_sparsity_param(fc) == pytest.approx(3.0)

    def test_sbm_returns_none(self) -> None:
        fc = FamilyConfig(
            name="test",
            n_nodes=20,
            graph_cfg=SBMConfig(n_blocks=4, p_intra=0.5, p_inter=0.01),
            mech_cfg=LinearMechanismConfig(),
        )
        assert _extract_sparsity_param(fc) is None


# ── _build_family_metadata ───────────────────────────────────────────────


class TestBuildFamilyMetadata:
    def test_empty_input(self) -> None:
        assert _build_family_metadata({}) == {}

    def test_single_family(self) -> None:
        cfgs = {
            "id_linear_er20": FamilyConfig(
                name="id_linear_er20",
                n_nodes=20,
                graph_cfg=ErdosRenyiConfig(sparsity=0.0526),
                mech_cfg=LinearMechanismConfig(),
            )
        }
        result = _build_family_metadata(cfgs)
        assert "id_linear_er20" in result
        entry = result["id_linear_er20"]
        assert entry["n_nodes"] == 20
        assert "er" in entry["graph_type"].lower()
        assert "linear" in entry["mech_type"].lower()
        assert entry["sparsity_param"] == pytest.approx(0.0526)

    def test_multiple_families(self) -> None:
        cfgs = {
            "id_linear_sf2": FamilyConfig(
                name="sf",
                n_nodes=20,
                graph_cfg=ScaleFreeConfig(m=2),
                mech_cfg=LinearMechanismConfig(),
            ),
            "ood_graph_sbm_linear": FamilyConfig(
                name="sbm",
                n_nodes=20,
                graph_cfg=SBMConfig(n_blocks=4, p_intra=0.5, p_inter=0.01),
                mech_cfg=LinearMechanismConfig(),
            ),
        }
        result = _build_family_metadata(cfgs)
        assert len(result) == 2
        assert result["id_linear_sf2"]["sparsity_param"] == pytest.approx(2.0)
        assert "sparsity_param" not in result["ood_graph_sbm_linear"]


# ── _compute_distances ───────────────────────────────────────────────────


class TestComputeDistances:
    def test_no_train_family_returns_empty(self) -> None:
        dm = MagicMock()
        dm.train_family = None
        result = _compute_distances(dm)
        assert result == {}

    def test_missing_train_family_attr_returns_empty(self) -> None:
        """Data module with no train_family attribute at all."""
        dm = MagicMock(spec=[])  # No attributes
        result = _compute_distances(dm)
        assert result == {}

    def test_with_train_and_test_families(self) -> None:
        dm = MagicMock()
        dm.train_family = MagicMock()
        dm.spectral_distances = {"id_linear_er20": 0.05, "ood_mech_periodic_er40": 1.2}
        dm.test_families = {
            "id_linear_er20": MagicMock(),
            "ood_mech_periodic_er40": MagicMock(),
        }

        result = _compute_distances(dm)
        assert "id_linear_er20" in result
        assert "ood_mech_periodic_er40" in result
        assert result["id_linear_er20"]["spectral"] == pytest.approx(0.05)
        assert result["ood_mech_periodic_er40"]["spectral"] == pytest.approx(1.2)
        # kl_degree should be populated (either computed or fallback to 0.0)
        assert "kl_degree" in result["id_linear_er20"]

    def test_spectral_distances_none_uses_zero(self) -> None:
        dm = MagicMock()
        dm.train_family = MagicMock()
        dm.spectral_distances = None
        dm.test_families = {"fam_a": MagicMock()}

        result = _compute_distances(dm)
        assert result["fam_a"]["spectral"] == pytest.approx(0.0)
