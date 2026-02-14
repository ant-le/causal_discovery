from __future__ import annotations

import torch

from causal_meta.datasets.generators.graphs.er import ErdosRenyiGenerator
from causal_meta.datasets.generators.mechanisms.linear import \
    LinearMechanismFactory
from causal_meta.datasets.scm import SCMFamily
from causal_meta.runners.metrics.scm import SCMMetrics
from causal_meta.runners.utils.scoring import LinearGaussianScorer


def test_linear_gaussian_scorer_fit_and_score_nll() -> None:
    torch.manual_seed(0)
    n = 64

    x0 = torch.randn(n)
    x1 = 2.0 * x0 + 0.1 * torch.randn(n)
    data = torch.stack([x0, x1], dim=1)

    adjacency = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    scorer = LinearGaussianScorer(adjacency=adjacency, obs_data=data)
    scorer.fit()

    nll = scorer.score_nll(data)
    assert torch.isfinite(torch.tensor(nll))
    assert nll >= 0.0


def test_scm_metrics_inil_with_family_generated_interventions() -> None:
    family = SCMFamily(
        n_nodes=3,
        graph_generator=ErdosRenyiGenerator(edge_prob=0.4),
        mechanism_factory=LinearMechanismFactory(weight_scale=0.5),
    )
    seed = 13

    instance = family.sample_task(seed)
    obs_data = instance.sample(32)
    graph_samples = instance.adjacency_matrix.unsqueeze(0)

    metrics = SCMMetrics(metrics=["inil"])
    metrics.update(
        obs_data=obs_data,
        graph_samples=graph_samples,
        family=family,
        seeds=[seed],
        prefix="toy",
    )

    summary = metrics.compute(summary_stats=False)
    assert "inil" in summary
    assert "toy/inil" in summary
    assert torch.isfinite(torch.tensor(summary["inil"]))


def test_scm_metrics_inil_with_precomputed_interventional_data() -> None:
    family = SCMFamily(
        n_nodes=3,
        graph_generator=ErdosRenyiGenerator(edge_prob=0.3),
        mechanism_factory=LinearMechanismFactory(weight_scale=0.4),
    )
    seed = 9
    instance = family.sample_task(seed)

    obs_data = instance.sample(24)
    graph_samples = instance.adjacency_matrix.unsqueeze(0)
    interventional_data = SCMMetrics._generate_interventional_data(
        family=family,
        seed=seed,
        n_samples=24,
    )

    metrics = SCMMetrics(metrics=["inil"])
    metrics.update(
        obs_data=obs_data,
        graph_samples=graph_samples,
        interventional_data=interventional_data,
        prefix="toy",
    )

    raw = metrics.get_raw_results()
    assert "inil" in raw
    assert len(raw["inil"]) == 1
    assert torch.isfinite(torch.tensor(raw["inil"][0]))
