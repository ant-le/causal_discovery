import networkx as nx
import torch

from causal_meta.datasets.scm import SCMFamily
from causal_meta.datasets.generators.graphs import ErdosRenyiGenerator
from causal_meta.datasets.generators.mechanisms import LinearMechanismFactory


def test_scm_family_generates_dag() -> None:
    family = SCMFamily(
        n_nodes=4,
        graph_generator=ErdosRenyiGenerator(edge_prob=0.5),
        mechanism_factory=LinearMechanismFactory(),
    )

    instance = family.sample_task(seed=21)
    adjacency = instance.adjacency_matrix
    graph = nx.from_numpy_array(adjacency.numpy(), create_using=nx.DiGraph)

    assert adjacency.shape == (4, 4)
    assert nx.is_directed_acyclic_graph(graph)


def test_ancestral_sampling_returns_finite_values() -> None:
    family = SCMFamily(
        n_nodes=3,
        graph_generator=ErdosRenyiGenerator(edge_prob=0.6),
        mechanism_factory=LinearMechanismFactory(weight_scale=0.1),
    )
    instance = family.sample_task(seed=3)

    samples = instance.sample(num_samples=15)

    assert samples.shape == (15, 3)
    assert torch.isfinite(samples).all()


def test_scm_family_sample_graph_matches_sample_task_adjacency() -> None:
    family = SCMFamily(
        n_nodes=4,
        graph_generator=ErdosRenyiGenerator(edge_prob=0.5),
        mechanism_factory=LinearMechanismFactory(weight_scale=0.1),
    )
    seed = 123
    adj_only = family.sample_graph(seed)
    instance = family.sample_task(seed)
    assert torch.equal(adj_only, instance.adjacency_matrix)
