import networkx as nx
import torch

from causal_meta.datasets.scm import SCMFamily
from causal_meta.datasets.generators.graphs import (
    ErdosRenyiGenerator,
    SBMGenerator,
    ScaleFreeGenerator,
)
from causal_meta.datasets.generators.mechanisms import (
    LinearMechanismFactory,
    MLPMechanismFactory,
)


def test_scm_instance_sampling_shape() -> None:
    generator = SBMGenerator(n_blocks=2, p_intra=0.8, p_inter=0.1)
    factory = MLPMechanismFactory(hidden_dim=8)
    family = SCMFamily(n_nodes=6, graph_generator=generator, mechanism_factory=factory)

    instance = family.sample_task(seed=7)
    samples = instance.sample(n_samples=5)

    assert samples.shape == (5, 6)
    assert torch.isfinite(samples).all()


def test_sample_task_seed_reproducibility() -> None:
    generator = ErdosRenyiGenerator(edge_prob=0.3)
    factory = LinearMechanismFactory(weight_scale=0.5)
    family = SCMFamily(n_nodes=5, graph_generator=generator, mechanism_factory=factory)

    instance_a = family.sample_task(seed=123)
    instance_b = family.sample_task(seed=123)

    assert torch.equal(instance_a.adjacency_matrix, instance_b.adjacency_matrix)

    for mech_a, mech_b in zip(instance_a.mechanisms, instance_b.mechanisms):
        for key, tensor_a in mech_a.state_dict().items():
            tensor_b = mech_b.state_dict()[key]
            assert torch.allclose(tensor_a, tensor_b)


def test_erdos_renyi_density() -> None:
    n_nodes = 80
    edge_prob = 0.2
    generator = ErdosRenyiGenerator(edge_prob=edge_prob)

    adjacency = generator(n_nodes, seed=0)
    possible_edges = n_nodes * (n_nodes - 1) / 2
    density = adjacency.sum().item() / possible_edges

    assert abs(density - edge_prob) < 0.03


def test_scale_free_connected() -> None:
    generator = ScaleFreeGenerator(m=2)
    adjacency = generator(n_nodes=20, seed=99)

    undirected = nx.from_numpy_array(adjacency.numpy(), create_using=nx.Graph)
    assert nx.is_connected(undirected)
