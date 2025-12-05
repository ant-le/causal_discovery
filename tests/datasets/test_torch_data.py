import torch

from causal_meta.datasets.data_module import CausalMetaModule, FamilyConfig
from causal_meta.datasets.generators.graphs import ErdosRenyiGenerator
from causal_meta.datasets.generators.mechanisms import LinearMechanismFactory
from causal_meta.datasets.scm import SCMFamily
from causal_meta.datasets.torch_datasets import MetaFixedDataset, MetaIterableDataset
from causal_meta.datasets.utils import collate_fn_scm


def _simple_family(n_nodes: int = 3) -> SCMFamily:
    generator = ErdosRenyiGenerator(edge_prob=0.4)
    mechanism = LinearMechanismFactory(weight_scale=0.1)
    return SCMFamily(n_nodes=n_nodes, graph_generator=generator, mechanism_factory=mechanism)


def test_collate_fn_normalizes_batch() -> None:
    batch = [
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.zeros(2, 2)),
        (torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.ones(2, 2)),
    ]
    normalized, adjs = collate_fn_scm(batch)

    flat = normalized.reshape(-1, normalized.shape[-1])
    assert torch.allclose(flat.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(flat.std(dim=0, unbiased=False), torch.ones(2), atol=1e-6)
    assert adjs.shape == (2, 2, 2)


def test_meta_iterable_dataset_yields_tensors() -> None:
    family = _simple_family(n_nodes=4)
    dataset = MetaIterableDataset(family, base_seed=0, samples_per_task=5)

    x, adj = next(iter(dataset))
    assert isinstance(x, torch.Tensor)
    assert isinstance(adj, torch.Tensor)
    assert x.shape == (5, 4)
    assert adj.shape == (4, 4)


def test_meta_fixed_dataset_is_deterministic_and_caches() -> None:
    family = _simple_family(n_nodes=3)
    dataset = MetaFixedDataset(family, seeds=[42], cache=True, samples_per_task=6)

    x1, adj1 = dataset[0]
    x2, adj2 = dataset[0]

    assert torch.allclose(x1, x2)
    assert torch.equal(adj1, adj2)
    assert 42 in dataset._cache


def test_causal_meta_module_initializes_and_sets_datasets() -> None:
    train_cfg = FamilyConfig(
        name="train",
        graph_type="er",
        mech_type="linear",
        n_nodes=4,
        sparsity=0.3,
    )
    test_cfg = FamilyConfig(
        name="test",
        graph_type="scale_free",
        mech_type="mlp",
        n_nodes=4,
        graph_params={"m": 2},
        mech_params={"hidden_dim": 8},
    )

    module = CausalMetaModule(
        train_family_cfg=train_cfg,
        test_family_cfg=test_cfg,
        seeds_train=[0, 1, 2],
        seeds_test=[10, 11, 12],
        base_seed=0,
        samples_per_task=4,
    )

    module.setup()

    assert module.train_dataset is not None
    assert module.test_dataset is not None
    assert module.spectral_distance is not None


def test_causal_meta_module_dataloaders() -> None:
    train_cfg = FamilyConfig(name="train", graph_type="er", mech_type="linear", n_nodes=4, sparsity=0.3)
    test_cfg = FamilyConfig(name="test", graph_type="er", mech_type="linear", n_nodes=4, sparsity=0.3)

    module = CausalMetaModule(
        train_family_cfg=train_cfg,
        test_family_cfg=test_cfg,
        seeds_train=[0],
        seeds_test=[1],
        num_workers=2,
        pin_memory=True,
    )
    module.setup()

    train_loader = module.train_dataloader()
    test_loader = module.test_dataloader()

    assert train_loader.num_workers == 2
    assert train_loader.pin_memory is True
    assert test_loader.num_workers == 0  # Fixed set usually 0 workers
    assert test_loader.pin_memory is True
