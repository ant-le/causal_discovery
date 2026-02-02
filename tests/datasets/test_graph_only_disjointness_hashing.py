from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.datasets.generators.graphs import ErdosRenyiGenerator
from causal_meta.datasets.generators.mechanisms import LinearMechanismFactory
from causal_meta.datasets.scm import SCMFamily


class CountingMechanismFactory(LinearMechanismFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return super().__call__(*args, **kwargs)


def test_disjointness_hashing_does_not_build_mechanisms() -> None:
    dummy_cfg = {
        "train_family": {
            "name": "train",
            "n_nodes": 3,
            "graph_cfg": {"type": "er", "sparsity": 0.3},
            "mech_cfg": {"type": "linear"},
        },
        "test_families": {
            "test": {
                "name": "test",
                "n_nodes": 3,
                "graph_cfg": {"type": "er", "sparsity": 0.3},
                "mech_cfg": {"type": "linear"},
            }
        },
        "seeds_val": [0],
        "seeds_test": [1],
    }
    module = CausalMetaModule.from_config(dummy_cfg)

    mech = CountingMechanismFactory(weight_scale=0.1)
    family = SCMFamily(
        n_nodes=3, graph_generator=ErdosRenyiGenerator(edge_prob=0.4), mechanism_factory=mech
    )
    _ = module._sample_hashes(family, seeds=[0, 1, 2])
    assert mech.calls == 0

