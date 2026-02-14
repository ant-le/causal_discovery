import importlib.util

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


def test_bayesdag_wrapper_requires_dependency() -> None:
    if _module_available("causica"):
        pytest.skip("causica is installed; skipping missing-dependency check.")

    model = BayesDAGModel(num_nodes=3)
    x = torch.zeros(1, 4, 3)

    with pytest.raises(RuntimeError, match="Project-BayesDAG"):
        _ = model.sample(x)
