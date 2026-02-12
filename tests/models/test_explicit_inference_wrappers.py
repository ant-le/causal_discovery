import importlib.util

import pytest
import torch

from causal_meta.models.bayesdag.model import BayesDAGModel
from causal_meta.models.dibs.model import DiBSModel


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def test_dibs_wrapper_requires_dependency() -> None:
    if _module_available("dibs"):
        pytest.skip("dibs-lib is installed; skipping missing-dependency check.")

    model = DiBSModel(num_nodes=3)
    x = torch.zeros(1, 4, 3)

    with pytest.raises(RuntimeError, match="dibs-lib"):
        _ = model.sample(x)


def test_bayesdag_wrapper_requires_dependency() -> None:
    if _module_available("causica"):
        pytest.skip("causica is installed; skipping missing-dependency check.")

    model = BayesDAGModel(num_nodes=3)
    x = torch.zeros(1, 4, 3)

    with pytest.raises(RuntimeError, match="Project-BayesDAG"):
        _ = model.sample(x)
