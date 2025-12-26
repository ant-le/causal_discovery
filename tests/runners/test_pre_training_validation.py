import torch
from torch.utils.data import DataLoader

from causal_meta.runners.tasks.pre_training import validate


class _DummyDataModule:
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self.val_called = False

    def val_dataloader(self):
        self.val_called = True
        return {"id": self._loader}

    def test_dataloader(self):
        raise AssertionError("test_dataloader must not be called for validation.")


class _DummyModel(torch.nn.Module):
    def __init__(self, n_nodes: int) -> None:
        super().__init__()
        self.n_nodes = n_nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.zeros(batch_size, self.n_nodes, self.n_nodes)

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.zeros(batch_size, n_samples, self.n_nodes, self.n_nodes)


def test_pre_training_validate_uses_validation_loader() -> None:
    n_nodes = 3
    x = torch.zeros(5, n_nodes)
    adj = torch.zeros(n_nodes, n_nodes)
    loader = DataLoader([(x, adj)], batch_size=1)

    data_module = _DummyDataModule(loader)
    model = _DummyModel(n_nodes=n_nodes)

    metrics = validate(model, data_module, torch.device("cpu"))
    assert data_module.val_called is True
    assert metrics["id/shd"] == 0.0
    assert "mean_f1" in metrics
