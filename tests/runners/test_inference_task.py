import gzip

import torch
from omegaconf import OmegaConf

from causal_meta.runners.tasks.inference import run as inference_run


class _DummyFixedDataset:
    def __init__(self) -> None:
        self.seeds = [123, 456]

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, idx: int):
        num_samples = 5
        n_nodes = 3
        x = torch.zeros(num_samples, n_nodes)
        adj = torch.zeros(n_nodes, n_nodes)
        return {"seed": int(self.seeds[idx]), "data": x, "adjacency": adj}


class _DummyDataModule:
    def __init__(self) -> None:
        self.test_datasets = {"dummy": _DummyFixedDataset()}

    def setup(self) -> None:
        return None


class _DummyExplicitModel(torch.nn.Module):
    @property
    def needs_pretraining(self) -> bool:
        return False

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        # x: (B, S, N)
        b, _, n = x.shape
        return torch.zeros(b, num_samples, n, n)


def test_inference_writes_artifacts(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "inference": {"n_samples": 3, "output_dir": str(tmp_path)},
        }
    )
    model = _DummyExplicitModel()
    data_module = _DummyDataModule()

    written = inference_run(cfg, model, data_module)
    assert written["dummy"] == 2

    assert (tmp_path / "inference" / "dummy" / "seed_123.pt").exists()
    assert (tmp_path / "inference" / "dummy" / "seed_456.pt").exists()


class _DummyExplicitModelOnes(torch.nn.Module):
    @property
    def needs_pretraining(self) -> bool:
        return False

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        b, _, n = x.shape
        return torch.ones(b, num_samples, n, n)


def test_inference_can_compress_and_limit_samples(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "inference": {
                "n_samples": 5,
                "output_dir": str(tmp_path),
                "cache_compress": True,
                "cache_dtype": "uint8",
                "cache_n_samples": 2,
            },
        }
    )
    model = _DummyExplicitModelOnes()
    data_module = _DummyDataModule()

    written = inference_run(cfg, model, data_module)
    assert written["dummy"] == 2

    path = tmp_path / "inference" / "dummy" / "seed_123.pt.gz"
    assert path.exists()
    with gzip.open(path, "rb") as f:
        artifact = torch.load(f, map_location="cpu")

    assert artifact["graph_samples"].dtype == torch.uint8
    assert artifact["graph_samples"].shape[1] == 2
