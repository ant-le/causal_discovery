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
        self.test_families = {"dummy": type("Family", (), {"n_nodes": 3})()}

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


class _DummyResizableExplicitModel(torch.nn.Module):
    def __init__(self, num_nodes: int) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_node_updates: list[int] = []

    @property
    def needs_pretraining(self) -> bool:
        return False

    def set_num_nodes(self, num_nodes: int) -> None:
        self.num_nodes = num_nodes
        self.num_node_updates.append(num_nodes)

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        b, _, n = x.shape
        if n != self.num_nodes:
            raise ValueError(
                "Input data node count does not match configured num_nodes."
            )
        return torch.zeros(b, num_samples, n, n)


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


def test_inference_batched_produces_same_results(tmp_path) -> None:
    """Test that batched inference produces the same results as unbatched."""

    class _BatchTrackingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.call_batch_sizes: list[int] = []

        @property
        def needs_pretraining(self) -> bool:
            return False

        def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
            b, _, n = x.shape
            self.call_batch_sizes.append(b)
            # Return unique values per batch item to verify correct unbatching
            result = torch.zeros(b, num_samples, n, n)
            for i in range(b):
                result[i] = float(i + 1)
            return result

    # Test with batch_size=1 (default)
    cfg_unbatched = OmegaConf.create(
        {
            "inference": {"n_samples": 2, "output_dir": str(tmp_path / "unbatched")},
        }
    )
    model_unbatched = _BatchTrackingModel()
    data_module = _DummyDataModule()

    written_unbatched = inference_run(cfg_unbatched, model_unbatched, data_module)
    assert written_unbatched["dummy"] == 2
    assert model_unbatched.call_batch_sizes == [1, 1]

    # Test with batch_size=2
    cfg_batched = OmegaConf.create(
        {
            "inference": {
                "n_samples": 2,
                "output_dir": str(tmp_path / "batched"),
                "batch_size": 2,
            },
        }
    )
    model_batched = _BatchTrackingModel()

    written_batched = inference_run(cfg_batched, model_batched, data_module)
    assert written_batched["dummy"] == 2
    assert model_batched.call_batch_sizes == [2]  # Single batched call

    # Verify both produce the same number of artifacts
    unbatched_dir = tmp_path / "unbatched" / "inference" / "dummy"
    batched_dir = tmp_path / "batched" / "inference" / "dummy"
    assert len(list(unbatched_dir.glob("*.pt"))) == 2
    assert len(list(batched_dir.glob("*.pt"))) == 2


def test_inference_updates_model_num_nodes_per_dataset(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "inference": {"n_samples": 2, "output_dir": str(tmp_path)},
        }
    )
    model = _DummyResizableExplicitModel(num_nodes=60)
    data_module = _DummyDataModule()

    written = inference_run(cfg, model, data_module)

    assert written["dummy"] == 2
    assert model.num_nodes == 3
    assert model.num_node_updates == [3]
