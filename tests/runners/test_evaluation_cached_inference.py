import gzip

import torch
from omegaconf import OmegaConf

from causal_meta.runners.tasks.evaluation import run as evaluation_run


class _DummyFixedDataset:
    def __init__(self) -> None:
        self.seeds = [123]
        self.num_samples = 5
        self.n_nodes = 3

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, idx: int):
        x = torch.zeros(self.num_samples, self.n_nodes)
        adj = torch.zeros(self.n_nodes, self.n_nodes)
        return {"seed": int(self.seeds[idx]), "data": x, "adjacency": adj}


class _DummyLoader:
    def __init__(self, dataset: _DummyFixedDataset) -> None:
        self.dataset = dataset


class _DummyDataModule:
    def __init__(self, dataset: _DummyFixedDataset) -> None:
        self._dataset = dataset

    def test_dataloader(self):
        return {"dummy": _DummyLoader(self._dataset)}

    def test_interventional_dataloader(self):
        return {}


class _ModelThatShouldNotSample(torch.nn.Module):
    @property
    def needs_pretraining(self) -> bool:
        return False

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        raise RuntimeError("sample() should not be called when cached inference exists")


def test_evaluation_uses_cached_inference_artifact(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "name": "eval_cached",
            "inference": {
                "n_samples": 5,
                "inil_graph_samples": 2,
                "use_cached_inference": True,
                "cache_inference": False,
                "cache_compress": True,
            },
        }
    )

    dataset = _DummyFixedDataset()
    data_module = _DummyDataModule(dataset)
    model = _ModelThatShouldNotSample()

    out_dir = tmp_path / "inference" / "dummy"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "seed_123.pt.gz"

    graph_samples = torch.zeros(
        1, 3, dataset.n_nodes, dataset.n_nodes, dtype=torch.uint8
    )
    with gzip.open(artifact_path, "wb") as f:
        torch.save(
            {
                "seed": 123,
                "idx": 0,
                "graph_samples": graph_samples,
                "true_adj": torch.zeros(
                    dataset.n_nodes, dataset.n_nodes, dtype=torch.uint8
                ),
            },
            f,
        )

    evaluation_run(cfg, model, data_module, output_dir=tmp_path)

    assert (tmp_path / "metrics.json").exists()
