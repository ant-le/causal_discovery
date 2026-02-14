from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from causal_meta.models.base import BaseModel
from causal_meta.runners.tasks.pre_training import _build_scheduler
from causal_meta.runners.tasks.pre_training import run as pre_training_run
from causal_meta.runners.tasks.pre_training import save_checkpoint


class _DummyPretrainModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    @property
    def needs_pretraining(self) -> bool:
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        n_nodes = x.shape[-1]
        return self.weight * torch.ones(batch, n_nodes, n_nodes, device=x.device)

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        batch = x.shape[0]
        n_nodes = x.shape[-1]
        return torch.zeros(batch, num_samples, n_nodes, n_nodes, device=x.device)

    def calculate_loss(
        self, output: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        _ = kwargs
        diff = (output - target).pow(2)
        return diff.view(diff.shape[0], -1).mean(dim=1)


class _DummyTrainDataset:
    def __init__(self, base_seed: int) -> None:
        self.base_seed = int(base_seed)


class _DummyDataModule:
    def __init__(self, base_seed: int = 10) -> None:
        n_nodes = 3
        n_samples = 5
        self.train_dataset = _DummyTrainDataset(base_seed=base_seed)
        self._train_batches = [
            {
                "seed": torch.tensor([1]),
                "data": torch.zeros(1, n_samples, n_nodes),
                "adjacency": torch.zeros(1, n_nodes, n_nodes),
            }
        ]
        self._val_batches = [
            {
                "seed": torch.tensor([2]),
                "data": torch.zeros(1, n_samples, n_nodes),
                "adjacency": torch.zeros(1, n_nodes, n_nodes),
            }
        ]

    def train_dataloader(self):
        return self._train_batches

    def val_dataloader(self):
        return {"val": self._val_batches}


def _make_cfg(max_steps: int = 2) -> OmegaConf:
    return OmegaConf.create(
        {
            "name": "pretrain_test",
            "seed": 777,
            "data": {
                "batch_size_train": 1,
                "base_seed": 10,
                "num_workers": 0,
            },
            "trainer": {
                "lr": 1e-3,
                "max_steps": max_steps,
                "log_every_n_steps": 100,
                "val_check_interval": 100,
                "checkpoint_every_n_steps": 100,
                "accumulate_grad_batches": 1,
                "scheduler": "none",
                "amp": False,
                "amp_dtype": "bf16",
            },
            "inference": {"n_samples": 2},
        }
    )


def test_save_checkpoint_contains_stream_resume_metadata(tmp_path) -> None:
    cfg = _make_cfg(max_steps=3)
    model = _DummyPretrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    out = tmp_path / "checkpoint.pt"
    save_checkpoint(
        cfg,
        model,
        optimizer,
        scheduler,
        scaler,
        step=3,
        filepath=out,
        train_stream_initial_base_seed=10,
        world_size=2,
        train_batch_size=4,
        accumulate_grad_batches=2,
    )

    state = torch.load(out, map_location="cpu")
    assert state["step"] == 3
    assert state["experiment_seed"] == 777
    assert state["train_stream_initial_base_seed"] == 10
    assert state["train_stream_world_size"] == 2
    assert state["train_stream_batch_size_train"] == 4
    assert state["train_stream_accumulate_grad_batches"] == 2
    assert state["train_stream_next_base_seed_if_num_workers_0"] == 58
    assert "scheduler_state_dict" in state


def test_pre_training_run_resumes_model_and_stream_seed(tmp_path) -> None:
    cfg = _make_cfg(max_steps=2)
    model = _DummyPretrainModel()
    data_module = _DummyDataModule(base_seed=10)

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_path = ckpt_dir / "last.pt"

    with torch.no_grad():
        model.weight.fill_(5.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    save_checkpoint(
        cfg,
        model,
        optimizer,
        scheduler=None,
        scaler=scaler,
        step=2,
        filepath=resume_path,
        train_stream_initial_base_seed=10,
        world_size=1,
        train_batch_size=1,
        accumulate_grad_batches=1,
    )

    with torch.no_grad():
        model.weight.zero_()

    pre_training_run(cfg, model, data_module, output_dir=tmp_path)

    assert torch.isclose(model.weight.detach().cpu(), torch.tensor(5.0))
    assert data_module.train_dataset.base_seed == 12

    last_state = torch.load(resume_path, map_location="cpu")
    assert last_state["step"] == 2


def test_build_scheduler_none_and_invalid() -> None:
    param = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([param], lr=1e-3)

    cfg_none = OmegaConf.create({"trainer": {"scheduler": "none", "max_steps": 10}})
    assert _build_scheduler(optimizer, cfg_none) is None

    cfg_invalid = OmegaConf.create(
        {"trainer": {"scheduler": "linear", "max_steps": 10}}
    )
    try:
        _build_scheduler(optimizer, cfg_invalid)
    except ValueError as exc:
        assert "trainer.scheduler" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported scheduler type")
