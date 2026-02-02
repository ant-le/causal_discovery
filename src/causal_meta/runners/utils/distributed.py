from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


def select_device(*, local_rank: int, is_distributed: bool) -> torch.device:
    """
    Select the preferred device for this process.

    CUDA is preferred when available (for DDP compatibility). If running
    single-process and CUDA is unavailable, MPS is selected when available;
    otherwise CPU is used.

    Args:
        local_rank: Local rank provided by torchrun/submitit.
        is_distributed: Whether the process is in a distributed context.

    Returns:
        torch.device: The selected device for this process.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if not is_distributed and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class DistributedContext:
    """
    Lightweight helper for torch.distributed / DDP execution.

    `setup()` initializes the default process group when running under torchrun/submitit.
    `current()` is side-effect free and just reads the current distributed state.
    """

    is_distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.is_distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()

    @classmethod
    def current(cls) -> DistributedContext:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        is_distributed = dist.is_available() and dist.is_initialized()
        device = select_device(local_rank=local_rank, is_distributed=is_distributed)

        if is_distributed:
            return cls(
                is_distributed=True,
                rank=int(dist.get_rank()),
                world_size=int(dist.get_world_size()),
                local_rank=local_rank,
                device=device,
            )

        return cls(
            is_distributed=False,
            rank=0,
            world_size=1,
            local_rank=local_rank,
            device=device,
        )

    @classmethod
    def setup(cls) -> DistributedContext:
        """
        Initialize torch.distributed if requested by environment variables.

        This uses the default `env://` initialization which requires torchrun/submitit
        to provide RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT.
        """
        if dist.is_available() and dist.is_initialized():
            return cls.current()

        wants_distributed = "LOCAL_RANK" in os.environ or "RANK" in os.environ
        if not wants_distributed:
            return cls.current()

        missing = [
            k
            for k in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
            if k not in os.environ
        ]
        if missing:
            raise ValueError(
                "Distributed run detected (LOCAL_RANK/RANK set) but missing required "
                f"environment variables for env:// init: {missing}."
            )

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
        dist_ctx = cls.current()
        device = select_device(local_rank=local_rank, is_distributed=True)
        return cls(
            is_distributed=dist_ctx.is_distributed,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
            local_rank=dist_ctx.local_rank,
            device=device,
        )

    @staticmethod
    def cleanup() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
