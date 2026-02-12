from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable, Sequence, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.models.base import BaseModel
from causal_meta.runners.logger.base import BaseLogger
from causal_meta.runners.metrics.graph import Metrics
from causal_meta.runners.utils.artifacts import (get_model_name,
                                                 resolve_output_dir)
from causal_meta.runners.utils.distributed import DistributedContext
from causal_meta.runners.utils.seeding import get_experiment_seed

log = logging.getLogger(__name__)


def run(
    cfg: DictConfig,
    model: nn.Module,
    data_module: CausalMetaModule,
    *,
    logger: BaseLogger | None = None,
    output_dir: Path | None = None,
) -> None:
    """
    Executes the pre-training loop for meta-learning models (e.g., Avici, BCNP).

    This function handles:
    1.  DDP setup and model wrapping.
    2.  Loading the infinite training stream (`train_dataloader`).
    3.  Optimization loop (forward pass, loss calculation, backward pass).
    4.  Periodic validation and checkpointing (best/last).
    5.  Logging to the provided logger (e.g., WandB).

    Args:
        cfg: Experiment configuration.
        model: The model instance to train.
        data_module: The data module providing training and validation dataloaders.
        logger: Optional logger instance (BaseLogger).
        output_dir: Directory to save checkpoints and logs.
    """
    # 0. DDP Utilities
    dist_ctx = DistributedContext.current()
    is_distributed = dist_ctx.is_distributed
    rank = dist_ctx.rank
    world_size = dist_ctx.world_size

    # 1. Setup
    if rank == 0:
        log.info(f"Starting experiment: {cfg.name}")
        log.info(f"Working directory: {os.getcwd()}")

    # Device is already set in pipe.py, but we can double check or get it from model
    device = next(model.parameters()).device

    # 2. Data
    # train_dataloader handles DDP seeding internally
    train_loader = data_module.train_dataloader()

    train_batch_size = int(getattr(cfg.data, "batch_size_train", 1))
    if train_batch_size < 1:
        raise ValueError("data.batch_size_train must be >= 1")

    accumulate_grad_batches = int(cfg.trainer.get("accumulate_grad_batches", 1))
    if accumulate_grad_batches < 1:
        raise ValueError("trainer.accumulate_grad_batches must be >= 1")

    # Unwrap model for attribute access
    unwrapped = model.module if isinstance(model, DDP) else model
    model_unwrapped = cast(BaseModel, unwrapped)
    if not isinstance(model_unwrapped, BaseModel):
        raise TypeError("Expected model to be a BaseModel or DDP-wrapped BaseModel.")

    if not model_unwrapped.needs_pretraining:
        if rank == 0:
            log.info("Model does not require pre-training. Exiting.")
        return

    # 3. Optimizer
    lr = float(cfg.trainer.lr)
    weight_decay = float(cfg.trainer.get("weight_decay", 1e-4))
    betas = _maybe_parse_betas(cfg.trainer.get("optimizer_betas", None))
    optimizer_eps = _maybe_parse_eps(cfg.trainer.get("optimizer_eps", None))
    optimizer_kwargs: dict[str, Any] = {"lr": lr, "weight_decay": weight_decay}
    if betas is not None:
        optimizer_kwargs["betas"] = betas
    if optimizer_eps is not None:
        optimizer_kwargs["eps"] = optimizer_eps
    optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)

    scheduler = _build_scheduler(optimizer, cfg)

    amp_enabled = bool(cfg.trainer.get("amp", False)) and device.type == "cuda"
    amp_dtype_name = str(cfg.trainer.get("amp_dtype", "bf16")).lower()
    if amp_dtype_name in {"bf16", "bfloat16"}:
        amp_dtype = torch.bfloat16
        scaler = GradScaler("cuda", enabled=False)
    elif amp_dtype_name in {"fp16", "float16"}:
        amp_dtype = torch.float16
        scaler = GradScaler("cuda", enabled=amp_enabled)
    else:
        raise ValueError("trainer.amp_dtype must be one of: bf16, fp16")

    # 4. Training Loop
    max_steps = int(cfg.trainer.max_steps)
    log_every_n_steps = int(cfg.trainer.get("log_every_n_steps", 100))
    val_check_interval = int(cfg.trainer.get("val_check_interval", 1000))
    regulariser_update_interval = int(cfg.trainer.get("regulariser_update_interval", 0))
    grad_clip_norm = float(cfg.trainer.get("grad_clip_norm", 1.0))

    best_val_metric = float("-inf")
    step = 0

    if rank == 0:
        log.info("Starting training loop...")

    # Determine output directory
    output_dir = resolve_output_dir(cfg, output_dir)
    model_name = get_model_name(cfg, model)

    path = output_dir / "checkpoints"
    if rank == 0:
        path.mkdir(parents=True, exist_ok=True)

    # RESUME LOGIC
    start_step = 0
    resume_path = path / "last.pt"
    train_stream_initial_base_seed = int(getattr(cfg.data, "base_seed", 0))
    if resume_path.exists():
        if rank == 0:
            log.info(f"Resuming from checkpoint: {resume_path}")
        # Load to CPU for portability across nodes/devices.
        checkpoint = torch.load(resume_path, map_location="cpu")

        model_unwrapped.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_step = checkpoint["step"]
        step = start_step

        train_stream_initial_base_seed = int(
            checkpoint.get(
                "train_stream_initial_base_seed", train_stream_initial_base_seed
            )
        )

        if rank == 0:
            log.info(f"Resumed at step {step}")

        saved_world_size = int(checkpoint.get("train_stream_world_size", world_size))
        if saved_world_size != world_size and rank == 0:
            log.warning(
                f"Checkpoint world_size={saved_world_size} differs from current world_size={world_size}; "
                "training data stream resume will be best-effort."
            )

        saved_batch_size = int(
            checkpoint.get("train_stream_batch_size_train", train_batch_size)
        )
        if saved_batch_size != train_batch_size and rank == 0:
            log.warning(
                f"Checkpoint batch_size_train={saved_batch_size} differs from current batch_size_train={train_batch_size}; "
                "training data stream resume will be best-effort."
            )

        saved_accum = int(
            checkpoint.get(
                "train_stream_accumulate_grad_batches", accumulate_grad_batches
            )
        )
        if saved_accum != accumulate_grad_batches and rank == 0:
            log.warning(
                f"Checkpoint accumulate_grad_batches={saved_accum} differs from current accumulate_grad_batches={accumulate_grad_batches}; "
                "training data stream resume will be best-effort."
            )

        if getattr(cfg.data, "num_workers", 0) > 0:
            log.warning(
                "Resuming with an IterableDataset stream is not strictly reproducible when num_workers>0; "
                "the data stream may differ from the original run."
            )

    # Deterministic-ish resume for the streaming dataset when DataLoader uses num_workers=0.
    # With num_workers=0, MetaIterableDataset uses a single stream per rank with stride=world_size.
    # If batch_size_train or gradient accumulation is used, each optimizer step consumes
    # multiple stream items per rank.
    if (
        hasattr(data_module, "train_dataset")
        and data_module.train_dataset is not None
        and getattr(cfg.data, "num_workers", 0) == 0
    ):
        data_module.train_dataset.base_seed = (
            train_stream_initial_base_seed
            + step * world_size * train_batch_size * accumulate_grad_batches
        )
        if rank == 0 and start_step > 0:
            log.info(
                f"Resumed train stream base_seed={data_module.train_dataset.base_seed} "
                f"(initial={train_stream_initial_base_seed}, step={step}, world_size={world_size}, "
                f"batch_size_train={train_batch_size}, accumulate_grad_batches={accumulate_grad_batches})."
            )

    checkpoint_every_n_steps = int(
        cfg.trainer.get("checkpoint_every_n_steps", val_check_interval)
    )

    train_iter = iter(train_loader)

    while step < max_steps:

        model.train()
        optimizer.zero_grad()

        # Loss is tracked as a scalar mean across micro-batches.
        loss_accum = torch.tensor(0.0, device=device)

        next_step = step + 1
        update_regulariser = (
            regulariser_update_interval > 0
            and next_step % regulariser_update_interval == 0
        )

        for micro in range(accumulate_grad_batches):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_data = batch["data"].to(device)
            adjacency_matrix = batch["adjacency"].to(device)

            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                    output = model(input_data)
                    loss_vec = model_unwrapped.calculate_loss(
                        output,
                        adjacency_matrix,
                        update_regulariser=update_regulariser
                        and (micro == accumulate_grad_batches - 1),
                    )
                    loss = loss_vec.mean() / float(accumulate_grad_batches)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                output = model(input_data)
                loss_vec = model_unwrapped.calculate_loss(
                    output,
                    adjacency_matrix,
                    update_regulariser=update_regulariser
                    and (micro == accumulate_grad_batches - 1),
                )
                loss = loss_vec.mean() / float(accumulate_grad_batches)
                loss.backward()

            loss_accum = loss_accum + loss.detach()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            _maybe_clip_grad_norm(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            _maybe_clip_grad_norm(model.parameters(), grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        step += 1

        # Logging
        if step % log_every_n_steps == 0:
            loss_value = _reduce_loss_for_logging(loss_accum, world_size)
            if rank == 0:
                log_str = f"Step {step}/{max_steps} | Loss: {loss_value:.4f}"
                log.info(log_str)

                if logger:
                    logger.log_metrics({"train/loss": loss_value}, step=step)

        # Validation
        if step % val_check_interval == 0:
            val_n_samples = int(cfg.get("inference", {}).get("n_samples", 10))
            # Validate involves sampling, which might need unwrapped model or handling in validate
            val_metrics = validate(
                model_unwrapped,
                data_module,
                device,
                num_samples=val_n_samples,
            )

            if rank == 0:
                log.info(f"Validation at step {step}: {val_metrics}")

                if logger:
                    prefixed = {k: v for k, v in val_metrics.items() if "/" in k}
                    log_payload = {f"val/{k}": v for k, v in prefixed.items()}
                    if "mean_e-edgef1" in val_metrics:
                        log_payload["val/mean_e-edgef1"] = val_metrics["mean_e-edgef1"]
                    logger.log_metrics(log_payload, step=step)

                # Save Best (using E-F1 as metric)
                current_metric = val_metrics.get("mean_e-edgef1", 0.0)
                if current_metric > best_val_metric:
                    best_val_metric = current_metric
                    save_checkpoint(
                        cfg,
                        model_unwrapped,
                        optimizer,
                        scheduler,
                        scaler,
                        step,
                        path / "best.pt",
                        train_stream_initial_base_seed=train_stream_initial_base_seed,
                        world_size=world_size,
                        train_batch_size=train_batch_size,
                        accumulate_grad_batches=accumulate_grad_batches,
                    )
                    log.info(f"New best model saved! E-F1: {best_val_metric:.4f}")

        # Periodic "last" checkpoint (rank 0 only)
        if checkpoint_every_n_steps > 0 and step % checkpoint_every_n_steps == 0:
            if rank == 0:
                save_checkpoint(
                    cfg,
                    model_unwrapped,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                    path / "last.pt",
                    train_stream_initial_base_seed=train_stream_initial_base_seed,
                    world_size=world_size,
                    train_batch_size=train_batch_size,
                    accumulate_grad_batches=accumulate_grad_batches,
                )

    if rank == 0:
        log.info("Training finished.")
        save_checkpoint(
            cfg,
            model_unwrapped,
            optimizer,
            scheduler,
            scaler,
            step,
            path / "last.pt",
            train_stream_initial_base_seed=train_stream_initial_base_seed,
            world_size=world_size,
            train_batch_size=train_batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
        )

    if is_distributed:
        dist_ctx.barrier()  # Wait for rank 0 to save

    best_path = path / "best.pt"
    if best_path.exists():
        if rank == 0:
            log.info(f"Reloading best model from {best_path}")

        checkpoint = torch.load(best_path, map_location="cpu")
        model_unwrapped.load_state_dict(checkpoint["model_state_dict"])

    if is_distributed:
        dist_ctx.barrier()


def validate(
    model: BaseModel,
    data_module: CausalMetaModule,
    device: torch.device,
    num_samples: int = 10,
) -> dict[str, float]:
    model.eval()
    val_loaders = data_module.val_dataloader()

    metrics_handler = Metrics()

    with torch.no_grad():
        for name, loader in val_loaders.items():
            for batch in loader:
                input_data = batch["data"].to(device)
                adjacency_matrix = batch["adjacency"].to(device)

                samples = model.sample(input_data, num_samples=num_samples)
                samples_for_metrics = samples.permute(1, 0, 2, 3)

                metrics_handler.update(
                    adjacency_matrix, samples_for_metrics, prefix=name
                )
    avg_metrics = metrics_handler.compute(summary_stats=False)

    # Calculate mean F1 across all datasets (Alias for checkpointing)
    if "e-edgef1" in avg_metrics:
        avg_metrics["mean_e-edgef1"] = avg_metrics["e-edgef1"]

    return avg_metrics


def save_checkpoint(
    cfg: DictConfig,
    model: BaseModel,
    optimizer: optim.Optimizer,
    scheduler: Any | None,
    scaler: GradScaler,
    step: int,
    filepath: Path,
    *,
    train_stream_initial_base_seed: int,
    world_size: int,
    train_batch_size: int,
    accumulate_grad_batches: int,
) -> None:
    experiment_seed = get_experiment_seed(
        cfg, fallback=int(train_stream_initial_base_seed)
    )
    state: dict[str, Any] = {
        "step": step,
        "experiment_seed": int(experiment_seed),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_stream_initial_base_seed": int(train_stream_initial_base_seed),
        "train_stream_world_size": int(world_size),
        "train_stream_batch_size_train": int(train_batch_size),
        "train_stream_accumulate_grad_batches": int(accumulate_grad_batches),
        "train_stream_next_base_seed_if_num_workers_0": int(
            train_stream_initial_base_seed
            + step * world_size * train_batch_size * accumulate_grad_batches
        ),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(
        state,
        filepath,
    )


def _build_scheduler(optimizer: optim.Optimizer, cfg: DictConfig) -> Any | None:
    scheduler_type = str(cfg.trainer.get("scheduler", "cosine")).lower()
    if scheduler_type in {"none", "off", "false"}:
        return None
    if scheduler_type not in {"cosine"}:
        raise ValueError(
            f"trainer.scheduler must be one of: cosine, none. Got '{scheduler_type}'."
        )

    max_steps = int(cfg.trainer.max_steps)
    t_max = int(cfg.trainer.get("scheduler_t_max", max_steps))
    eta_min = float(cfg.trainer.get("scheduler_eta_min", 0.0))
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)


def _maybe_clip_grad_norm(
    parameters: Iterable[torch.nn.Parameter], max_norm: float
) -> None:
    if max_norm <= 0:
        return
    torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def _reduce_loss_for_logging(loss: torch.Tensor, world_size: int) -> float:
    loss_value = loss.detach()
    if dist.is_available() and dist.is_initialized():
        loss_value = loss_value.clone()
        dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
        if world_size > 0:
            loss_value = loss_value / float(world_size)
    return float(loss_value.item())


def _maybe_parse_betas(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, Sequence) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError("trainer.optimizer_betas must be a sequence of two floats.")


def _maybe_parse_eps(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


if __name__ == "__main__":
    pass
