import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.amp import GradScaler

from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.models.base import BaseModel
from causal_meta.runners.metrics.graph import Metrics
from causal_meta.runners.utils.distributed import DistributedContext
from causal_meta.runners.utils.seeding import get_experiment_seed

log = logging.getLogger(__name__)


def run(
    cfg: DictConfig,
    model: BaseModel,
    data_module: CausalMetaModule,
    *,
    logger=None,
    output_dir: Path | None = None,
):
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

    # Unwrap model for attribute access
    model_unwrapped = model.module if is_distributed else model

    if not model_unwrapped.needs_pretraining:
        if rank == 0:
            log.info("Model does not require pre-training. Exiting.")
        return

    # 3. Optimizer
    lr = float(cfg.trainer.lr)
    weight_decay = float(cfg.trainer.get("weight_decay", 1e-4))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.trainer.max_steps)

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
    max_steps = cfg.trainer.max_steps
    log_every_n_steps = cfg.trainer.get("log_every_n_steps", 100)
    val_check_interval = cfg.trainer.get("val_check_interval", 1000)
    regulariser_update_interval = cfg.trainer.get("regulariser_update_interval", 0)

    best_val_metric = float("-inf")
    step = 0

    if rank == 0:
        log.info("Starting training loop...")

    # Determine output directory
    if output_dir is None:
        try:
            output_dir = Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            output_dir = Path(os.getcwd())
    else:
        output_dir = Path(output_dir)

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
        if "scheduler_state_dict" in checkpoint:
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

        if getattr(cfg.data, "num_workers", 0) > 0:
            log.warning(
                "Resuming with an IterableDataset stream is not strictly reproducible when num_workers>0; "
                "the data stream may differ from the original run."
            )

    # Deterministic-ish resume for the streaming dataset when DataLoader uses num_workers=0.
    # With num_workers=0, MetaIterableDataset uses a single stream per rank with stride=world_size.
    if (
        hasattr(data_module, "train_dataset")
        and data_module.train_dataset is not None
        and getattr(cfg.data, "num_workers", 0) == 0
    ):
        data_module.train_dataset.base_seed = (
            train_stream_initial_base_seed + step * world_size
        )
        if rank == 0 and start_step > 0:
            log.info(
                f"Resumed train stream base_seed={data_module.train_dataset.base_seed} "
                f"(initial={train_stream_initial_base_seed}, step={step}, world_size={world_size})."
            )

    checkpoint_every_n_steps = int(
        cfg.trainer.get("checkpoint_every_n_steps", val_check_interval)
    )

    train_iter = iter(train_loader)

    while step < max_steps:

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_data = batch["data"].to(device)
        adjacency_matrix = batch["adjacency"].to(device)

        # Forward
        model.train()
        optimizer.zero_grad()
        if amp_enabled:
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                output = model(input_data)
        else:
            output = model(input_data)

        # Loss
        next_step = step + 1
        update_regulariser = (
            regulariser_update_interval > 0
            and next_step % regulariser_update_interval == 0
        )

        # Call calculate_loss on the unwrapped model
        # The output tensor from DDP forward is on the correct device
        if amp_enabled:
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                loss = model_unwrapped.calculate_loss(
                    output,
                    adjacency_matrix,
                    update_regulariser=update_regulariser,
                )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        else:
            loss = model_unwrapped.calculate_loss(
                output, adjacency_matrix, update_regulariser=update_regulariser
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        step += 1

        # Logging
        if step % log_every_n_steps == 0:
            if rank == 0:
                log_str = f"Step {step}/{max_steps} | Loss: {loss.item():.4f}"
                log.info(log_str)

                if logger:
                    logger.log_metrics({"train/loss": loss.item()}, step=step)

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
                    logger.log_metrics(
                        {f"val/{k}": v for k, v in val_metrics.items()}, step=step
                    )

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


def validate(model, data_module, device, num_samples: int = 10):
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
    scheduler: Any,
    scaler: Any,
    step: int,
    filepath: Path,
    *,
    train_stream_initial_base_seed: int,
    world_size: int,
) -> None:
    experiment_seed = get_experiment_seed(
        cfg, fallback=int(train_stream_initial_base_seed)
    )
    torch.save(
        {
            "step": step,
            "experiment_seed": int(experiment_seed),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_stream_initial_base_seed": int(train_stream_initial_base_seed),
            "train_stream_world_size": int(world_size),
            "train_stream_next_base_seed_if_num_workers_0": int(
                train_stream_initial_base_seed + step * world_size
            ),
        },
        filepath,
    )


if __name__ == "__main__":
    pass
