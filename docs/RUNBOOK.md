# Runbook: Running Experiments

This guide describes how to execute causal discovery experiments using the `causal_meta` pipeline, exclusively relying on **Hydra** for configuration and job submission (including **Slurm** via Submitit).

## 1. Environment & Setup

### Core Project Environment (uv)

We use `uv` for dependency management.

```bash
uv lock
uv sync --extra cluster --extra wandb --frozen --no-editable
# If DiBS should use NVIDIA GPUs on CUDA 12 clusters:
uv pip install --python .venv/bin/python --upgrade "jax[cuda12-local]"
source .venv/bin/activate
```

After pulling new commits, rerun `uv sync --extra cluster --extra wandb --frozen --no-editable` to refresh the `causal_meta` installation.

### BayesDAG External Environment (required for multimodel)

BayesDAG depends on an older Python/Torch stack.

```bash
./bootstrap_uv.sh
export CAUSAL_META_BAYESDAG_PYTHON="$PWD/.venv-bayesdag/bin/python"
```

_Troubleshooting:_ If BayesDAG fails with `ModuleNotFoundError: No module named 'pkg_resources'`, pin setuptools:
`uv pip install --python .venv-bayesdag/bin/python "setuptools<81"`

### W&B Logging

If using W&B (`logger.wandb.enabled=true`), set `WANDB_MODE=offline` on clusters without internet access.

## 2. Configuration & Running Experiments

Pre-training is executed only for models with `needs_pretraining == True` (e.g., Avici, BCNP).

### Canonical Configs

- `default.yaml`: Minimal single-run smoke config.
- `smoke_multimodel.yaml`: Multirun sweep over smoke model variants.
- `full_multimodel.yaml`: Multirun sweep over RQ1 all-data models (`avici,bcnp,dibs,bayesdag`).

### Local Execution

_Smoke pre-training (single run):_

```bash
causal-meta --config-name default
```

_Smoke multirun (all smoke models):_

```bash
causal-meta --multirun --config-name smoke_multimodel
```

## 3. Distributed & Cluster Execution (Slurm)

We use `hydra-submitit-launcher` to submit jobs to Slurm.

### Cluster Submission

```bash
export CUDA_VISIBLE_DEVICES="" # Prevent eager CUDA init
source .bootstrap_env.sh        # Sets CAUSAL_META_BAYESDAG_PYTHON

# Full RQ1 sweep, single A100 (vsc_a100)
causal-meta --multirun --config-name full_multimodel hydra/launcher=vsc_a100

# 4× A100 DDP (vsc_a100s)
causal-meta --multirun --config-name full_multimodel hydra/launcher=vsc_a100s
```

### Resource & Naming Overrides

You can override Slurm resources and naming directly:

```bash
causal-meta --multirun \
  --config-name full_multimodel \
  hydra/launcher=vsc_a100 \
  hydra.launcher.name='cm_${model.id}' \
  hydra.sweep.subdir='${model.id}_${hydra.job.num}' \
  hydra.launcher.mem_gb=128
```

## 4. Monitoring & Profiling

### Validation Monitoring

The training loop logs per-family metrics and aggregate group metrics (`mean_id_e-edgef1`, `mean_id_auc`, and `mean_ood_e-edgef1`). Checkpoint selection uses `mean_id_e-edgef1`.

### Performance Profiling Checklist

1. **Data Throughput:** Monitor GPU usage (`nvidia-smi`). Set `num_workers > 0` and `pin_memory=True` in `config.data`.
2. **Distributed Overheads:** Use `val_check_interval >= 1000` steps for validation.
3. **Memory Usage:** Enable `tf32` (`trainer.tf32=true`) and Mixed Precision (`trainer.amp=true`).
4. **Artifact I/O:** Ensure `output_dir` is on a fast filesystem (e.g., scratch/SSD), not NFS. Use `cache_compress=true`.

## 5. Output Structure

Hydra manages output directories (`experiments/runs/${name}/`):

- `checkpoints/`: Model weights.
- `results/`: Model-specific and aggregated JSON results.
- `inference/<model_name>/`: Sampled graphs (compressed if configured).
- `pipe_<model_name>.log`: Job logs.
