# Runbook: Running Experiments

This guide describes how to execute causal discovery experiments using the `causal_meta` pipeline, exclusively relying on **Hydra** for configuration and job submission (including **Slurm** via Submitit).

## 1. Local Execution

To run a single experiment locally using the default configuration:

```bash
causal-meta
```

To run a specific packaged config:

```bash
causal-meta --config-name smoke_multimodel
```

## 2. Distributed & Cluster Execution (Slurm)

We use the `hydra-submitit-launcher` to submit jobs to a Slurm cluster. This handles both single jobs and parallel sweeps.

### Single Job on Slurm

To submit a single job to the cluster:

```bash
causal-meta --multirun \
  --config-name default \
  hydra/launcher=submitit_slurm \
  name=my_slurm_run
```

### Parallel Sweeps (Recommended)

To run multiple seeds or hyperparameter variations in parallel on the cluster:

```bash
causal-meta --multirun \
    --config-name smoke_multimodel \
    hydra/launcher=submitit_slurm \
    data.base_seed=0,1,2,3
```

### TU Wien VSC H100 preset

Use the dedicated launcher preset with the full multimodel sweep:

```bash
# Prevent eager CUDA initialization on login node (fixes pickling errors)
export CUDA_VISIBLE_DEVICES=""

causal-meta --multirun \
  --config-name full_multimodel \
  hydra/launcher=vsc_h100
```

This preset keeps the 4x H100 resource layout and sets per-job naming/output to
avoid collisions: each multirun job gets its own Slurm name (`cm_${model.id}`)
and its own Hydra sweep subdirectory (`${model.id}_${hydra.job.num}`).

### TU Wien VSC A100 preset

Use the A100 launcher preset (recommended for the RQ1 all-data sweep):

```bash
# Prevent eager CUDA initialization on login node (fixes pickling errors)
export CUDA_VISIBLE_DEVICES=""

causal-meta --multirun \
  --config-name full_multimodel \
  hydra/launcher=vsc_a100
```

This preset requests 1 A100 GPU per job and submits to the `GPU-a100` partition.
`full_multimodel` currently resolves to run name `rq1_all_data_multimodel`.

### TU Wien VSC A100s preset

Use the multi-GPU A100 launcher preset:

```bash
# Prevent eager CUDA initialization on login node (fixes pickling errors)
export CUDA_VISIBLE_DEVICES=""

causal-meta --multirun \
  --config-name full_multimodel \
  hydra/launcher=vsc_a100s
```

This preset requests 4 A100 GPUs per job (DDP).

### Per-job naming

You can override naming patterns directly when launching:

```bash
causal-meta --multirun \
  --config-name full_multimodel \
  hydra/launcher=vsc_h100 \
  hydra.launcher.name='cm_${model.id}' \
  hydra.sweep.subdir='${model.id}_${hydra.job.num}'
```

Example alternatives:

```bash
hydra.launcher.name='cm_${model.id}'
hydra.sweep.subdir='${model.id}'
```

### Resource Overrides

You can override Slurm resources directly from the command line:

```bash
causal-meta --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.partition=gpu_high \
  hydra.launcher.gpus_per_node=4 \
  hydra.launcher.tasks_per_node=4 \
  hydra.launcher.mem_gb=128
```

## 3. Environment & Setup

- **W&B Logging:** If using W&B (`logger.wandb.enabled=true`), set `WANDB_MODE=offline` on clusters without internet access.
- **Errors:** Use `HYDRA_FULL_ERROR=1` for detailed tracebacks.
- **Environment (uv, recommended):**
  ```bash
  uv lock
  uv sync --extra cluster --extra wandb --frozen --no-editable
  # If DiBS should use NVIDIA GPUs on CUDA 12 clusters:
  uv pip install --python .venv/bin/python --upgrade "jax[cuda12-local]"
  source .venv/bin/activate
  ```
- **BayesDAG External Env (required for multimodel):**
  ```bash
  ./bootstrap_uv.sh
  export CAUSAL_META_BAYESDAG_PYTHON="$PWD/.venv-bayesdag/bin/python"
  ```
  See `docs/BAYESDAG_SETUP.md` for details.

## 4. Output Structure

Hydra manages output directories:

- `experiments/runs/${name}/`: Single run directory and multirun aggregate.
- Each directory contains:
  - `checkpoints/`: Model weights (`best_<model_name>.pt`, `last_<model_name>.pt`).
  - `results/`: `<model_name>.json` and `aggregated.json`.
  - `inference/<model_name>/`: Sampled graphs (`seed_<seed>.pt` or `seed_<seed>.pt.gz`).
  - `pipe_<model_name>.log`: Job logs.

**Note:** If you sweep different hyperparameters for the same model, use different `name` values
to avoid overwriting artifacts in a shared run folder.

## 5. Validation And Baseline Profiles

- Validation uses multiple ID families (`id_*`) and one OOD family (`ood_*`) to
  track robustness during pre-training.
- DiBS and BayesDAG use YAML-driven profile overrides (Linear/NeuralNet/GPCDE)
  selected automatically per evaluation family.
- For reporting, separate your summary into:
  - ID slice (`id_*`) for paper-comparable metrics.
  - OOD slice (`ood_*`) for robustness checks.
