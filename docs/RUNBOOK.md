# Runbook: Running Experiments

This guide describes how to execute causal discovery experiments using the `causal_meta` pipeline, exclusively relying on **Hydra** for configuration and job submission (including **Slurm** via Submitit).

## 1. Local Execution

To run a single experiment locally using the default configuration:

```bash
causal-meta
```

To run a specific YAML config from the repo:

```bash
causal-meta --config-path experiments/examples --config-name smoke_test
```

## 2. Distributed & Cluster Execution (Slurm)

We use the `hydra-submitit-launcher` to submit jobs to a Slurm cluster. This handles both single jobs and parallel sweeps.

### Single Job on Slurm
To submit a single job to the cluster:

```bash
causal-meta --multirun \
  --config-path experiments/benchmark_suite --config-name train_competence \
  hydra/launcher=submitit_slurm \
  name=my_slurm_run
```

### Parallel Sweeps (Recommended)
To run multiple seeds or hyperparameter variations in parallel on the cluster:

```bash
causal-meta --multirun \
    --config-path experiments/benchmark_suite --config-name train_competence \
    hydra/launcher=submitit_slurm \
    data.base_seed=0,1,2,3 \
    trainer.lr=0.001,0.0001
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
- **Environment:**
  ```bash
  conda env create -f environment.yml
  conda activate causal_meta
  pip install -e .
  ```

## 4. Output Structure

Hydra manages output directories:
- `experiments/${name}/${date}/${time}/`: Single run directory.
- `experiments/${name}/multirun/${date}/${time}/`: Sweep/Slurm directory.
  - Subdirectories `0/`, `1/`, etc., contain results for each job in the sweep.
- Each directory contains:
  - `checkpoints/`: Model weights.
  - `results/`: `metrics_summary.json` and `metrics_raw.json`.
  - `inference/`: Sampled graphs.
  - `pipe.log`: Job logs.