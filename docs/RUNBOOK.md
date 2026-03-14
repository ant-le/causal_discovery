# Runbook: Running Experiments

This guide uses `uv` + Hydra for configuration and `sbatch` + `torchrun` for cluster execution.

## 1. Environment Model

Use two environments:

- `.venv`: main stack (Avici, BCNP, DiBS)
- `.venv-bayesdag`: BayesDAG legacy stack

## 2. One-Time Setup

```bash
./bootstrap_uv.sh
```

## 3. Daily Update After Pull

```bash
uv sync --extra cluster --extra wandb --frozen --no-editable
uv pip sync --python .venv-bayesdag/bin/python requirements-bayesdag.txt
```

Optional (DiBS on CUDA 12):

```bash
uv pip install --python .venv/bin/python --upgrade "jax[cuda12-local]"
```

## 4. Local Runs

```bash
uv run causal-meta --config-name default
uv run causal-meta --multirun --config-name smoke_multimodel
```

## 5. Cluster Runs (No Submitit)

Cluster submission is script-based.

- DDP models (`avici`, `bcnp`) run with `torchrun` on multi-GPU.
- Explicit models (`dibs`, `bayesdag`) run single-process/single-GPU.

Submit one model:

```bash
scripts/cluster/submit_model.sh bcnp full_multimodel rq1_bcnp_only
```

Note: the DDP submit path requests both typed GRES and `--gpus-per-task` to
avoid single-visible-GPU allocations on clusters with strict task GPU cgroups.

Submit full RQ1 model set (`avici,bcnp,dibs,bayesdag`):

```bash
CAUSAL_META_BAYESDAG_PYTHON="$PWD/.venv-bayesdag/bin/python" \
scripts/cluster/submit_rq1.sh
```

Common cluster overrides are environment variables:

```bash
PARTITION=GPU-a100 GPU_TYPE=a100 TIME_HOURS=72 DDP_GPUS=4 scripts/cluster/submit_model.sh avici
```

## 6. Comparability Checklist

Keep these fixed across models:

- `--config-name full_multimodel`
- same lockfile (`uv.lock`) + frozen sync
- same seeds (`data.base_seed`, evaluation seeds)
- same inference settings (`inference.n_samples`, AUC controls)
- same partition/GPU type/time budget class

If W&B online is unavailable:

```bash
export WANDB_MODE=offline
```

## 7. Output Structure

Run outputs are under `experiments/runs/${name}/`:

- `checkpoints/`
- `results/`
- `inference/<model_name>/`
- `main.log`
- `slurm_<jobid>.out`, `slurm_<jobid>.err`
