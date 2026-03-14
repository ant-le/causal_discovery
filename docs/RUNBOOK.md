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

GPU-only default notes:

- `bootstrap_uv.sh` installs CUDA JAX (`jax[cuda12-local]`) by default.
- Cluster scripts fail fast when CUDA is unavailable in main torch, DiBS JAX, or BayesDAG torch.
- Setup-time strict GPU checks are off by default (`CAUSAL_META_STRICT_GPU=0`) so login-node setup works.
- Optional strict setup-time checks: `CAUSAL_META_STRICT_GPU=1 ./bootstrap_uv.sh`.

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

Defaults are tuned for VSC A100s (`PARTITION=GPU-a100s`) with
`GPU_REQUEST_MODE=auto`.

GPU request style is configurable via `GPU_REQUEST_MODE`:

- `gpus-per-node`
- `gpus`
- `gres`

In `auto` mode, the script tries `gpus-per-node` -> `gpus` -> `gres`.

For clusters that reject `--gres`, auto mode will fall back automatically.

Submit full RQ1 model set (`avici,bcnp,dibs,bayesdag`):

```bash
scripts/cluster/submit_rq1.sh
```

The script auto-loads `.bootstrap_env.sh` and falls back to
`.venv-bayesdag/bin/python` for BayesDAG when available.

Common cluster overrides are environment variables:

```bash
PARTITION=GPU-a100 GPU_REQUEST_MODE=gpus-per-node TIME_HOURS=72 DDP_GPUS=4 scripts/cluster/submit_model.sh avici
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
