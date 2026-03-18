# Runbook: Running Experiments

This guide uses `uv` + Hydra for configuration and per-model Slurm scripts in `scripts/`.

## 1. Environment Model

Use two environments:

- `.venv`: main stack (Avici, BCNP, DiBS)
- `.venv-bayesdag`: BayesDAG legacy stack

## 2. One-Time Setup

```bash
scripts/setup_cluster.sh
```

## 3. Daily Update After Pull

```bash
uv sync --extra cluster --extra wandb --frozen --no-editable
uv pip install --python .venv-bayesdag/bin/python -r requirements-bayesdag.txt
```

Note: do not use `uv pip sync` for `.venv-bayesdag`; it removes packages not
listed in `requirements-bayesdag.txt` (including `causica`, installed from
Project-BayesDAG source by `bootstrap_uv.sh`).

GPU-only default notes:

- `bootstrap_uv.sh` installs CUDA JAX (`jax[cuda12-local]`) by default.

## 4. Local Runs

```bash
uv run causal-meta --config-name default
uv run causal-meta --config-name canary_multimodel model=avici
```

## 5. Cluster Runs (No Submitit)

Each model has a dedicated launcher script with hardcoded GPU specs:

- `scripts/run_avici.sh`: 4x A100
- `scripts/run_bcnp.sh`: 4x A100
- `scripts/run_dibs.sh`: 1x A100
- `scripts/run_bayesdag.sh`: 1x A100

Submit one model:

```bash
scripts/run_bcnp.sh
```

Submit all four models:

```bash
scripts/run_all_models.sh
```

Submit a fast BCNP multi-GPU smoke test:

```bash
scripts/run_bcnp_smoke_multigpu.sh
```

Optional arguments for each script:

```bash
scripts/run_avici.sh <config_name> <run_name> [hydra_overrides...]
```

For BayesDAG, `CAUSAL_META_BAYESDAG_PYTHON` defaults to
`.venv-bayesdag/bin/python` when unset.

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
- `metrics.json`
- `inference/`
- `main.log`
- `slurm_<jobid>.out`, `slurm_<jobid>.err`

## 8. Analysis from Run IDs

Generate tables/figures directly from selected run directories:

```bash
uv run python -m causal_meta.analysis.run_analysis experiments/runs \
  --run-id canary_20260317_120000_avici \
  --run-id canary_20260317_120000_bcnp
```

If no `--run-id`/`--run-dir` is provided, analysis discovers all `metrics.json`
under the supplied runs root.
