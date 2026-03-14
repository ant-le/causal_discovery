# Causal Meta-Learning Benchmarks

A scalable, Hydra-configured framework for benchmarking Bayesian Causal Discovery and Meta-Learning algorithms.

## Features

- **Meta-Learning Focus:** Infinite streaming datasets (`MetaIterableDataset`) with rank-aware seeding for massive parallel training.
- **Evaluation:** Strict O.O.D. generalization tests with disjoint graph hashing and cached inference artifacts.
- **Scalability:** Distributed Data Parallel (DDP) support, preemption-safe checkpointing, and cluster-ready per-model Slurm scripts.
- **Metrics:** Comprehensive graph metrics (SHD, SID, F1, AUROC) and likelihood proxies.

## Quick Start

### Installation (uv)

```bash
uv sync --extra cluster --extra wandb --frozen --no-editable

# If DiBS should use NVIDIA GPUs on CUDA 12 clusters:
uv pip install --python .venv/bin/python --upgrade "jax[cuda12-local]"
```

Use `--no-editable` for robust cross-platform imports of `causal_meta`.

For full multimodel runs including BayesDAG, use:

```bash
./bootstrap_uv.sh
```

Cluster scripts auto-detect BayesDAG Python from `.bootstrap_env.sh` or
`.venv-bayesdag/bin/python`.

### Running a Smoke Test

```bash
# Run a minimal local test
uv run causal-meta name=smoke_test
```

Smoke configs use online Weights & Biases logging. Ensure `wandb login` has been
run in your environment.

### Running on a Cluster (Slurm)

```bash
scripts/run_bcnp.sh
```

Run all models with:

```bash
scripts/run_avici.sh
scripts/run_bcnp.sh
scripts/run_dibs.sh
scripts/run_bayesdag.sh
```

## Documentation

- [Runbook](docs/RUNBOOK.md): Detailed guide on running experiments and sweeps.
- [Design](docs/DESIGN.md): Architectural overview.
