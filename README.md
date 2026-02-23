# Causal Meta-Learning Benchmarks

A scalable, Hydra-configured framework for benchmarking Bayesian Causal Discovery and Meta-Learning algorithms.

## Features

- **Meta-Learning Focus:** Infinite streaming datasets (`MetaIterableDataset`) with rank-aware seeding for massive parallel training.
- **Evaluation:** Strict O.O.D. generalization tests with disjoint graph hashing and cached inference artifacts.
- **Scalability:** Distributed Data Parallel (DDP) support, preemption-safe checkpointing, and cluster-ready Submitit integration.
- **Metrics:** Comprehensive graph metrics (SHD, SID, F1, AUROC) and likelihood proxies.

## Quick Start

### Installation (uv)

```bash
uv lock
uv sync --extra cluster --extra wandb --frozen --no-editable
source .venv/bin/activate

# If DiBS should use NVIDIA GPUs on CUDA 12 clusters:
uv pip install --python .venv/bin/python --upgrade "jax[cuda12-local]"
```

Use `--no-editable` for robust cross-platform imports of `causal_meta`.

For full multimodel runs including BayesDAG, use:

```bash
./bootstrap_uv.sh
export CAUSAL_META_BAYESDAG_PYTHON="$PWD/.venv-bayesdag/bin/python"
```

### Running a Smoke Test

```bash
# Run a minimal local test
causal-meta name=smoke_test
```

### Running on a Cluster (Slurm)

```bash
# Submit to Slurm via Submitit
causal-meta --multirun --config-name smoke_multimodel hydra/launcher=submitit_slurm
```

## Documentation

- [Runbook](docs/RUNBOOK.md): Detailed guide on running experiments and sweeps.
- [Design](docs/DESIGN.md): Architectural overview.
- [Class Structure](docs/CLASS_STRUCTURE.md): Key classes and data flow.
