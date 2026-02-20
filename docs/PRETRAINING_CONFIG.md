# Pre-Training Configuration (Current)

This document reflects the current Hydra config layout in `src/causal_meta/configs`.

## Where Pre-Training Lives

- Entrypoint: `src/causal_meta/main.py`
- Training task: `src/causal_meta/runners/tasks/pre_training.py`
- Inference task: `src/causal_meta/runners/tasks/inference.py`
- Evaluation task: `src/causal_meta/runners/tasks/evaluation.py`

Pre-training is executed only for models with `needs_pretraining == True` (e.g., Avici, BCNP).

## Canonical Configs

Top-level configs:

- `src/causal_meta/configs/default.yaml`
  - Minimal single-run smoke config.
- `src/causal_meta/configs/smoke_multimodel.yaml`
  - Multirun sweep over smoke model variants.
- `src/causal_meta/configs/full_multimodel.yaml`
  - Multirun sweep over full models (`avici_full,bcnp_full,dibs,bayesdag`).

Config groups used by those top-level configs:

- Data: `src/causal_meta/configs/data/smoke.yaml`, `src/causal_meta/configs/data/full.yaml`
- Trainer: `src/causal_meta/configs/trainer/smoke.yaml`, `src/causal_meta/configs/trainer/long.yaml`
- Inference: `src/causal_meta/configs/inference/smoke.yaml`, `src/causal_meta/configs/inference/long.yaml`
- Model: `src/causal_meta/configs/model/*.yaml`

## Current Training Defaults

Smoke trainer (`trainer=smoke`):

- `max_steps: 30`
- `lr: 1e-3`
- `amp: false`

Long trainer (`trainer=long`):

- `max_steps: 500000`
- `lr: 1e-4`
- `amp: true` (`bf16`)

## How To Run

Local smoke pre-training (single run):

```bash
causal-meta --config-name default
```

Local smoke multirun (all smoke models):

```bash
causal-meta --multirun --config-name smoke_multimodel
```

Cluster full multirun (all full models):

```bash
export CUDA_VISIBLE_DEVICES=""
causal-meta --multirun --config-name full_multimodel hydra/launcher=vsc_a100
```

Cluster full multirun with 4 GPUs per job:

```bash
export CUDA_VISIBLE_DEVICES=""
causal-meta --multirun --config-name full_multimodel hydra/launcher=vsc_a100s
```

## BayesDAG Note

`full_multimodel` includes BayesDAG. Set:

```bash
export CAUSAL_META_BAYESDAG_PYTHON="$PWD/.venv-bayesdag/bin/python"
```

See `docs/BAYESDAG_SETUP.md` for full setup.

## Legacy Note

Older documentation that referenced `experiments/benchmark_suite/*` is obsolete.
The canonical source of truth is now `src/causal_meta/configs/*`.
