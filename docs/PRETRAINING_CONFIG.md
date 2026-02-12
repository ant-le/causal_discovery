# Pre-Training Configuration (Current)

This documents the _current_ pre-training setup as implemented by the runner code, and the canonical Hydra configs used for pre-training runs.

## Where Pre-Training Lives

- Entrypoint: `src/causal_meta/runners/pipe.py`
  - Builds `CausalMetaModule` from `cfg.data`
  - Builds a model from `cfg.model` via `ModelFactory`
  - If `model.needs_pretraining == True`:
    - calls `src/causal_meta/runners/tasks/pre_training.py::run`
  - Otherwise:
    - calls `src/causal_meta/runners/tasks/inference.py::run`
  - Always runs evaluation afterwards:
    - `src/causal_meta/runners/tasks/evaluation.py::run`

## Config Files That Matter

### Minimal default (installed usage / smoke)

- `src/causal_meta/runners/configs/default.yaml`
  - Small ER+Linear toy setup, `trainer.max_steps=10`, WandB disabled.

### Canonical “competence curriculum” pre-training (Avici)

- `experiments/benchmark_suite/train_competence.yaml`
  - Intended long pre-training run for `model.type=avici`.

### Canonical audit suites (reuse competence training, change tests)

- `experiments/benchmark_suite/full_audit.yaml`
- `experiments/benchmark_suite/audit_functional.yaml`
- `experiments/benchmark_suite/audit_structural.yaml`
- `experiments/benchmark_suite/audit_pnl.yaml`
- `experiments/benchmark_suite/audit_chaos.yaml`

### BCNP local “large” run

- `experiments/benchmark_suite/bcnp_local_large.yaml`
  - `defaults: [full_audit, _self_]` so it inherits `train_competence.yaml` and replaces model/trainer bits.

### BCNP paper “All Data” run

- `experiments/benchmark_suite/bcnp_paper_all_data.yaml`
  - Dataset mixture follows the BCNP "All Data" curriculum from `src/causal_meta/datasets/initialisation.md`.
  - Includes explicit permutation settings (`n_perm_samples`, `sinkhorn_iter`, `q_before_l`).

## Data Configuration (What Pre-Training Sees)

Data is built by `src/causal_meta/datasets/data_module.py` from `cfg.data` (parsed by `src/causal_meta/datasets/generators/factory.py`).

### Train stream (infinite)

- Source: `CausalMetaModule.train_dataloader()`
- Dataset type: `MetaIterableDataset` (infinite stream)
- Loop control: `trainer.max_steps` (there is no epoch concept)
- Batch size: always `1` task per step (one SCM task per batch)
- Key knobs:
  - `data.train_family`: graph distribution + mechanism distribution
  - `data.base_seed`: seed for the infinite stream
  - `data.samples_per_task`: number of samples _within_ each SCM task
  - `data.num_workers`, `data.pin_memory`
  - `data.normalize_data` exists in `DataModuleConfig` and defaults to `True` if not set in YAML.

### Validation (fixed)

- Source: `CausalMetaModule.val_dataloader()`
- Dataset type: `MetaFixedDataset` keyed by `data.seeds_val`
- If `data.val_families` is omitted/empty, validation defaults to an “id” family equal to `train_family`.

### Test (fixed, used in evaluation phase)

- Source: `CausalMetaModule.test_dataloader()`
- Dataset type: `MetaFixedDataset` keyed by `data.seeds_test`
- Audit configs mostly override `data.test_families` to define OOD test suites.

## Model Configuration (Pre-Training Targets)

Pre-training is triggered by `BaseModel.needs_pretraining == True`.

- Avici: `src/causal_meta/models/avici/model.py` (`model.type=avici`)
- BCNP: `src/causal_meta/models/bcnp/model.py` (`model.type=bcnp`)

### Avici (train_competence defaults)

From `experiments/benchmark_suite/train_competence.yaml`:

- `num_nodes: 20`
- `d_model: 128`, `nhead: 8`, `num_layers: 6`, `dim_feedforward: 512`, `dropout: 0.1`

### BCNP (bcnp_local_large overrides)

From `experiments/benchmark_suite/bcnp_local_large.yaml`:

- `num_nodes: 20`
- `d_model: 128`, `nhead: 8`
- `num_layers: 4`, `num_layers_decoder: 4`, `dim_feedforward: 512`, `dropout: 0.1`
- `emb_depth: 2`, `use_positional_encoding: false`
- `n_perm_samples: 10`, `sinkhorn_iter: 20`, `q_before_l: true`

## Trainer Configuration (What the Pre-Training Loop Uses)

Implemented in `src/causal_meta/runners/tasks/pre_training.py`.

### Optimizer

- AdamW: `torch.optim.AdamW(model.parameters(), lr=trainer.lr, weight_decay=trainer.weight_decay(default=1e-4))`

### LR schedule

- Cosine annealing:
  - `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trainer.max_steps)`
  - scheduler stepped every training step.

### Mixed precision

- Controlled by:
  - `trainer.amp` (bool)
  - `trainer.amp_dtype` in `{bf16, fp16}` (default in code is `"bf16"`)
- Effective only on CUDA devices.
- BF16 path disables GradScaler; FP16 uses GradScaler.

### Gradient clipping

- Configurable via `trainer.grad_clip_norm` (disabled when `<= 0`).

### Logging cadence

- `trainer.log_every_n_steps` (default 100 in code if unset)
- In DDP, loss is averaged across ranks before logging.

### Validation cadence

- `trainer.val_check_interval` (default 1000 in code if unset)
- Validation sampling count uses **inference** config:
  - `val_n_samples = cfg.inference.n_samples` (default fallback is 10 in code)

### Regulariser update cadence (Avici dual update hook)

- `trainer.regulariser_update_interval` (default `0` in code)
- If `> 0`, passes `update_regulariser=True` into `model.calculate_loss(...)` at that interval.
  - In Avici, this controls updates to the acyclicity dual weight.
  - Note: Avici’s code comment says “Should update every 250 steps”, but the _runner default is 0_, so it never updates unless set in config.

### Scheduler

- Configurable via `trainer.scheduler` (default: `cosine`).
- Cosine settings: `trainer.scheduler_t_max`, `trainer.scheduler_eta_min`.

### Optimizer constants

- AdamW betas: `trainer.optimizer_betas` (optional, `[beta1, beta2]`).
- AdamW epsilon: `trainer.optimizer_eps` (optional).

## Checkpointing / Resume

Implemented in `src/causal_meta/runners/tasks/pre_training.py`.

### Files

- Stored under `<output_dir>/checkpoints/`
  - `last_<model_name>.pt`: periodic + end-of-run checkpoint
  - `best_<model_name>.pt`: best-by-validation checkpoint

### What is saved

- model state dict
- optimizer state dict
- scheduler state dict
- AMP scaler state dict
- `step`
- training stream metadata:
  - `train_stream_initial_base_seed`
  - `train_stream_world_size`
  - `train_stream_next_base_seed_if_num_workers_0`

### Resume behavior

- If `<output_dir>/checkpoints/last_<model_name>.pt` exists:
  - loads model/optimizer/scheduler/scaler
  - resumes `step`
  - attempts “deterministic-ish” resumption of the infinite stream **only** when `data.num_workers == 0` by setting:
    - `data_module.train_dataset.base_seed = initial_base_seed + step * world_size`
  - warns if resuming with `num_workers > 0` (stream is not strictly reproducible)

### “Best” selection metric

- Pre-training validation computes graph metrics and aliases:
  - `mean_e-edgef1` (currently just equals `e-edgef1`)
- `best.pt` is updated when `mean_e-edgef1` improves.

### Post-train reload

- At the end, if `best_<model_name>.pt` exists, the model reloads it before returning to `pipe.py` (so the following evaluation uses best weights).

## Distributed (DDP) Notes

- DDP wrapping is done in `src/causal_meta/runners/pipe.py` using `torch.nn.parallel.DistributedDataParallel`.
- Rank 0:
  - prints/logs most status
  - writes checkpoints
- Validation metrics aggregation:
  - uses `Metrics` which gathers history across ranks via `torch.distributed.all_gather_object` (so metric means are across all ranks’ shards).

## Current Pre-Training Data Recipes (Canonical)

### `train_competence.yaml` (Avici)

- Train family:
  - `n_nodes: 20`
  - graph = mixture: ER(sparsity=0.2) + SF(m=2) with weights `[0.5, 0.5]`
  - mechanisms = mixture: Linear(weight_scale=1.0, noise_concentration=2.0) + MLP(hidden_dim=32) + GP(rff_dim=256, length_scale_range=[0.5, 2.0])
- Validation family:
  - ER(sparsity=0.2) + Linear
- `samples_per_task: 256`
- `seeds_val: [1000..1004]`, `seeds_test: [2000..2004]`
- `num_workers: 4`, `pin_memory: true`
- Trainer:
  - `lr: 1e-4`, `max_steps: 100000`
  - `log_every_n_steps: 100`, `val_check_interval: 1000`
- Inference (also used for validation sampling count):
  - `n_samples: 50`

### `bcnp_local_large.yaml` (BCNP)

- Inherits competence training + full audit test suites, then overrides:
  - `trainer.max_steps: 20000`
  - `trainer.val_check_interval: 2000`
  - `trainer.checkpoint_every_n_steps: 2000`
  - `data.num_workers: 0`, `pin_memory: false`
  - `inference.n_samples: 20`

### `bcnp_paper_all_data.yaml` (BCNP)

- Paper-style mixture curriculum (ER multi-density + SF) and balanced mechanism mix.
- Includes explicit permutation settings and cosine schedule defaults.

## Where the Training Script Can Be Improved (Concrete Gaps)

1. **Config drift / unused config**
   - `cfg.objective` is present in YAMLs (e.g. `objective.name=bce`) but the runner never uses it; loss is entirely model-defined via `model.calculate_loss(...)`.
   - `src/causal_meta/runners/configs/default.yaml` includes `inference.inil_graph_samples` but there is no runtime usage of that key in `src/`.

2. **Acyclicity regulariser schedule likely misconfigured**
   - Avici’s dual update is gated by `trainer.regulariser_update_interval`.
   - Runner default is `0` (never updates), while Avici’s code comment suggests ~250-step updates.
   - Improvement: set a sensible default (e.g. 250) for Avici runs, and log the dual weight + cyclicity value explicitly.

3. **Validation sampling is coupled to evaluation sampling**
   - Validation uses `cfg.inference.n_samples`, which is often expensive (e.g. 50) for frequent validation.
   - Improvement: add `trainer.val_n_samples` (separate from test-time `inference.n_samples`) and optionally cap number of validation tasks per check.

4. **Logging lacks key signals**
   - Only `train/loss` is logged.
   - Improvement: also log `lr`, grad norm, throughput, and (for Avici) cyclicity + regulariser weight.

5. **Checkpointing robustness**
   - Pre-training uses `torch.save` directly (not atomic); evaluation uses atomic save utilities elsewhere.
   - Improvement: switch pre-training checkpoints to atomic writes, and optionally keep a rolling history (`last_k`).

6. **Resume reproducibility**
   - Stream resume is only “deterministic-ish” when `num_workers==0`.
   - Improvement: optionally store/restore RNG states (`torch`, `cuda`, `numpy`, `random`) and document that exact resumption with multi-worker iterable streams is not guaranteed.

7. **Docs mismatch**
   - `src/causal_meta/runners/architecture.md` describes an `ObjectiveFactory` and says validation uses `test_dataloader()`, but the code uses `val_dataloader()` and no objective factory.
   - Improvement: align docs to code (or refactor code to match the documented design).

## How To Run (Examples)

- Local competence pre-training:
  - `causal-meta --config-path experiments/benchmark_suite --config-name train_competence`

- BCNP local large:
  - `causal-meta --config-path experiments/benchmark_suite --config-name bcnp_local_large`

- Slurm multirun:
  - `causal-meta --multirun --config-path experiments/benchmark_suite --config-name train_competence hydra/launcher=submitit_slurm`
