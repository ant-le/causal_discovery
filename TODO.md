# Cluster Suitability & Workflow Improvements

## 1. Validation & Smoke Testing
- [ ] **Verify End-to-End Pipeline**
    - [ ] Create and run a minimal `smoke_test.yaml` config (small model, few iterations).
    - [ ] Verify `results/metrics_summary.json` and `results/metrics_raw.json` are produced correctly.

## 2. Training Stability & Resumption (High Priority)
- [ ] **Implement Resume Logic (`src/causal_meta/runners/tasks/pre_training.py`)**
    - [ ] Detect `last.pt` automatically.
    - [ ] Load `model_state_dict`, `optimizer_state_dict`, and `step`.
    - [ ] **Crucial:** Advance the `MetaIterableDataset` stream to the correct state to avoid training on the exact same data sequence again (e.g., offset `base_seed` by `current_step * batch_size` or implement `advance()` in dataset).
- [ ] **Fix DDP Best Model Loading (`src/causal_meta/runners/tasks/pre_training.py`)**
    - [ ] Ensure `model.load_state_dict(...)` is executed on **ALL** ranks, not just inside the `if rank == 0:` block.
    - [ ] Verify `map_location` handles local rank correctly for all processes.

## 3. Inference Architecture (Bayesian/MCMC Support)
- [ ] **Implement Explicit Inference Workflow (`src/causal_meta/runners/tasks/inference.py`)**
    - [ ] Implement the `run()` function for Bayesian/MCMC models.
    - [ ] Save posterior samples/artifacts to `experiments/<name>/inference/`.
- [ ] **Update Evaluation to Consume Artifacts (`src/causal_meta/runners/tasks/evaluation.py`)**
    - [ ] Modify `run()` to check for existing artifacts in `experiments/<name>/inference/` before calling `model.sample()`.
    - [ ] Ensure Amortized models continue to use direct sampling.

## 4. Benchmark Completeness (OOD & Shifts)
- [ ] **Implement the documented training curriculum + OOD audit suite (`src/causal_meta/datasets/initialisation.md`)**
    - [ ] Add missing mechanism families: `GPCDE`, `Square`, `Periodic`, `LogisticMap`, `PostNonlinear (PNL)` as configurable factories.
    - [ ] Ensure mixtures match the intended proportions for training and are selectable per-node/per-graph as needed.
    - [ ] Add any missing dependencies as optional extras (e.g., `gpytorch` if used for GPCDE).
- [ ] **Add intervention support in SCM sampling (for I-NIL)**
    - [ ] Extend `SCMInstance`/dataset layer to sample under `do(...)` interventions.
    - [ ] Add fixed interventional test sets (seeded) alongside observational test sets.

## 5. Rigorous Reporting & Analysis
- [ ] **Freeze and version benchmark definitions**
    - [ ] Store explicit seed registries (train stream base seed + val/test seed lists) and family configs in each run directory.
- [ ] **Add robust metrics**
    - [ ] Implement NIL + I-NIL (predictive / interventional likelihood) where applicable.
    - [ ] Report posterior uncertainty summaries (e.g., edge entropy) to contextualize OOD degradation.
- [ ] **Meta-Analysis Aggregation**
    - [ ] Add an aggregator script to turn multiple runs (JSON results) into LaTeX tables and Plots with confidence intervals.
- [ ] **Fairness and reliability**
    - [ ] Multi-seed evaluation (multiple training seeds + multiple dataset seed lists).
    - [ ] Enforce comparable compute budgets across model families (time/steps/samples) and log wall-clock + peak memory.