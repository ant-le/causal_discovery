# Goals

- Ship the `src/` code as an installable package for conda, pip, and uv.

# Recommended workflow (high level)

- Treat `src/` as a reusable library (e.g., `causal_meta`) with clear subpackages: `datasets/`, `models/`, and `pipeline/`.
- Use a single experiment driver that consumes structured configs (YAML or pydantic) to set up SCM families, models, training/inference, and evaluation.
- Store reproducible artifacts under `artifacts/{model}/{run_id}/` and datasets under `datasets/data/{family}/{split}/{seed}/`.
- Keep observational and interventional data generation deterministic via seeded `SCMFamily` and `SCMInstance` objects.
- Expose a thin CLI entrypoint (e.g., `python -m causal_meta.run --config experiments/foo.yml`) and add tests under `tests/` for core components.

# Implementation structure (suggested)

- `datasets/`
  - `families.py`: definitions for i.d. and multiple o.o.d. SCM families (graph priors, mechanism types, noise choices) plus distance metrics between families/sets.
  - `scm.py`: `SCMInstance` with graph, mechanisms, noise; supports observational/interventional sampling and metadata export.
  - `dataset_spec.py`: config objects describing splits, sizes, and interventions; handles caching of generated test sets.
- `models/`
  - `base.py`: `BaseCausalModel` with `train()`, `infer_graph_posterior()`, optional `infer_mechanism_posterior()`, and a `trainable` flag.
  - Subfolders `bncp/`, `dibs/`, `bayesdag/`, `avici/` sharing common components (encoders/decoders) where possible.
- `pipeline/`
  - `pipe.py` or `runner.py`: orchestrates data generation/loading, model setup, training/inference, evaluation metrics, and logging; assigns devices for parallel runs.
  - `train_metrics.py` / `eval_metrics.py`: E-SHD, AUROC, NIL/I-NIL, plus posterior entropy for qualitative analysis.
- `experiments/` (new): YAML configs describing dataset family, model, hyperparams, and metrics for each run.
- `pyproject.toml`: single source of truth for packaging; use extras for GPU/ML dependencies if needed.

# Gemini alternative (config-first variant)

- Same package idea but more opinionated: a declarative experiment runner (`ExperimentRunner`) that reads YAML, instantiates `SCMFamily`, model module, and metrics, and logs via a standard logger (e.g., MLflow/W&B). This keeps scripts minimal (`scripts/run_experiment.py`, `scripts/analyze_results.py`).
