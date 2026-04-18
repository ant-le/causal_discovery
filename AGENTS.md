# Agent Guidelines for Causal Meta

This repository is a Hydra-configured Python benchmarking framework under `src/causal_meta`.

## 1. Core Expectations & Main Tasks
- Keep code, experiments, and thesis text aligned.
- **Main Tasks**:
  1. Write/update code in `src/causal_meta/`.
  2. Write the thesis in `paper/final_thesis/`.
  3. Check the thesis literature in `paper/markdown/`.
  4. Analyse results in `experiments/thesis_runs/` using the `src/causal_meta/analysis/` scripts, which output directly into the thesis at `paper/final_thesis/graphics/`.
- **Key Reference Papers**:
  - AVICI: `paper/markdown/introduction/avici.md`
  - Meta-learning: `paper/markdown/methdology/meta_learning.md`
  - DiBS: `paper/markdown/methdology/dibs.md`
  - BayesDAG: `paper/markdown/methdology/bayesdag.md`
- Prefer small, local changes over broad rewrites.
- Respect Hydra/OmegaConf; put defaults in YAML, not inline in Python.
- Preserve reproducibility: seeds, dataset hashing, cached inference, and DDP behavior are part of the design.
- Assume the main package lives under `src/causal_meta` and tests mirror it under `tests/`.

## 2. Repo Facts
- Environment and package management use `uv`.
- Packaging is defined in `pyproject.toml` with Hatchling.
- CLI entry point: `causal-meta = causal_meta.main:main`.
- `pytest` sets `pythonpath = ["src"]`, so run Python tests from the repo root.
- No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` files were found.

## 3. Repo Layout
- `src/causal_meta/datasets`: SCM generation, dataset wrappers, collate utils.
- `src/causal_meta/models`: AviCi, BCNP, DiBS, BayesDAG, random baseline.
- `src/causal_meta/runners`: training, inference, evaluation, logging, metrics.
- `src/causal_meta/analysis`: thesis analysis organized around `rq1`, `rq2`, `rq3`, shared thesis utilities, and appendix generation. Scripts here output to `paper/final_thesis/graphics/`.
- `src/causal_meta/configs`: Hydra defaults and experiment config groups.
- `tests/`: pytest suite mirroring the Python package layout.
- `paper/final_thesis/`: thesis text that may need updates when logic changes. Contains `graphics/` for analysis output.
- `paper/markdown/`: thesis literature and reference papers.
- `experiments/thesis_runs/`: results to be analyzed.

## 4. Setup and Build Commands
### Python environment
```bash
uv sync --extra dev --extra cluster --extra wandb --frozen --no-editable
scripts/bootstrap_uv.sh
uv pip install --python .venv-bayesdag/bin/python -r requirements-bayesdag.txt
```
### Build and run
```bash
uv build
uv run causal-meta --config-name default
uv run causal-meta --config-name dg_2pretrain_smoke model=avici
export WANDB_MODE=offline

# General analysis (for experiments/runs)
uv run python -m causal_meta.analysis.run_analysis experiments/runs --strict

# Thesis analysis - USE --skip-posterior to avoid timeout from posterior graph loading
uv run python -m causal_meta.analysis.run_thesis_analysis --input-root experiments/thesis_runs --thesis-root paper/final_thesis --skip-posterior
uv run python -m causal_meta.analysis.run_appendix_generation --thesis-root paper/final_thesis --configs-root src/causal_meta/configs
```
### Python tests
```bash
uv run pytest
uv run pytest tests/runners/test_logger_integration.py
uv run pytest tests/runners/test_logger_integration.py::test_pipeline_integration_smoke
uv run pytest -k checkpointing tests/runners/test_pre_training_checkpointing.py
uv run pytest -v tests/analysis
```
- Use `uv run pytest path/to/test.py::test_name` as the default quick verification loop.
- Prefer running Python commands from the repo root.

## 5. Linting and Formatting Reality
- There is no repo-wide Python Ruff, Black, Flake8, or mypy config checked in.
- Follow the existing Python style manually instead of introducing formatter churn.
- Use black-like formatting: readable wraps, stable trailing commas, and compact diffs.
- Preserve existing formatting and section comments unless you are editing that code for a reason.

## 6. Python Style Guide
### Imports
- Group imports as standard library, third-party, then local package imports.
- Keep one blank line between import groups.
- Prefer explicit imports over wildcard imports.
- Avoid reorder-only import diffs in untouched files.
### Structure and formatting
- Use `from __future__ import annotations` in new or heavily edited modules.
- Keep functions small enough that data and config flow stay obvious.
- Prefer `pathlib.Path` over raw string paths.
- Add comments only for non-obvious invariants or tricky control flow.
- Preserve helpful existing section dividers in long modules.
### Types and annotations
- Add type hints for public functions, methods, constructors, and edited helpers.
- Be explicit about `torch.Tensor`, `np.ndarray`, Python scalars, and `Path`.
- Use `DictConfig` or `Mapping[str, Any]` at Hydra boundaries.
- In new code, prefer `dict[str, Any]` and `Foo | None`, but match surrounding files that still use `Dict` or `Optional`.
- Use `Protocol`, `TypedDict`, and `@dataclass` when they fit existing patterns.
- Use `cast(...)` sparingly and only at narrow boundaries.
### Naming
- Classes: `PascalCase`.
- Functions, methods, variables, modules: `snake_case`.
- Constants: `SCREAMING_SNAKE_CASE`.
- Private helpers and internal state: `_leading_underscore`.
- Config dataclasses usually end with `Config`.
- Module loggers are usually named `log`.
### Docstrings
- Public classes, public functions, and reusable helpers should have docstrings.
- Prefer Google-style `Args`, `Returns`, and `Raises` sections when useful.
- Document tensor shapes, batch semantics, and invariants when relevant.
- Tests only need docstrings when the scenario is not self-explanatory.

## 7. Error Handling and Logging
- Fail fast on invalid config or impossible runtime state.
- Use `ValueError` for invalid inputs/config, `RuntimeError` for invalid runtime state, and `KeyError` for missing named entries.
- Preserve context with `raise ... from exc` when translating exceptions.
- Catch broad exceptions only at boundaries and log enough context to debug.
- Use `try`/`finally` for temporary `cwd` changes, cleanup, and distributed teardown.
- Prefer `logging.getLogger(__name__)` over `print(...)`.
- In distributed code, gate noisy logs and artifact writes behind `dist_ctx.is_main_process` or rank checks.

## 8. Hydra, Data, and Model Conventions
- Keep experiment defaults in `src/causal_meta/configs`, not inline in Python.
- Reuse existing abstractions such as `BaseModel`, `CausalMetaModule`, and `ModelFactory` instead of bypassing them.
- Do not break worker-aware or rank-aware seeding in iterable datasets.
- Do not add non-picklable objects to DataLoader outputs.
- Preserve Hydra output structure and cached artifact layout unless migration is part of the task.
- Respect the current split between datasets, models, runners, analysis, and thesis artifacts.

## 9. Reproducibility Guardrails
- Keep training streams deterministic relative to configured seeds.
- Preserve graph hashing and train/validation/test disjointness logic.
- Be careful around cached inference paths and resume/checkpoint metadata.
- Do not quietly change run directory structure under `experiments/runs/`.
- If you alter distributed behavior, verify rank-aware logging and writes.

## 10. Testing Conventions
- Put tests in the matching area under `tests/`.
- Prefer focused `pytest` tests over heavy end-to-end additions.
- Use `tmp_path`, `monkeypatch`, `caplog`, `pytest.raises`, and `pytest.approx` the way existing tests do.
- Build small synthetic configs with `OmegaConf.create(...)` for runner tests.
- Keep smoke tests tiny: small models, few steps, low sample counts.
- When fixing a bug, add the narrowest test that proves the fix.

## 11. Thesis Synchronization
- Treat code and thesis text as one system.
- If you change logic in `src/`, check whether `paper/final_thesis/` also needs updates.
- Likely mapping:
  - research questions / framing -> `paper/final_thesis/sections/1_Introduction.tex`
  - related work / baselines -> `paper/final_thesis/sections/3_RelatedWork.tex`
  - methodology / algorithms -> `paper/final_thesis/sections/4_Methodology.tex`
  - experiments / results -> `paper/final_thesis/sections/5_Results.tex`
  - thesis-wide interpretation / limitations -> `paper/final_thesis/sections/6_Conclusion.tex`
- In your report, explicitly state which thesis sections may be affected.

## 12. Cluster and Release Notes
- Cluster launchers under `scripts/` are part of the repo's operational contract.
- If you change cluster, bootstrap, or distributed launch behavior, verify the full command path, not just local imports.
- For cluster-readiness changes, commit and push before claiming the workflow is ready for GPU execution.

## 13. Practical Defaults
- Start from the repo root for Python work.
- Preserve unrelated local worktree changes.
- Match surrounding file style before introducing new abstractions.
- If editor-specific instructions are added later, extend this file; none currently exist via Cursor or Copilot rules.
