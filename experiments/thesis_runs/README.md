# Curated Thesis Runs

This folder is the manual entry point for the thesis analysis pipeline.

Expected layout:

```text
experiments/thesis_runs/
  avici/
  bcnp/
  dibs/
  bayesdag/
```

Each model folder should contain exactly the run artifacts that should be used
for the thesis, for example by copying or symlinking the selected cluster run.

Minimum required contents per model folder:

- `metrics.json`
- `inference/` (needed for posterior diagnostics)
- `main.log`

The thesis analysis command reads these four folders directly and rebuilds
`paper/final_thesis/generated/` from them.
