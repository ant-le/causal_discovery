# BayesDAG External Environment Setup (pyenv)

BayesDAG (Project-BayesDAG / `causica`) targets Python 3.8.x and depends on
Torch + functorch versions that are not compatible with the main 3.11
environment. Use a dedicated pyenv virtualenv and run the model via the
external runner hook.

## 1. Create the pyenv environment

```bash
pyenv install 3.9.18
pyenv virtualenv 3.9.18 bayesdag-3.9
```

## 2. Install pinned dependencies

```bash
/Users/anton/.pyenv/versions/3.9.18/envs/bayesdag-3.9/bin/python -m pip install -r requirements-bayesdag.txt
```

## 3. Install Project-BayesDAG (causica)

The `src/pyproject.toml` expects a local `README.md`, so we install from a
local clone with a copied README:

```bash
tmpdir=$(mktemp -d) && \
  git clone --depth 1 https://github.com/microsoft/Project-BayesDAG.git "$tmpdir/Project-BayesDAG" && \
  cp "$tmpdir/Project-BayesDAG/README.md" "$tmpdir/Project-BayesDAG/src/README.md" && \
  /Users/anton/.pyenv/versions/3.9.18/envs/bayesdag-3.9/bin/python -m pip install --no-deps "$tmpdir/Project-BayesDAG/src"
```

## 4. Run the pipeline using the external env

```bash
causal-meta \
  model.type=bayesdag \
  model.num_nodes=10 \
  +model.pyenv_env=bayesdag-3.9 \
  +model.max_epochs=1 \
  +model.batch_size=16 \
  inference.n_samples=2
```

## Notes

- The external runner is `src/causal_meta/models/bayesdag/external_infer.py`.
- The internal wrapper calls the pyenv environment via `pyenv which python`.
- If you prefer a direct interpreter path, set `+model.external_python=/path/to/python`.
