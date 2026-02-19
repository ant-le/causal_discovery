# BayesDAG External Environment Setup (uv)

BayesDAG (Project-BayesDAG / `causica`) depends on an older Python and Torch
stack that is not compatible with the main project environment. Run it through
the external interpreter hook.

## Quick setup (recommended)

From the repository root:

```bash
./bootstrap_uv.sh
export CAUSAL_META_BAYESDAG_PYTHON="$PWD/.venv-bayesdag/bin/python"
```

This creates:

- `.venv` (Python 3.11) for the main project.
- `.venv-bayesdag` (Python 3.9) for `causica` and BayesDAG.

## Manual setup (equivalent)

```bash
uv python install 3.11 3.9
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[cluster,wandb]"

uv venv .venv-bayesdag --python 3.9
uv pip install --python .venv-bayesdag/bin/python -r requirements-bayesdag.txt

tmpdir=$(mktemp -d) && \
  git clone --depth 1 https://github.com/microsoft/Project-BayesDAG.git "$tmpdir/Project-BayesDAG" && \
  cp "$tmpdir/Project-BayesDAG/README.md" "$tmpdir/Project-BayesDAG/src/README.md" && \
  uv pip install --python .venv-bayesdag/bin/python --no-deps "$tmpdir/Project-BayesDAG/src"
```

## Run with the external BayesDAG interpreter

```bash
export CAUSAL_META_BAYESDAG_PYTHON="$PWD/.venv-bayesdag/bin/python"

python -m causal_meta.main \
  model.type=bayesdag \
  model.num_nodes=10 \
  +model.max_epochs=1 \
  +model.batch_size=16 \
  inference.n_samples=2
```

## Cluster notes

- Use an absolute interpreter path on shared storage for `CAUSAL_META_BAYESDAG_PYTHON`.
- Keep the main launcher in the 3.11 env (`.venv`) and let BayesDAG execute via
  the external interpreter path.
- The external runner is `src/causal_meta/models/bayesdag/external_infer.py`.
