#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "pyproject.toml" ]]; then
  echo "Error: run this script from the repository root." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

echo "==> Installing Python runtimes with uv"
uv python install 3.11 3.9

echo "==> Creating main environment (.venv, Python 3.11)"
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[cluster,wandb]"

echo "==> Creating BayesDAG environment (.venv-bayesdag, Python 3.9)"
uv venv .venv-bayesdag --python 3.9
uv pip install --python .venv-bayesdag/bin/python -r requirements-bayesdag.txt

echo "==> Installing Project-BayesDAG (causica) into .venv-bayesdag"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
git clone --depth 1 https://github.com/microsoft/Project-BayesDAG.git "$tmp_dir/Project-BayesDAG"
cp "$tmp_dir/Project-BayesDAG/README.md" "$tmp_dir/Project-BayesDAG/src/README.md"
uv pip install --python .venv-bayesdag/bin/python --no-deps "$tmp_dir/Project-BayesDAG/src"

echo
echo "Setup complete."
echo "Main env python:      $ROOT_DIR/.venv/bin/python"
echo "BayesDAG env python:  $ROOT_DIR/.venv-bayesdag/bin/python"
echo
echo "Use this before runs that include BayesDAG:"
echo "  export CAUSAL_META_BAYESDAG_PYTHON=\"$ROOT_DIR/.venv-bayesdag/bin/python\""
echo
echo "Run local smoke test:"
echo "  $ROOT_DIR/.venv/bin/python -m causal_meta.main name=smoke_test"
