#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBMIT_MODEL="${ROOT_DIR}/scripts/cluster/submit_model.sh"

# Auto-load BayesDAG external python from bootstrap snippet if available.
if [[ -z "${CAUSAL_META_BAYESDAG_PYTHON:-}" ]] && [[ -f "${ROOT_DIR}/.bootstrap_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${ROOT_DIR}/.bootstrap_env.sh"
fi

CONFIG_NAME="${CONFIG_NAME:-full_multimodel}"
RUN_PREFIX="${RUN_PREFIX:-rq1}"
MODELS=(avici bcnp dibs bayesdag)

for model in "${MODELS[@]}"; do
  run_name="${RUN_PREFIX}_${model}_only"
  "${SUBMIT_MODEL}" "${model}" "${CONFIG_NAME}" "${run_name}"
done
