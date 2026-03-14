#!/bin/bash
set -euo pipefail

MODEL="${1:?Usage: train_single.sbatch <model> [config_name] [run_name] [hydra_overrides...] }"
CONFIG_NAME="${2:-full_multimodel}"
RUN_NAME="${3:-rq1_${MODEL}_single}"

if [[ "$#" -ge 3 ]]; then
  shift 3
else
  shift "$#"
fi

if [[ "${MODEL}" == "bayesdag" ]] && [[ -z "${CAUSAL_META_BAYESDAG_PYTHON:-}" ]]; then
  echo "CAUSAL_META_BAYESDAG_PYTHON is required for model=bayesdag." >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1

uv run causal-meta \
  --config-name "${CONFIG_NAME}" \
  "model=${MODEL}" \
  "name=${RUN_NAME}" \
  "$@"
