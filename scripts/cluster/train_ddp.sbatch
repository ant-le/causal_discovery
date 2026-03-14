#!/bin/bash
set -euo pipefail

MODEL="${1:?Usage: train_ddp.sbatch <model> [config_name] [run_name] [hydra_overrides...] }"
CONFIG_NAME="${2:-full_multimodel}"
RUN_NAME="${3:-rq1_${MODEL}_ddp}"

if [[ "$#" -ge 3 ]]; then
  shift 3
else
  shift "$#"
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1

uv run torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" -m causal_meta.main \
  --config-name "${CONFIG_NAME}" \
  "model=${MODEL}" \
  "name=${RUN_NAME}" \
  "$@"
