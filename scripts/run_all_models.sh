#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_NAME="${1:-full_multimodel}"
RUN_PREFIX="${2:-rq1}"

"${SCRIPT_DIR}/run_avici.sh" "${CONFIG_NAME}" "${RUN_PREFIX}_avici_only"
"${SCRIPT_DIR}/run_bcnp.sh" "${CONFIG_NAME}" "${RUN_PREFIX}_bcnp_only"
"${SCRIPT_DIR}/run_dibs.sh" "${CONFIG_NAME}" "${RUN_PREFIX}_dibs_only"
"${SCRIPT_DIR}/run_bayesdag.sh" "${CONFIG_NAME}" "${RUN_PREFIX}_bayesdag_only"
