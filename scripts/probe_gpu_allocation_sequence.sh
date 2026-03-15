#!/usr/bin/env bash
set -euo pipefail

# Wait for an idle node, then request 1..MAX_GPU GPUs in sequence.
# This helps distinguish true scheduler clipping from transient contention.
#
# Usage:
#   scripts/probe_gpu_allocation_sequence.sh [PARTITION] [MAX_GPU]
# Example:
#   scripts/probe_gpu_allocation_sequence.sh GPU-a100 4

PARTITION="${1:-GPU-a100}"
MAX_GPU="${2:-4}"
CPUS_PER_GPU="${CPUS_PER_GPU:-16}"
TIME_LIMIT="${TIME_LIMIT:-00:05:00}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-7200}"
POLL_SECONDS="${POLL_SECONDS:-15}"

if ! [[ "${MAX_GPU}" =~ ^[0-9]+$ ]] || [[ "${MAX_GPU}" -lt 1 ]]; then
  echo "MAX_GPU must be a positive integer, got: ${MAX_GPU}" >&2
  exit 1
fi

if ! [[ "${CPUS_PER_GPU}" =~ ^[0-9]+$ ]] || [[ "${CPUS_PER_GPU}" -lt 1 ]]; then
  echo "CPUS_PER_GPU must be a positive integer, got: ${CPUS_PER_GPU}" >&2
  exit 1
fi

RUN_DIR="$(pwd)/gpu_alloc_probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"

extract_gpu_count() {
  local text="$1"
  if [[ "${text}" =~ gres/gpu=([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo 0
  fi
}

node_cfg_gpus() {
  local node="$1"
  local info cfg_tres
  info="$(scontrol show node "${node}" 2>/dev/null || true)"
  if [[ "${info}" =~ CfgTRES=([^[:space:]]+) ]]; then
    cfg_tres="${BASH_REMATCH[1]}"
    extract_gpu_count "${cfg_tres}"
  else
    echo 0
  fi
}

find_idle_node_with_capacity() {
  local required="$1"
  local node cfg
  while read -r node; do
    [[ -z "${node}" ]] && continue
    cfg="$(node_cfg_gpus "${node}")"
    if (( cfg >= required )); then
      echo "${node}"
      return 0
    fi
  done < <(sinfo -N -h -p "${PARTITION}" -t idle -o "%N")
  return 1
}

wait_for_job() {
  local jobid="$1"
  local start now
  start="$(date +%s)"

  while squeue -h -j "${jobid}" | grep -q .; do
    sleep "${POLL_SECONDS}"
    now="$(date +%s)"
    if (( now - start > WAIT_TIMEOUT_SECONDS )); then
      echo "Timed out waiting for job ${jobid}" >&2
      squeue -j "${jobid}" || true
      return 1
    fi
  done
  return 0
}

get_sacct_row() {
  local jobid="$1"
  local row=""
  local i
  for i in {1..20}; do
    row="$(sacct -j "${jobid}" -X -n -P -o JobID,State,ReqTRES,AllocTRES | head -n 1)"
    if [[ -n "${row}" ]]; then
      echo "${row}"
      return 0
    fi
    sleep 2
  done
  echo "${row}"
  return 0
}

echo "===== Probe configuration ====="
echo "PARTITION=${PARTITION}"
echo "MAX_GPU=${MAX_GPU}"
echo "CPUS_PER_GPU=${CPUS_PER_GPU}"
echo "TIME_LIMIT=${TIME_LIMIT}"
echo "WAIT_TIMEOUT_SECONDS=${WAIT_TIMEOUT_SECONDS}"
echo "RUN_DIR=${RUN_DIR}"

echo "===== Waiting for idle node with >= ${MAX_GPU} GPUs ====="
target_node=""
start_wait="$(date +%s)"
while true; do
  if target_node="$(find_idle_node_with_capacity "${MAX_GPU}")"; then
    break
  fi

  now_wait="$(date +%s)"
  if (( now_wait - start_wait > WAIT_TIMEOUT_SECONDS )); then
    echo "No idle node with >= ${MAX_GPU} GPUs found in ${PARTITION}" >&2
    exit 2
  fi

  sleep "${POLL_SECONDS}"
done

echo "Selected node: ${target_node}"
echo ""

summary_file="${RUN_DIR}/summary.tsv"
summary_report_file="${RUN_DIR}/summary_report.txt"
printf "request_gpus\tjobid\tstate\treq_tres\talloc_tres\tslurm_gpus_on_node\tslurm_job_gpus\n" > "${summary_file}"

for request_gpus in $(seq 1 "${MAX_GPU}"); do
  cpus=$(( request_gpus * CPUS_PER_GPU ))
  job_script="${RUN_DIR}/job_${request_gpus}.sbatch"

  cat > "${job_script}" <<EOF
#!/bin/bash
#SBATCH -p ${PARTITION}
#SBATCH -N 1
#SBATCH --nodelist=${target_node}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${cpus}
#SBATCH --gpus-per-node=${request_gpus}
#SBATCH -t ${TIME_LIMIT}
#SBATCH -J gpu_probe_${request_gpus}
#SBATCH -o ${RUN_DIR}/req${request_gpus}_%j.out
#SBATCH -e ${RUN_DIR}/req${request_gpus}_%j.err

set -euo pipefail
echo "SLURM_GPUS_ON_NODE=\${SLURM_GPUS_ON_NODE:-unset}"
echo "SLURM_JOB_GPUS=\${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU_DEVICE_ORDINAL=\${GPU_DEVICE_ORDINAL:-unset}"
nvidia-smi -L || true
EOF

  submit_out="$(sbatch "${job_script}")"
  jobid="$(awk '{print $4}' <<< "${submit_out}")"
  if [[ -z "${jobid}" ]]; then
    echo "Failed to parse job id from: ${submit_out}" >&2
    exit 3
  fi

  echo "Submitted request ${request_gpus} GPU(s): job ${jobid}"

  wait_for_job "${jobid}"

  sacct_row="$(get_sacct_row "${jobid}")"
  IFS='|' read -r _jobid state req_tres alloc_tres <<< "${sacct_row}"

  out_file="${RUN_DIR}/req${request_gpus}_${jobid}.out"
  slurm_gpus_on_node=""
  slurm_job_gpus=""
  if [[ -f "${out_file}" ]]; then
    slurm_gpus_on_node="$(grep -m1 '^SLURM_GPUS_ON_NODE=' "${out_file}" | cut -d= -f2-)"
    slurm_job_gpus="$(grep -m1 '^SLURM_JOB_GPUS=' "${out_file}" | cut -d= -f2-)"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${request_gpus}" "${jobid}" "${state:-unknown}" "${req_tres:-}" \
    "${alloc_tres:-}" "${slurm_gpus_on_node:-}" "${slurm_job_gpus:-}" \
    >> "${summary_file}"
done

echo ""
echo "===== Summary ====="
cat "${summary_file}"
echo ""
{
  echo "===== Probe configuration ====="
  echo "PARTITION=${PARTITION}"
  echo "MAX_GPU=${MAX_GPU}"
  echo "CPUS_PER_GPU=${CPUS_PER_GPU}"
  echo "TIME_LIMIT=${TIME_LIMIT}"
  echo "WAIT_TIMEOUT_SECONDS=${WAIT_TIMEOUT_SECONDS}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "TARGET_NODE=${target_node}"
  echo ""
  echo "===== Summary ====="
  cat "${summary_file}"
} > "${summary_report_file}"

echo "Artifacts written to: ${RUN_DIR}"
echo "Summary report: ${summary_report_file}"
