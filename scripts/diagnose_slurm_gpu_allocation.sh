#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/diagnose_slurm_gpu_allocation.sh [PARTITION] [NODE_NAME] [GPU_REQUEST]
# Example:
#   scripts/diagnose_slurm_gpu_allocation.sh GPU-a100 a-a100-o-1 4

PARTITION="${1:-GPU-a100}"
NODE_NAME="${2:-a-a100-o-1}"
GPU_REQUEST="${3:-4}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-1800}"

if ! [[ "${GPU_REQUEST}" =~ ^[0-9]+$ ]] || [[ "${GPU_REQUEST}" -lt 1 ]]; then
  echo "GPU_REQUEST must be a positive integer, got: ${GPU_REQUEST}" >&2
  exit 1
fi

echo "===== Input ====="
echo "PARTITION=${PARTITION}"
echo "NODE_NAME=${NODE_NAME}"
echo "GPU_REQUEST=${GPU_REQUEST}"
echo "MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS}"

echo "===== Versions ====="
sinfo -V
srun --version || true

echo "===== Partition + node view ====="
sinfo -N -p "${PARTITION}" -o "%N %t %G %m %c" || true

echo "===== Target node detail ====="
scontrol show node "${NODE_NAME}" || true

echo "===== User/account/QOS limits ====="
sacctmgr show assoc where user="$USER" format=User,Account,QOS,MaxTRES%50,GrpTRES%50 || true
sacctmgr show qos format=Name,MaxTRESPerUser%50,MaxTRESPerJob%50,GrpTRES%50 || true

echo "===== Core Slurm config knobs ====="
scontrol show config | egrep -i "SelectType=|SelectTypeParameters=|TaskPlugin=|ProctrackType=|GresTypes=|AccountingStorageEnforce=" || true

diag_script="/tmp/gpu_diag_${GPU_REQUEST}_$$.sbatch"
job_out="$(pwd)/gpu_diag_${GPU_REQUEST}_%j.out"
job_err="$(pwd)/gpu_diag_${GPU_REQUEST}_%j.err"

cat > "${diag_script}" <<EOF
#!/bin/bash
#SBATCH -p ${PARTITION}
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=${GPU_REQUEST}
#SBATCH -t 00:10:00
#SBATCH -J gpu_diag_${GPU_REQUEST}
#SBATCH -o ${job_out}
#SBATCH -e ${job_err}

set -euo pipefail

echo "===== Job identity ====="
echo "SLURM_JOB_ID=\${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NODELIST=\${SLURM_JOB_NODELIST:-unset}"
echo "SLURM_JOB_GPUS=\${SLURM_JOB_GPUS:-unset}"
echo "SLURM_GPUS_ON_NODE=\${SLURM_GPUS_ON_NODE:-unset}"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU_DEVICE_ORDINAL=\${GPU_DEVICE_ORDINAL:-unset}"

echo "===== scontrol job view ====="
scontrol show job "\$SLURM_JOB_ID"

echo "===== nvidia-smi topology ====="
nvidia-smi -L || true

echo "===== MIG + memory ====="
nvidia-smi --query-gpu=name,uuid,mig.mode.current,memory.total --format=csv || true
EOF

echo "===== Submitting diagnostic job ====="
submit_out="$(sbatch "${diag_script}")"
echo "${submit_out}"

jobid="$(awk '{print $4}' <<< "${submit_out}")"
if [[ -z "${jobid}" ]]; then
  echo "Failed to parse job id from sbatch output: ${submit_out}" >&2
  exit 2
fi

echo "Waiting for job ${jobid} to finish..."
start_time="$(date +%s)"

while squeue -h -j "${jobid}" | grep -q .; do
  sleep 3
  now="$(date +%s)"
  if (( now - start_time > MAX_WAIT_SECONDS )); then
    echo "Timed out waiting for job ${jobid} after ${MAX_WAIT_SECONDS}s" >&2
    echo "Current status:" >&2
    squeue -j "${jobid}" || true
    exit 3
  fi
done

echo "===== Requested vs allocated TRES ====="
sacct -j "${jobid}" -X -o JobID,State,ReqTRES,AllocTRES%100 -P || true

echo "===== Diagnostic logs ====="
out_file="gpu_diag_${GPU_REQUEST}_${jobid}.out"
err_file="gpu_diag_${GPU_REQUEST}_${jobid}.err"

if [[ -f "${err_file}" ]]; then
  echo "----- ${err_file} -----"
  cat "${err_file}"
else
  echo "Missing ${err_file}" >&2
fi

if [[ -f "${out_file}" ]]; then
  echo "----- ${out_file} -----"
  cat "${out_file}"
else
  echo "Missing ${out_file}" >&2
fi

rm -f "${diag_script}"

echo "===== Done ====="
echo "If you share the output, we can pinpoint MIG vs limits vs scheduler bug."
