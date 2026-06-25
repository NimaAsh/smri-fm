#!/usr/bin/env bash
#SBATCH --job-name=smri_fm_internal_eval
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=slurms/%x-%j.out
#SBATCH --error=slurms/%x-%j.err
#SBATCH --time=04:00:00


set -euo pipefail

if [[ -f .env ]]; then
  echo "Setting up environment variables."
  source scripts/internal_evals/load_env.sh
  echo "Done setting up environment variables."
fi

echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-not_set}"
echo "Working directory: $(pwd)"

MODEL="${MODEL:-smri_mae}"
TASK="${TASK:-dlbs_age}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/${USER}/internal_smri_evals_single}"
NAME="${NAME:-${MODEL}__${TASK}}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"

IFS=' ' read -r -a EXTRA_OVERRIDES_ARRAY <<< "${EXTRA_OVERRIDES:-}"

uv run python -m evaluation.main_linear "${MODEL}" "${TASK}" \
  --overrides \
  "output_root=${OUTPUT_ROOT}" \
  "name=${NAME}" \
  "device=${DEVICE}" \
  "batch_size=${BATCH_SIZE}" \
  "num_workers=${NUM_WORKERS}" \
  "${EXTRA_OVERRIDES_ARRAY[@]}"
