#!/usr/bin/env bash
# Submit the internal frozen-feature linear-eval matrix to SLURM.
#
# Common overrides:
#   OUT_ROOT=/data/$USER/internal_smri_evals_test scripts/internal_evals/submit_linear_matrix.sh
#   TASKS="dlbs_age adni_age" scripts/internal_evals/submit_linear_matrix.sh
#   MODEL_SET=resenc scripts/internal_evals/submit_linear_matrix.sh
#   MODEL_SET=smri_mae scripts/internal_evals/submit_linear_matrix.sh   # Mihir's MAE ViT-L, pools cls/reg/patch

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-/data/${USER}/internal_smri_evals_${TIMESTAMP}}"
JOB_NAME="${JOB_NAME:-smri-linear-matrix}"

mkdir -p "${OUT_ROOT}/slurms"

echo "Submitting internal eval matrix"
echo "  repo:     ${REPO_ROOT}"
echo "  out root: ${OUT_ROOT}"
echo "  tasks:    ${TASKS:-dlbs_age adni_age adni_sex adni_ad_cn adni_ad_cn_bag}"
echo "  models:   ${MODEL_SET:-all}"

sbatch --parsable \
  --job-name="${JOB_NAME}" \
  --output="${OUT_ROOT}/slurms/%x-%j.out" \
  --error="${OUT_ROOT}/slurms/%x-%j.err" \
  --export=ALL,OUT_ROOT="${OUT_ROOT}" \
  "${REPO_ROOT}/scripts/internal_evals/run_linear_matrix.sbatch"
