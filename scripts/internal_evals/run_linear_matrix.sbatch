#!/usr/bin/env bash
# Run frozen-feature linear probes for a standard sMRI eval matrix.
# Submit via scripts/internal_evals/submit_linear_matrix.sh.

#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/data/${USER}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

OUT_ROOT="${OUT_ROOT:-/data/${USER}/internal_smri_evals_${SLURM_JOB_ID:-manual}}"
FOMO_ROOT="${FOMO_ROOT:-/data/${USER}/fomo26}"
NEUROJEPA_REPO="${NEUROJEPA_REPO:-/admin/home/${USER}/smri-proj/Neuro-JEPA}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP="${AMP:-true}"
MODEL_SET="${MODEL_SET:-all}"
TASKS="${TASKS:-dlbs_age dlbs_mmse dlbs_edu dlbs_bmi adni_age adni_sex adni_ad_cn adni_ad_cn_bag}"
SKIP_MISSING="${SKIP_MISSING:-true}"

OFFICIAL_RESENC_CKPT="${OFFICIAL_RESENC_CKPT:-${FOMO_ROOT}/share/official_amaes_resenc_b_fomo300k/resenc_unet_b.ckpt}"
NIMA_PDF1M_RESENC_CKPT="${NIMA_PDF1M_RESENC_CKPT:-${FOMO_ROOT}/share/pt902_pdf1m_resenc6m_55204/last.ckpt}"

MIHIR_MAE_CKPT="${MIHIR_MAE_CKPT:-/data/mihir-stuff/smri-pretrained/pretrain_full_90_10_h100/checkpoint-last.pth}"
SMRI_MAE_POOLS="${SMRI_MAE_POOLS:-cls reg patch}"

PRETRAIN_FOMO300_CONFIG="${PRETRAIN_FOMO300_CONFIG:-${NEUROJEPA_REPO}/configs/pretrain/pretrain_fomo300.yaml}"
FAITHFUL_PRE96_CONFIG="${FAITHFUL_PRE96_CONFIG:-${NEUROJEPA_REPO}/configs/pretrain/pretrain_fomo300_faithful_pre96.yaml}"

mkdir -p "${OUT_ROOT}" "${HF_DATASETS_CACHE}"

echo "Running internal eval matrix"
echo "  host:        $(hostname)"
echo "  job id:      ${SLURM_JOB_ID:-manual}"
echo "  repo:        ${REPO_ROOT}"
echo "  out root:    ${OUT_ROOT}"
echo "  fomo root:   ${FOMO_ROOT}"
echo "  neurojepa:   ${NEUROJEPA_REPO}"
echo "  tasks:       ${TASKS}"
echo "  model set:   ${MODEL_SET}"
echo "  device:      ${DEVICE}"
echo "  batch size:  ${BATCH_SIZE}"
echo "  workers:     ${NUM_WORKERS}"

IFS=' ' read -r -a TASK_ARRAY <<< "${TASKS}"

run_eval() {
  local model="$1"
  local stem="$2"
  local task="$3"
  shift 3

  local run_name="${stem}__${task}"
  echo
  echo "=== ${run_name} ==="

  local cmd=(
    uv run python -m evaluation.main_linear "${model}" "${task}"
    --overrides
    "output_root=${OUT_ROOT}"
    "name=${run_name}"
    "device=${DEVICE}"
    "batch_size=${BATCH_SIZE}"
    "num_workers=${NUM_WORKERS}"
    "$@"
  )

  printf 'command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

require_file() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    return 0
  fi
  echo "missing file: ${path}"
  if [[ "${SKIP_MISSING}" == "true" ]]; then
    return 1
  fi
  exit 2
}

should_run_model_set() {
  local wanted="$1"
  [[ "${MODEL_SET}" == "all" || "${MODEL_SET}" == "${wanted}" ]]
}

failures=0

if should_run_model_set resenc; then
  RESENC_SPECS=(
    "official_amaes_resenc_b|${OFFICIAL_RESENC_CKPT}"
    "nima_pdf1m_resenc_b|${NIMA_PDF1M_RESENC_CKPT}"
  )

  for spec in "${RESENC_SPECS[@]}"; do
    IFS='|' read -r stem ckpt <<< "${spec}"
    if ! require_file "${ckpt}"; then
      echo "skipping ${stem}"
      continue
    fi
    for task in "${TASK_ARRAY[@]}"; do
      if ! run_eval resenc "${stem}" "${task}" \
        "model_kwargs.ckpt_path=${ckpt}" \
        "model_kwargs.amp=${AMP}"; then
        echo "FAILED: ${stem} ${task}"
        failures=$((failures + 1))
      fi
    done
  done
fi

if should_run_model_set neurojepa; then
  NEUROJEPA_SPECS=(
    "neurojepa_fomo300_e60|${NEUROJEPA_REPO}/runs/pretrain_fomo300/ckpt/e60.pt|${PRETRAIN_FOMO300_CONFIG}"
    "neurojepa_fomo300_latest|${NEUROJEPA_REPO}/runs/pretrain_fomo300/ckpt/latest.pt|${PRETRAIN_FOMO300_CONFIG}"
    "neurojepa_fomo300_cooldown_latest|${NEUROJEPA_REPO}/runs/pretrain_fomo300/ckpt/cooldown/latest.pt|${PRETRAIN_FOMO300_CONFIG}"
    "neurojepa_fomo300_faithful_pre96_latest|${NEUROJEPA_REPO}/runs/pretrain_fomo300_faithful_pre96/ckpt/latest.pt|${FAITHFUL_PRE96_CONFIG}"
    "neurojepa_fomo300_faithful_pre96_cooldown_latest|${NEUROJEPA_REPO}/runs/pretrain_fomo300_faithful_pre96/ckpt/cooldown/latest.pt|${FAITHFUL_PRE96_CONFIG}"
  )

  for spec in "${NEUROJEPA_SPECS[@]}"; do
    IFS='|' read -r stem ckpt config <<< "${spec}"
    if ! require_file "${ckpt}"; then
      echo "skipping ${stem}"
      continue
    fi
    if ! require_file "${config}"; then
      echo "skipping ${stem}"
      continue
    fi
    for task in "${TASK_ARRAY[@]}"; do
      if ! run_eval neurojepa "${stem}" "${task}" \
        "model_kwargs.ckpt_path=${ckpt}" \
        "model_kwargs.config_path=${config}" \
        "model_kwargs.repo_path=${NEUROJEPA_REPO}" \
        "model_kwargs.pooling=mean" \
        "model_kwargs.amp=${AMP}"; then
        echo "FAILED: ${stem} ${task}"
        failures=$((failures + 1))
      fi
    done
  done
fi

if should_run_model_set smri_mae; then
  SMRI_MAE_SPECS=(
    "mihir_mae_vitl_full90_10|${MIHIR_MAE_CKPT}"
  )
  IFS=' ' read -r -a POOL_ARRAY <<< "${SMRI_MAE_POOLS}"

  for spec in "${SMRI_MAE_SPECS[@]}"; do
    IFS='|' read -r stem ckpt <<< "${spec}"
    if ! require_file "${ckpt}"; then
      echo "skipping ${stem}"
      continue
    fi
    # smri_mae's transform casts to fp16 internally and its builder takes no amp arg.
    for pool in "${POOL_ARRAY[@]}"; do
      for task in "${TASK_ARRAY[@]}"; do
        if ! run_eval smri_mae "${stem}_${pool}" "${task}" \
          "model_kwargs.ckpt_path=${ckpt}" \
          "model_kwargs.global_pool=${pool}"; then
          echo "FAILED: ${stem}_${pool} ${task}"
          failures=$((failures + 1))
        fi
      done
    done
  done
fi

echo
echo "Matrix complete with ${failures} failed run(s)."
exit "${failures}"
