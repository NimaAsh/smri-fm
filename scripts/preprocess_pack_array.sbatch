#!/bin/bash
# FOMO300 reprocessing: Layer B (register->MNI + SynthSeg mask) + Layer A (pack to
# sparse-WDS), as a throttled SLURM job ARRAY over dataset SUBSETS (PT###_<Subset>).
#
# One array task == one subset == one GPU. With 16 CPUs/task, 8 tasks pack onto a
# single 8xH100 node, so the array soaks up idle GPUs without holding whole nodes.
# Each task: Layer B -> pack that subset into $OUT/part-<subset> -> (optional) delete
# its intermediate NIfTIs, so /data never accumulates the multi-TB processed tree.
#
# Run per-subset (not the whole root at once) because pipeline.py writes a *flat*
# processed/ dir whose stems would collide across subsets otherwise.
#
# Usage:
#   RAW=/data/<you>/fomo_raw            # contains PT###_<Subset>/... raw NIfTI
#   OUT=/data/<you>/fomo300_extra       # shards land in $OUT/part-<subset>/
#   N=$(find "$RAW" -mindepth 1 -maxdepth 1 -type d -name 'PT*' | wc -l)
#   sbatch --array=0-$((N-1))%8 scripts/preprocess_pack_array.sbatch "$RAW" "$OUT"
#
# Priority: defaults to --qos=bottom (only idle GPUs, yields instantly to training).
#   Raise it per-run: sbatch --qos=low ... (CLI overrides the #SBATCH default below).
# Tunables (env): CLEANUP=1 to rm intermediates after packing;
#   REF_SHARDS to change the de-dup reference (default the FOMO300 shards).
#
#SBATCH --job-name=fomo300-reprocess
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=bottom
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=slurms/%A_%a.out
#SBATCH --error=slurms/%A_%a.err

set -euo pipefail

RAW="${1:?Usage: sbatch --array=0-N%K scripts/preprocess_pack_array.sbatch <raw_dir> <out_dir>}"
OUT="${2:?Usage: sbatch --array=0-N%K scripts/preprocess_pack_array.sbatch <raw_dir> <out_dir>}"
CLEANUP="${CLEANUP:-0}"         # 1 => delete processed/derivatives after packing
REF_SHARDS="${REF_SHARDS:-/data/smri-datasets/FOMO300/shard.*.tar}"   # de-dup against existing

mkdir -p slurms
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Deterministic, sorted subset list; pick this task's subset.
mapfile -t SUBSETS < <(find "$RAW" -mindepth 1 -maxdepth 1 -type d -name 'PT*' | sort)
IDX="${SLURM_ARRAY_TASK_ID:?run with --array=0-N%K}"
if [ "$IDX" -ge "${#SUBSETS[@]}" ]; then
  echo "task $IDX >= ${#SUBSETS[@]} subsets -- nothing to do"; exit 0
fi
SUBSET="${SUBSETS[$IDX]}"
NAME="$(basename "$SUBSET")"
CPUS="${SLURM_CPUS_PER_TASK:-16}"

echo "=== task $IDX: subset $NAME ==="
echo "raw=$SUBSET  out=$OUT/part-$NAME  cpus=$CPUS  cleanup=$CLEANUP"

# --- Layer B: rigid->MNI + SynthSeg mask (resumable; skips done scans) ---
uv run python src/preprocessing/pipeline.py \
    --input "$SUBSET" \
    --synthseg-threads "$CPUS"

# --- Layer A: pack this subset into its own part dir; skip keys already in FOMO300 ---
uv run python src/preprocessing/pack_sparse_wds.py \
    --input "$SUBSET" \
    --output "$OUT/part-$NAME" \
    --exclude-existing "$REF_SHARDS"

# --- optional: reclaim /data by deleting the intermediate NIfTIs for this subset ---
if [ "$CLEANUP" = "1" ]; then
  echo "cleanup: removing $SUBSET/{processed,derivatives}"
  rm -rf "$SUBSET/processed" "$SUBSET/derivatives"
fi

echo "=== task $IDX done: $NAME ==="
