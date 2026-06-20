"""Validate reprocessed sparse-WDS shards against a reference (Mihir's FOMO300).

The Layer-B pipeline (registration + SynthSeg mask) is reproducible only to an
*equivalent*, not bit-identical, output. This harness quantifies that equivalence:
take keys present in BOTH a reference shard set and your candidate set, densify
each sample, and compare:

  - brain-mask Dice                      (geometry/skull-strip agreement)
  - Pearson r of z-scored intensities    (registration + normalization agreement)
  - Pearson r / MAE of raw intensities   (de-normalized, unit-ful)
  - |Δ| of raw_mean, raw_std, num_voxels (per-sample stats)

To use it: reprocess (Layer B + pack_sparse_wds.py) a handful of subsets whose
scans are ALSO in FOMO300, then compare the candidate shards to FOMO300. Tune
ANTs/SynthSeg until Dice and r clear the thresholds; then trust new shards.

Example:
  uv run python src/preprocessing/validate_sparse_wds.py \
    --reference '/data/smri-datasets/FOMO300/shard.*.tar' \
    --candidate '$OUT/fomo300_repro/shard.*.tar' \
    --num-samples 200 --out-csv repro_validation.csv
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import random
import tarfile
from glob import glob
from pathlib import Path

import numpy as np

log = logging.getLogger("validate")


def index_keys(globs: list[str]) -> dict[str, str]:
    """Map sample key -> containing tar path, from member names (metadata-only scan)."""
    index: dict[str, str] = {}
    tars = sorted({p for g in globs for p in glob(g)})
    for t in tars:
        with tarfile.open(t, "r") as tf:
            for name in tf.getnames():
                if name.endswith(".meta.json"):
                    index[name[: -len(".meta.json")]] = t
    return index


def load_sample(tar_path: str, key: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (image_values[f16], packed_mask[u8], meta) for `key` from `tar_path`."""
    with tarfile.open(tar_path, "r") as tf:
        def _read(suffix: str) -> bytes:
            return tf.extractfile(f"{key}.{suffix}").read()
        values = np.load(io.BytesIO(_read("image_values.npy")))
        packed = np.load(io.BytesIO(_read("img_mask.npy")))
        meta = json.loads(_read("meta.json"))
    return values, packed, meta


def densify(values: np.ndarray, packed: np.ndarray, dense_shape: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (dense normalized volume [flat], boolean brain mask [flat])."""
    numel = int(np.prod(dense_shape))
    bits = np.unpackbits(packed, bitorder="big")[:numel].astype(bool)
    dense = np.zeros(numel, dtype=np.float32)
    dense[bits] = values.astype(np.float32)
    return dense, bits


def denorm(dense: np.ndarray, meta: dict) -> np.ndarray:
    """Recover raw intensities: raw = normalized * scale + mean (in-mask only)."""
    scale = float(meta["normalization_scale"])
    mean = float(meta["raw_mean"])
    return dense * scale + mean


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare(ref: tuple, cand: tuple) -> dict:
    rv, rp, rm = ref
    cv, cp, cm = cand
    rds = rm["sparse_image"]["dense_shape"]
    cds = cm["sparse_image"]["dense_shape"]
    r_dense, r_mask = densify(rv, rp, rds)
    c_dense, c_mask = densify(cv, cp, cds)

    inter = r_mask & c_mask
    union = r_mask | c_mask
    dice = 2.0 * inter.sum() / (r_mask.sum() + c_mask.sum()) if (r_mask.sum() + c_mask.sum()) else float("nan")
    iou = inter.sum() / union.sum() if union.sum() else float("nan")

    r_raw = denorm(r_dense, rm)
    c_raw = denorm(c_dense, cm)
    corr_norm = pearson(r_dense[inter], c_dense[inter])
    corr_raw = pearson(r_raw[inter], c_raw[inter])
    mae_raw = float(np.abs(r_raw[inter] - c_raw[inter]).mean()) if inter.any() else float("nan")

    return {
        "dice": dice,
        "iou": iou,
        "corr_norm": corr_norm,
        "corr_raw": corr_raw,
        "mae_raw": mae_raw,
        "d_raw_mean": abs(rm["raw_mean"] - cm["raw_mean"]),
        "d_raw_std": abs(rm["raw_std"] - cm["raw_std"]),
        "ref_nvox": rm["sparse_image"]["num_voxels"],
        "cand_nvox": cm["sparse_image"]["num_voxels"],
        "rel_nvox": abs(rm["sparse_image"]["num_voxels"] - cm["sparse_image"]["num_voxels"])
        / max(rm["sparse_image"]["num_voxels"], 1),
        "modality": rm.get("modality", "?"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate candidate sparse-WDS shards vs a reference")
    ap.add_argument("--reference", action="append", required=True, help="glob(s) of reference shards (repeatable)")
    ap.add_argument("--candidate", action="append", required=True, help="glob(s) of candidate shards (repeatable)")
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dice-threshold", type=float, default=0.97)
    ap.add_argument("--corr-threshold", type=float, default=0.99)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ref_idx = index_keys(args.reference)
    cand_idx = index_keys(args.candidate)
    shared = sorted(set(ref_idx) & set(cand_idx))
    log.info("reference keys=%d  candidate keys=%d  shared=%d", len(ref_idx), len(cand_idx), len(shared))
    if not shared:
        log.error("no shared keys -- reprocess scans that are ALSO in the reference set.")
        raise SystemExit(2)

    random.seed(args.seed)
    if len(shared) > args.num_samples:
        shared = random.sample(shared, args.num_samples)

    rows = []
    for i, key in enumerate(shared, 1):
        try:
            m = compare(load_sample(ref_idx[key], key), load_sample(cand_idx[key], key))
            m["key"] = key
            rows.append(m)
        except Exception as e:  # noqa: BLE001
            log.warning("compare failed %s: %s", key, e)
        if i % 50 == 0:
            log.info("compared %d/%d", i, len(shared))

    if not rows:
        log.error("no comparisons succeeded")
        raise SystemExit(2)

    def col(name: str) -> np.ndarray:
        return np.array([r[name] for r in rows], dtype=float)

    dice, cnorm, craw = col("dice"), col("corr_norm"), col("corr_raw")
    log.info("=== SUMMARY over %d samples ===", len(rows))
    for name, arr in [("dice", dice), ("iou", col("iou")), ("corr_norm", cnorm),
                      ("corr_raw", craw), ("mae_raw", col("mae_raw")),
                      ("rel_nvox", col("rel_nvox")), ("d_raw_mean", col("d_raw_mean")),
                      ("d_raw_std", col("d_raw_std"))]:
        log.info("  %-10s mean=%.5f  median=%.5f  min=%.5f  max=%.5f",
                 name, np.nanmean(arr), np.nanmedian(arr), np.nanmin(arr), np.nanmax(arr))

    dice_ok = np.nanmean(dice) >= args.dice_threshold
    corr_ok = np.nanmean(cnorm) >= args.corr_threshold
    verdict = "PASS" if (dice_ok and corr_ok) else "FAIL"
    log.info("verdict: %s  (mean Dice %.4f vs >=%.2f, mean corr_norm %.4f vs >=%.2f)",
             verdict, np.nanmean(dice), args.dice_threshold, np.nanmean(cnorm), args.corr_threshold)

    if args.out_csv:
        import csv as _csv
        fields = ["key", "modality", "dice", "iou", "corr_norm", "corr_raw", "mae_raw",
                  "d_raw_mean", "d_raw_std", "ref_nvox", "cand_nvox", "rel_nvox"]
        with open(args.out_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fields})
        log.info("per-sample metrics -> %s", args.out_csv)

    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
