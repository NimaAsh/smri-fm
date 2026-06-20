"""Layer A packer: processed MNI NIfTI + brain mask -> sparse WebDataset shards.

Reproduces the FOMO300 sparse-WDS format (see analysis/fomo300_dataset_format.md),
which was reverse-engineered and round-trip verified byte-for-byte. Consumes the
output of Layer B (src/preprocessing/pipeline.py): per subset, a `processed/` dir
of `*_space-MNI152NLin2009cAsym_desc-processed.nii.gz`, a `derivatives/masks/`
dir of `*_desc-brain_mask.nii.gz`, and `derivatives/synthseg/*_qc.csv`.

Per sample it writes three members keyed `{subset}_{native_stem}`:
  - `<key>.image_values.npy` : float16, z-scored in-brain voxels (C-order)
  - `<key>.img_mask.npy`     : uint8, np.packbits(mask.ravel(), bitorder="big")
  - `<key>.meta.json`        : per-sample bookkeeping (matches Mihir's fields)
plus a dataset-level `metadata.json` with the same knobs/counts.

The encode math is exact (geometry, normalization, packing). The *content* is only
as faithful as Layer B's registration/mask -- validate new shards against the
existing FOMO300 shards with validate_sparse_wds.py before trusting them.

Example (one subset, for validation):
  uv run python src/preprocessing/pack_sparse_wds.py \
    --input  $RAW/PT001_ClevelandCCF \
    --output $OUT/fomo300_repro

Example (extend the corpus; skip keys already packed; continue shard numbering):
  uv run python src/preprocessing/pack_sparse_wds.py \
    --input  $RAW \
    --output $OUT/fomo300_extra \
    --start-shard 1135 \
    --exclude-existing '/data/smri-datasets/FOMO300/shard.*.tar'
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import tarfile
import time
from glob import glob
from pathlib import Path
from typing import Iterator

import nibabel as nib
import numpy as np

log = logging.getLogger("pack")

TEMPLATE_SPACE = "MNI152NLin2009cAsym"
DENSE_SHAPE = (1, 208, 240, 208)  # (C, D, H, W) target grid
PROCESSED_SUFFIX = f"_space-{TEMPLATE_SPACE}_desc-processed.nii.gz"
MASK_SUFFIX = f"_space-{TEMPLATE_SPACE}_desc-brain_mask.nii.gz"


# --------------------------------------------------------------------------- #
# Encode primitives (verified against analysis/fomo300_dataset_format.md)
# --------------------------------------------------------------------------- #
def center_crop_or_pad(arr: np.ndarray, target: tuple[int, ...]) -> np.ndarray:
    """Center crop or pad `arr` to `target` (per axis, floor-low / ceil-high)."""
    out = arr
    for axis, (cur, want) in enumerate(zip(out.shape, target)):
        if cur == want:
            continue
        if cur > want:  # crop
            lo = (cur - want) // 2
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(lo, lo + want)
            out = out[tuple(sl)]
        else:  # pad
            total = want - cur
            lo, hi = total // 2, total - total // 2
            pad = [(0, 0)] * out.ndim
            pad[axis] = (lo, hi)
            out = np.pad(out, pad, mode="constant", constant_values=0)
    return out


def encode_sample(
    image_path: Path,
    mask_path: Path,
    dense_shape: tuple[int, ...],
    eps: float,
) -> tuple[bytes, bytes, dict, int, list[int]]:
    """Return (image_values.npy bytes, img_mask.npy bytes, sparse_meta, n_vox, source_shape)."""
    img = np.asanyarray(nib.load(str(image_path)).dataobj).astype(np.float32)
    mask = np.asanyarray(nib.load(str(mask_path)).dataobj) > 0
    if img.shape != mask.shape:
        raise ValueError(f"image/mask shape mismatch: {img.shape} vs {mask.shape}")
    source_shape = list(img.shape)

    # Add channel axis, then center crop/pad image + mask to the dense grid.
    spatial = dense_shape[1:]
    img = center_crop_or_pad(img, spatial)[None]      # [1, D, H, W]
    mask = center_crop_or_pad(mask, spatial)[None]     # [1, D, H, W]
    if img.shape != tuple(dense_shape):
        raise ValueError(f"post-fit shape {img.shape} != dense {tuple(dense_shape)}")

    # Z-score over masked brain voxels (after the shape fit).
    brain = img[mask]
    if brain.size == 0:
        raise ValueError("empty brain mask after shape fit")
    raw_mean = float(brain.mean())
    raw_std = float(brain.std())
    scale = raw_std if raw_std > eps else eps
    normalized = (img - raw_mean) / scale

    # Sparsify: in-mask values (C-order) as float16 + MSB-first packed mask.
    flat_mask = mask.ravel()
    values = normalized.ravel()[flat_mask].astype(np.float16)
    packed = np.packbits(flat_mask, bitorder="big").astype(np.uint8)
    n_vox = int(values.shape[0])

    sparse_meta = {
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "normalization_scale": scale,
        "eps": eps,
        "num_voxels": n_vox,
        "source_shape": source_shape,
        "dense_shape": list(dense_shape),
    }
    return _npy_bytes(values), _npy_bytes(packed), sparse_meta, n_vox, source_shape


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)  # standard .npy (matches the verified round-trip)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# QC gate (SynthSeg --qc CSV); only ~1% of discovered scans were dropped.
# --------------------------------------------------------------------------- #
def read_qc_score(qc_path: Path, agg: str) -> float | None:
    """Aggregate the numeric QC columns of a SynthSeg qc.csv into one score, or None."""
    if not qc_path.exists():
        return None
    try:
        with open(qc_path, newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            return None
        header, data = rows[0], rows[1]
        scores: list[float] = []
        for h, v in zip(header, data):
            try:
                scores.append(float(v))
            except (TypeError, ValueError):
                continue  # skip the subject-path / non-numeric columns
        if not scores:
            return None
        return float(min(scores) if agg == "min" else np.mean(scores))
    except Exception as e:  # noqa: BLE001 - a malformed qc.csv == missing QC
        log.warning("unreadable qc %s: %s", qc_path, e)
        return None


# --------------------------------------------------------------------------- #
# Sample discovery
# --------------------------------------------------------------------------- #
def iter_subsets(root: Path) -> Iterator[Path]:
    """Yield subset dirs (a dir containing a `processed/`). `root` may be a subset itself."""
    if (root / "processed").is_dir():
        yield root
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "processed").is_dir():
            yield child


def find_samples(subset_dir: Path) -> Iterator[tuple[str, str, Path, Path, Path, str]]:
    """Yield (subset, native_stem, processed, mask, qc, mask_source_type) for a subset."""
    subset = subset_dir.name
    processed_dir = subset_dir / "processed"
    mask_dir = subset_dir / "derivatives" / "masks"
    seg_dir = subset_dir / "derivatives" / "synthseg"
    for proc in sorted(processed_dir.glob(f"*{PROCESSED_SUFFIX}")):
        stem = proc.name[: -len(PROCESSED_SUFFIX)]
        mask = mask_dir / f"{stem}{MASK_SUFFIX}"
        qc = seg_dir / f"{stem}_qc.csv"
        mask_source_type = "brain_mask"
        if not mask.exists():
            # Fall back to the SynthSeg parcellation (label > 0) -> matches the
            # `synthseg` mask-source label in the original metadata.
            dseg = seg_dir / f"{stem}_desc-synthseg_dseg.nii.gz"
            if dseg.exists():
                mask, mask_source_type = dseg, "synthseg"
            else:
                continue
        yield subset, stem, proc, mask, qc, mask_source_type


def modality_from_stem(stem: str) -> str:
    """BIDS suffix -> lowercase modality token (e.g. sub-01_ses-01_T1w -> t1w)."""
    return stem.rsplit("_", 1)[-1].lower()


# --------------------------------------------------------------------------- #
# Minimal ShardWriter (tarfile; no webdataset dependency)
# --------------------------------------------------------------------------- #
class ShardWriter:
    def __init__(self, out_dir: Path, start_shard: int, maxcount: int, maxsize: float):
        self.out_dir = out_dir
        self.pattern = "shard.%06d.tar"
        self.shard = start_shard
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.count = 0
        self.size = 0
        self.tar: tarfile.TarFile | None = None
        self.shards_written: list[str] = []

    def _open(self):
        path = self.out_dir / (self.pattern % self.shard)
        self.tar = tarfile.open(path, "w")
        self.shards_written.append(path.name)
        self.count = 0
        self.size = 0

    def _add(self, name: str, data: bytes):
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mtime = 0
        info.mode = 0o644
        self.tar.addfile(info, io.BytesIO(data))

    def write(self, key: str, members: list[tuple[str, bytes]]):
        sample_size = sum(len(d) for _, d in members)
        if self.tar is None:
            self._open()
        elif self.count >= self.maxcount or (self.count > 0 and self.size + sample_size > self.maxsize):
            self.tar.close()
            self.shard += 1
            self._open()
        for suffix, data in members:
            self._add(f"{key}.{suffix}", data)
        self.count += 1
        self.size += sample_size

    def close(self):
        if self.tar is not None:
            self.tar.close()
            self.tar = None


# --------------------------------------------------------------------------- #
def load_existing_keys(globs: list[str]) -> set[str]:
    """Build the set of already-packed keys from existing shards (member names only)."""
    keys: set[str] = set()
    tars = sorted({p for g in globs for p in glob(g)})
    log.info("scanning %d existing shard(s) for keys to skip", len(tars))
    for t in tars:
        with tarfile.open(t, "r") as tf:
            for name in tf.getnames():
                if name.endswith(".meta.json"):
                    keys.add(name[: -len(".meta.json")])
    log.info("found %d existing keys to skip", len(keys))
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack Layer-B outputs into FOMO300 sparse-WDS shards")
    ap.add_argument("--input", required=True, type=Path, help="subset dir, or a root of subset dirs")
    ap.add_argument("--output", required=True, type=Path, help="dir for new shards + metadata.json")
    ap.add_argument("--start-shard", type=int, default=0)
    ap.add_argument("--maxcount", type=int, default=150)
    ap.add_argument("--maxsize", type=float, default=3e9)
    ap.add_argument("--qc-threshold", type=float, default=0.7)
    ap.add_argument("--qc-agg", choices=["mean", "min"], default="mean")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--dense-shape", type=int, nargs=4, default=list(DENSE_SHAPE))
    ap.add_argument("--exclude-existing", action="append", default=[],
                    help="glob of existing shards whose keys should be skipped (repeatable)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N written (debug)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    dense_shape = tuple(args.dense_shape)
    args.output.mkdir(parents=True, exist_ok=True)

    skip_keys = load_existing_keys(args.exclude_existing) if args.exclude_existing else set()

    writer = ShardWriter(args.output, args.start_shard, args.maxcount, args.maxsize)
    discovered = written = 0
    skip_reason: dict[str, int] = {}
    modality_counts: dict[str, int] = {}
    mask_source_counts: dict[str, int] = {}
    nvox_min, nvox_max, nvox_sum = None, None, 0

    def bump(d: dict, k: str):
        d[k] = d.get(k, 0) + 1

    for subset_dir in iter_subsets(args.input):
        for subset, stem, proc, mask, qc, mask_src in find_samples(subset_dir):
            discovered += 1
            key = f"{subset}_{stem}"
            if key in skip_keys:
                bump(skip_reason, "already_packed")
                continue

            score = read_qc_score(qc, args.qc_agg)
            if score is None or score < args.qc_threshold:
                bump(skip_reason, "missing_synthseg_qc")
                continue

            try:
                vals_b, mask_b, sm, n_vox, source_shape = encode_sample(proc, mask, dense_shape, args.eps)
            except Exception as e:  # noqa: BLE001
                log.error("encode failed %s: %s", key, e)
                bump(skip_reason, "encode_error")
                continue

            modality = modality_from_stem(stem)
            meta = {
                "key": key,
                "subset": subset,
                "modality": modality,
                "native_stem": stem,
                "image_path": str(proc.resolve()),
                "mask_source_path": str(mask.resolve()),
                "mask_source_type": mask_src,
                "acquisition_label": "unknown",
                "quality_decision": "unknown",
                "raw_mean": sm["raw_mean"],
                "raw_std": sm["raw_std"],
                "normalization_scale": sm["normalization_scale"],
                "intensity_normalization": {
                    "method": "brain_mask_zscore",
                    "scope": "masked_brain_voxels_after_shape_fit",
                    "raw_mean": sm["raw_mean"],
                    "raw_std": sm["raw_std"],
                    "normalization_scale": sm["normalization_scale"],
                    "eps": args.eps,
                    "inverse": "raw = normalized * normalization_scale + raw_mean",
                },
                "qc": {"synthseg_qc": score},
                "sparse_image": {
                    "scheme": "mask_selected_values",
                    "source": "img_mask",
                    "source_shape": source_shape,
                    "dense_shape": list(dense_shape),
                    "shape_fit": "center_crop_or_pad",
                    "values_dtype": "float16",
                    "values_normalized": True,
                    "num_voxels": n_vox,
                },
            }
            meta_b = json.dumps(meta).encode()
            writer.write(key, [
                ("image_values.npy", vals_b),
                ("img_mask.npy", mask_b),
                ("meta.json", meta_b),
            ])

            written += 1
            bump(modality_counts, modality)
            bump(mask_source_counts, mask_src)
            nvox_min = n_vox if nvox_min is None else min(nvox_min, n_vox)
            nvox_max = n_vox if nvox_max is None else max(nvox_max, n_vox)
            nvox_sum += n_vox
            if written % 500 == 0:
                log.info("written=%d discovered=%d (shard %06d)", written, discovered, writer.shard)
            if args.limit is not None and written >= args.limit:
                break
        if args.limit is not None and written >= args.limit:
            break

    writer.close()

    metadata = {
        "format": "sparse_wds",
        "discovered_samples": discovered,
        "written_samples": written,
        "skipped_samples": sum(skip_reason.values()),
        "skip_reason_counts": skip_reason,
        "min_synthseg_qc": args.qc_threshold,
        "qc_aggregation": args.qc_agg,
        "image_shape": list(dense_shape),
        "image_values_dtype": "float16",
        "image_values_normalized": True,
        "mask_packed": True,
        "mask_source_counts": mask_source_counts,
        "modality_counts": dict(sorted(modality_counts.items())),
        "template_space": TEMPLATE_SPACE,
        "intensity_normalization": {
            "method": "brain_mask_zscore",
            "scope": "per_sample_masked_brain_voxels_after_shape_fit",
            "eps": args.eps,
            "inverse": "raw = normalized * normalization_scale + raw_mean",
        },
        "maxcount": args.maxcount,
        "maxsize": args.maxsize,
        "start_shard": args.start_shard,
        "shards_written": writer.shards_written,
        "sparse_num_voxels_min": nvox_min,
        "sparse_num_voxels_mean": (nvox_sum / written) if written else None,
        "sparse_num_voxels_max": nvox_max,
        "generated_by": "smri-fm/src/preprocessing/pack_sparse_wds.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(args.output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("done: wrote %d samples across %d shard(s) -> %s",
             written, len(writer.shards_written), args.output)
    log.info("skip reasons: %s", skip_reason)


if __name__ == "__main__":
    main()
