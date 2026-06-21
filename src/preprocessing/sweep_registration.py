"""Fast registration sweep: find the {transform, template} that matches FOMO300.

Registers a few raw scans with each (transform_type, template) combo and measures
voxelwise intensity correlation to Mihir's stored values *within his brain mask* --
skipping SynthSeg + packing, so the whole 2x2 grid runs in ~one full-run's time.
Use it to pick the recipe before committing a full pipeline run.

Pearson r is invariant to the z-score, so comparing our raw registered intensities
to Mihir's normalized values is valid; it isolates registration agreement.

Example:
  uv run python src/preprocessing/sweep_registration.py \
    --subset $RAW/PT001_ClevelandCCF \
    --reference '/data/smri-datasets/FOMO300/shard.*.tar' \
    --transforms Rigid Affine --templates brain head --num 5
"""

from __future__ import annotations

import argparse
import itertools
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import templateflow.api as tflow

from pipeline import register_to_template
from pack_sparse_wds import center_crop_or_pad
from validate_sparse_wds import densify, index_keys, load_sample, pearson

TEMPLATE_SPACE = "MNI152NLin2009cAsym"


def get_template(kind: str) -> Path:
    """'brain' = skull-stripped desc-brain T1w; 'head' = the whole-head T1w (no desc)."""
    if kind == "brain":
        p = tflow.get(TEMPLATE_SPACE, resolution=1, desc="brain", suffix="T1w", extension=".nii.gz")
        return Path(p if not isinstance(p, list) else p[0])
    ps = tflow.get(TEMPLATE_SPACE, resolution=1, suffix="T1w", extension=".nii.gz")
    ps = ps if isinstance(ps, list) else [ps]
    head = [x for x in ps if "desc-" not in str(x)]
    if not head:
        raise RuntimeError(f"no whole-head T1w template found among {ps}")
    return Path(head[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep ANTs transform x template vs FOMO300")
    ap.add_argument("--subset", required=True, type=Path, help="raw subset dir (PT###_...)")
    ap.add_argument("--reference", action="append", required=True, help="glob(s) of FOMO300 shards")
    ap.add_argument("--transforms", nargs="+", default=["Rigid", "Affine"])
    ap.add_argument("--templates", nargs="+", default=["brain", "head"])
    ap.add_argument("--interpolator", default="bSpline")
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--dense-shape", type=int, nargs=4, default=[1, 208, 240, 208])
    args = ap.parse_args()

    subset = args.subset.resolve()
    sub_name = subset.name
    ref_idx = index_keys(args.reference)

    raws = sorted(
        p for p in subset.rglob("*.nii.gz")
        if not any(d in p.parts for d in ("processed", "derivatives", "logs"))
    )
    scans = []
    for r in raws:
        key = f"{sub_name}_{r.name[:-len('.nii.gz')]}"
        if key in ref_idx:
            scans.append((r, key))
        if len(scans) >= args.num:
            break
    print(f"comparing {len(scans)} scans present in the reference set")

    templates = {k: get_template(k) for k in args.templates}
    for k, p in templates.items():
        print(f"  template[{k}] = {p}")
    spatial = tuple(args.dense_shape)[1:]

    results: dict[tuple[str, str], list[float]] = {}
    with tempfile.TemporaryDirectory() as td:
        xfm = Path(td) / "x.mat"
        for r, key in scans:
            hv, hp, hm = load_sample(ref_idx[key], key)
            his_dense, his_bits = densify(hv, hp, hm["sparse_image"]["dense_shape"])
            img = nib.load(str(r))
            for tt, tk in itertools.product(args.transforms, args.templates):
                try:
                    reg = register_to_template(img, templates[tk], xfm, tt, args.interpolator)
                    arr = center_crop_or_pad(
                        np.asanyarray(reg.dataobj).astype(np.float32), spatial
                    )[None].ravel()
                    c = pearson(arr[his_bits], his_dense[his_bits])
                except Exception as e:  # noqa: BLE001
                    print(f"  {key} {tt}/{tk} FAILED: {e}")
                    c = float("nan")
                results.setdefault((tt, tk), []).append(c)
                print(f"  {key:48s} {tt:7s} {tk:5s} corr={c:.4f}")

    print("\n=== mean corr within Mihir's brain mask (higher = closer to FOMO300) ===")
    for (tt, tk), cs in sorted(results.items(), key=lambda kv: -np.nanmean(kv[1])):
        print(f"  {tt:7s} {tk:5s}  mean={np.nanmean(cs):.4f}  median={np.nanmedian(cs):.4f}  n={len(cs)}")


if __name__ == "__main__":
    main()
