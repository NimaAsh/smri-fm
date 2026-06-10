#!/usr/bin/env python
"""Extract frozen sMRI-MAE ViT embeddings for the DLBS ridge probe."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from extract_resenc_features import (
    DlbsNiftiDataset,
    load_manifest,
    parse_modalities,
    parse_target_size,
    write_features,
)


def pool_mae_features(
    embeddings: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cls_embeddings, _, patch_embeddings = embeddings
    if cls_embeddings is None:
        raise ValueError("Checkpoint encoder does not have a class token")
    if cls_embeddings.ndim != 3 or cls_embeddings.shape[1] != 1:
        raise ValueError(f"Expected CLS embeddings shaped [B,1,D], got {cls_embeddings.shape}")
    if patch_embeddings.ndim != 3 or patch_embeddings.shape[1] == 0:
        raise ValueError(f"Expected patch embeddings shaped [B,L,D], got {patch_embeddings.shape}")
    return {
        "cls": cls_embeddings.squeeze(1),
        "mean": patch_embeddings.mean(dim=1),
    }


def load_frozen_encoder(checkpoint_path: str | Path, device: torch.device):
    import smri_mae.model_mae as models_mae

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing = {"model", "args"} - set(checkpoint)
    if missing:
        raise ValueError(f"sMRI-MAE checkpoint is missing keys: {sorted(missing)}")

    args = checkpoint["args"]
    model_name = args["model"]
    model_fn = getattr(models_mae, model_name, None)
    if model_fn is None or not callable(model_fn):
        raise ValueError(f"Unknown sMRI-MAE model: {model_name}")
    model_kwargs = {
        key: args[key]
        for key in ("img_size", "in_chans", "patch_size")
        if key in args
    }
    model_kwargs.update(args.get("model_kwargs") or {})
    model = model_fn(**model_kwargs)
    model.load_state_dict(checkpoint["model"])

    encoder = model.encoder
    encoder.requires_grad_(False)
    encoder.eval()
    encoder.to(device)
    checkpoint_info = {
        "model": model_name,
        "epoch": checkpoint.get("epoch"),
        "img_size": list(encoder.patchify.img_size),
        "patch_size": list(encoder.patchify.patch_size),
        "embedding_dim": int(encoder.patch_embed.out_features),
        "class_token": bool(encoder.has_class_token),
        "reg_tokens": int(encoder.num_reg_tokens),
    }
    del model
    return encoder, checkpoint_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV from build_dlbs_manifest.py")
    parser.add_argument("--checkpoint", required=True, help="Native sMRI-MAE .pth checkpoint")
    parser.add_argument("--cls-output", help="Output CSV for the CLS-token embedding")
    parser.add_argument("--mean-output", help="Output CSV for the mean patch-token embedding")
    parser.add_argument(
        "--image-column", default="image_path", help="Manifest NIfTI path column"
    )
    parser.add_argument("--modalities", default="T1w", help="T1w,T2w,dwi,pet or all")
    parser.add_argument(
        "--target-size",
        help="Optional D,H,W override; must match the checkpoint's native image size",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--amp", action="store_true", help="Use bfloat16 autocast on CUDA")
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=None,
        help="Optional encoder masking ratio; omit to use every brain-containing patch",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, help="Extract only the first N selected rows")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.cls_output and not args.mean_output:
        raise ValueError("At least one of --cls-output or --mean-output is required")
    if args.mask_ratio is not None and not 0 <= args.mask_ratio < 1:
        raise ValueError("--mask-ratio must be in [0, 1)")
    for value in (args.cls_output, args.mean_output):
        if value and Path(value).exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {value}")


def write_metadata(output_path: str | Path, metadata: dict, pooling: str) -> None:
    output_path = Path(output_path)
    payload = {**metadata, "output": str(output_path.resolve()), "pooling": pooling}
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    encoder, checkpoint_info = load_frozen_encoder(args.checkpoint, device)
    native_size = tuple(checkpoint_info["img_size"])
    target_size = parse_target_size(args.target_size) if args.target_size else native_size
    if target_size != native_size:
        raise ValueError(
            f"Target size {target_size} does not match checkpoint image size {native_size}"
        )

    modalities = parse_modalities(args.modalities)
    rows = load_manifest(args.manifest, modalities, args.image_column, args.limit)

    from asparagus.modules.transforms.presets import CPU_clsreg_val_test_transforms_crop

    dataset = DlbsNiftiDataset(
        rows,
        transform=CPU_clsreg_val_test_transforms_crop(target_size=target_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    pat_ids: list[str] = []
    feature_batches: dict[str, list[np.ndarray]] = {"cls": [], "mean": []}
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.amp and device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            embeddings = encoder.forward_embedding(
                image,
                mask=image != 0,
                mask_ratio=args.mask_ratio,
            )
            pooled = pool_mae_features(embeddings)
            pat_ids.extend(batch["pat_id"])
            for pooling, features in pooled.items():
                feature_batches[pooling].append(features.float().cpu().numpy())
            print(f"Extracted {len(pat_ids)}/{len(dataset)}", end="\r", flush=True)
    print()

    metadata = {
        "manifest": str(Path(args.manifest).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "image_column": args.image_column,
        "modalities": sorted(modalities),
        "target_size": list(target_size),
        "preprocessing": "asparagus_clsreg_volume_normalization_and_pad_center_crop",
        "brain_mask": "nonzero_after_preprocessing",
        "mask_ratio": args.mask_ratio,
        "seed": args.seed,
        "n_scans": len(pat_ids),
        "device": str(device),
        "amp_bfloat16": bool(args.amp and device.type == "cuda"),
        "checkpoint_info": checkpoint_info,
    }
    for pooling, output in (("cls", args.cls_output), ("mean", args.mean_output)):
        if not output:
            continue
        feature_matrix = np.concatenate(feature_batches[pooling], axis=0)
        write_features(output, pat_ids, feature_matrix)
        write_metadata(output, {**metadata, "n_features": int(feature_matrix.shape[1])}, pooling)
        print(f"Wrote {output} ({feature_matrix.shape[0]} x {feature_matrix.shape[1]})")


if __name__ == "__main__":
    main()
