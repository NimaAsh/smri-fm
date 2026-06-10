#!/usr/bin/env python
"""Extract frozen Asparagus ResEnc-B features for the DLBS ridge probe."""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


MRI_MODALITIES = {"T1w", "T2w", "dwi"}
ALL_MODALITIES = MRI_MODALITIES | {"pet"}


def parse_modalities(value: str) -> set[str]:
    if value.strip().lower() == "all":
        return set(ALL_MODALITIES)
    aliases = {"t1w": "T1w", "t2w": "T2w", "dwi": "dwi", "pet": "pet"}
    modalities = {aliases.get(item.strip().lower(), item.strip()) for item in value.split(",")}
    unknown = modalities - ALL_MODALITIES
    if not modalities or unknown:
        raise ValueError(f"Unknown modalities: {sorted(unknown)}")
    return modalities


def parse_target_size(value: str) -> tuple[int, int, int]:
    parts = tuple(int(item.strip()) for item in value.split(","))
    if len(parts) != 3 or any(item <= 0 for item in parts):
        raise ValueError("--target-size must contain three positive integers")
    return parts


def load_manifest(
    path: str | Path,
    modalities: set[str],
    image_column: str,
    limit: int | None = None,
) -> list[dict[str, str]]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"pat_id", "modality", image_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    frame = frame[frame["modality"].isin(modalities)].copy()
    if limit is not None:
        frame = frame.head(limit)
    if frame.empty:
        raise ValueError(f"No manifest rows matched modalities: {sorted(modalities)}")
    if frame["pat_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["pat_id"].duplicated(), "pat_id"].unique())
        raise ValueError(f"Manifest contains duplicate pat_id values: {duplicates[:5]}")

    rows = frame[["pat_id", "modality", image_column]].to_dict("records")
    for row in rows:
        row["image_path"] = row.pop(image_column)
    return rows


class DlbsNiftiDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = np.asarray(nib.load(image_path).dataobj, dtype=np.float32)
        if image.ndim != 3:
            raise ValueError(f"Expected a 3D image, got shape {image.shape}: {image_path}")
        if not np.all(np.isfinite(image)):
            raise ValueError(f"Image contains non-finite values: {image_path}")

        sample = {
            "image": torch.from_numpy(image.copy()).unsqueeze(0),
            "transforms_applied": {},
        }
        sample = self.transform(sample)
        return {
            "pat_id": row["pat_id"],
            "modality": row["modality"],
            "image_path": str(image_path),
            "image": sample["image"],
        }


def pool_resenc_features(encoded) -> torch.Tensor:
    deepest = encoded[-1] if isinstance(encoded, (list, tuple)) else encoded
    if deepest.ndim != 5:
        raise ValueError(f"Expected ResEnc features shaped [B,C,X,Y,Z], got {deepest.shape}")
    return F.adaptive_avg_pool3d(deepest, output_size=1).flatten(1)


def load_checkpoint_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        print(
            "Loading weights trained for "
            f"{checkpoint.get('global_step', '?')} steps / {checkpoint.get('epoch', '?')} epochs."
        )
        return checkpoint["state_dict"]
    if "network_weights" in checkpoint:
        print("Loading weights from external checkpoint (network_weights key).")
        return checkpoint["network_weights"]
    raise ValueError("Unsupported checkpoint format. Expected state_dict or network_weights key.")


def build_frozen_resenc(checkpoint: str | Path, device: torch.device):
    from asparagus.modules.lightning_modules.base_module import BaseModule
    from asparagus.modules.networks.resenc_unet import resenc_unet_b_clsreg

    class FrozenFeatureModule(BaseModule):
        def training_step(self, batch, batch_idx):  # pragma: no cover - inference only
            raise RuntimeError("FrozenFeatureModule is inference-only")

        def validation_step(self, batch, batch_idx):  # pragma: no cover - inference only
            raise RuntimeError("FrozenFeatureModule is inference-only")

    model = resenc_unet_b_clsreg(
        input_channels=1,
        output_channels=1,
        dimensions="3D",
        dropout_op_kwargs={
            "encoder_dropout_rate": 0.0,
            "decoder_dropout_rate": 0.0,
            "inplace": True,
        },
        late_fusion=False,
    )
    module = FrozenFeatureModule(
        model=model,
        weights=load_checkpoint_state_dict(str(checkpoint)),
        load_decoder=False,
        repeat_stem_weights=False,
    )
    module.requires_grad_(False)
    module.eval()
    return module.to(device)


def write_features(
    output_path: str | Path,
    pat_ids: list[str],
    features: np.ndarray,
) -> None:
    if features.ndim != 2 or features.shape[0] != len(pat_ids):
        raise ValueError("Expected one two-dimensional feature row per pat_id")
    if not np.all(np.isfinite(features)):
        raise ValueError("Extracted features contain non-finite values")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pat_id", *[f"Feature_{index}" for index in range(features.shape[1])]])
        for pat_id, feature_row in zip(pat_ids, features):
            writer.writerow([pat_id, *[f"{value:.9g}" for value in feature_row]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV from build_dlbs_manifest.py")
    parser.add_argument("--checkpoint", required=True, help="Asparagus ResEnc-B checkpoint")
    parser.add_argument("--output", required=True, help="Output feature CSV")
    parser.add_argument(
        "--image-column",
        default="image_path",
        help="Manifest column containing the model input NIfTI path",
    )
    parser.add_argument(
        "--modalities",
        default="T1w",
        help="Comma-separated T1w,T2w,dwi,pet or all (default: T1w)",
    )
    parser.add_argument("--target-size", default="128,128,128")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--amp", action="store_true", help="Use bfloat16 autocast on CUDA")
    parser.add_argument("--limit", type=int, help="Extract only the first N selected rows")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output_path}")

    modalities = parse_modalities(args.modalities)
    target_size = parse_target_size(args.target_size)
    rows = load_manifest(args.manifest, modalities, args.image_column, args.limit)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    from asparagus.modules.transforms.presets import CPU_clsreg_val_test_transforms_crop

    dataset = DlbsNiftiDataset(
        rows,
        transform=CPU_clsreg_val_test_transforms_crop(target_size=target_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    module = build_frozen_resenc(args.checkpoint, device)

    pat_ids: list[str] = []
    feature_batches: list[np.ndarray] = []
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.amp and device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            features = pool_resenc_features(module.model._encode(image)).float().cpu().numpy()
            pat_ids.extend(batch["pat_id"])
            feature_batches.append(features)
            print(f"Extracted {len(pat_ids)}/{len(dataset)}", end="\r", flush=True)
    print()

    feature_matrix = np.concatenate(feature_batches, axis=0)
    write_features(output_path, pat_ids, feature_matrix)
    metadata = {
        "manifest": str(Path(args.manifest).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output": str(output_path.resolve()),
        "image_column": args.image_column,
        "modalities": sorted(modalities),
        "target_size": list(target_size),
        "pooling": "global_average_final_encoder_stage",
        "model": "asparagus_resenc_unet_b",
        "n_scans": len(pat_ids),
        "n_features": int(feature_matrix.shape[1]),
        "device": str(device),
        "amp_bfloat16": bool(args.amp and device.type == "cuda"),
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {output_path} ({feature_matrix.shape[0]} x {feature_matrix.shape[1]})")


if __name__ == "__main__":
    main()
