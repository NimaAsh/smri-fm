#!/usr/bin/env python
"""Extract frozen Neuro-JEPA ViT embeddings for the DLBS ridge probe."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from extract_resenc_features import (
    DlbsNiftiDataset,
    load_manifest,
    parse_modalities,
    parse_target_size,
    write_features,
)


def add_neurojepa_to_path(repo: str | Path) -> Path:
    repo = Path(repo).expanduser().resolve()
    src = repo / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"Neuro-JEPA repo src directory not found: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return repo


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def config_get(section: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return section.get(key, default)


def to_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{key: to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def checkpoint_scalar(value: Any) -> int | float | str | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return f"tensor(shape={tuple(value.shape)})"
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def build_backbone_kwargs(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    model_cfg = config_get(config, "model", {}) or {}
    meta_cfg = config_get(config, "meta", {}) or {}
    if not isinstance(model_cfg, Mapping) or not isinstance(meta_cfg, Mapping):
        raise ValueError("Neuro-JEPA config must contain mapping-valued model/meta sections")

    backbone_name = config_get(
        model_cfg,
        "backbone_name",
        config_get(model_cfg, "model_name", "vit_base"),
    )
    kwargs: dict[str, Any] = {
        "device": device,
        "model_name": backbone_name,
        "img_size": config_get(model_cfg, "img_size", [224, 224, 64]),
        "patch_size": config_get(model_cfg, "patch_size", [16, 16, 4]),
        "in_chans": config_get(model_cfg, "in_chans", 1),
        "out_layers": config_get(model_cfg, "out_layers", None),
        "uniform_power": config_get(model_cfg, "uniform_power", False),
        "use_sdpa": config_get(meta_cfg, "use_sdpa", config_get(model_cfg, "use_sdpa", False)),
        "use_silu": config_get(model_cfg, "use_silu", False),
        "wide_silu": config_get(model_cfg, "wide_silu", False),
        "use_rope": config_get(model_cfg, "use_rope", False),
        "use_activation_checkpointing": False,
    }

    if config_get(model_cfg, "use_moe", False):
        kwargs["use_moe"] = True
        kwargs["moe_params"] = to_namespace(config_get(model_cfg, "moe_params", None))
    return kwargs


def load_frozen_backbone(
    checkpoint_path: str | Path,
    config_path: str | Path,
    neurojepa_repo: str | Path,
    device: torch.device,
    checkpoint_key: str,
    strict: bool,
):
    repo = add_neurojepa_to_path(neurojepa_repo)
    from neurojepa.utils.checkpoint_loader import robust_checkpoint_loader
    from neurojepa.utils.init_utils import (
        _clean_backbone_state_dict,
        _extract_backbone_state_dict,
        init_backbone,
    )

    config = load_yaml_config(config_path)
    backbone_kwargs = build_backbone_kwargs(config, device)
    backbone = init_backbone(**backbone_kwargs)

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = robust_checkpoint_loader(str(checkpoint_path), map_location=torch.device("cpu"))
    state_dict = _extract_backbone_state_dict(checkpoint, checkpoint_key=checkpoint_key)
    state_dict = _clean_backbone_state_dict(state_dict)
    loaded_keys = sorted(set(state_dict) & set(backbone.state_dict()))
    if not loaded_keys:
        raise ValueError(
            "Checkpoint did not contain any keys matching the Neuro-JEPA backbone. "
            f"Check --checkpoint-key '{checkpoint_key}' and the YAML architecture."
        )
    incompatible = backbone.load_state_dict(state_dict, strict=strict)

    backbone.requires_grad_(False)
    backbone.eval()
    backbone.to(device)

    checkpoint_info = {
        "repo": str(repo),
        "config": str(Path(config_path).expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_key": checkpoint_key,
        "strict": strict,
        "epoch": checkpoint_scalar(checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None),
        "global_step": checkpoint_scalar(
            checkpoint.get("global_step") if isinstance(checkpoint, Mapping) else None
        ),
        "loss": checkpoint_scalar(checkpoint.get("loss") if isinstance(checkpoint, Mapping) else None),
        "n_loaded_backbone_keys": len(loaded_keys),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "model": {
            "backbone_name": backbone_kwargs["model_name"],
            "img_size": list(backbone_kwargs["img_size"]),
            "patch_size": list(backbone_kwargs["patch_size"]),
            "embedding_dim": int(getattr(backbone, "embed_dim")),
            "num_patches": int(getattr(backbone, "num_patches")),
        },
    }
    del checkpoint
    return backbone, checkpoint_info


def patch_foreground_mask(
    image: torch.Tensor,
    patch_size: tuple[int, int, int],
    min_fraction: float = 0.1,
) -> torch.Tensor:
    if image.ndim != 5:
        raise ValueError(f"Expected image shaped [B,C,H,W,D], got {image.shape}")
    if any(dim % patch != 0 for dim, patch in zip(image.shape[2:], patch_size)):
        raise ValueError(f"Image shape {tuple(image.shape[2:])} is not divisible by {patch_size}")
    foreground = image.abs().amax(dim=1, keepdim=True) > 0
    patch_fraction = F.avg_pool3d(
        foreground.float(),
        kernel_size=patch_size,
        stride=patch_size,
    )
    return patch_fraction.flatten(1) >= min_fraction


def pool_neurojepa_tokens(
    tokens: torch.Tensor,
    image: torch.Tensor | None,
    patch_size: tuple[int, int, int],
    pooling: str,
    min_foreground_fraction: float = 0.1,
) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[1] == 0:
        raise ValueError(f"Expected token embeddings shaped [B,L,D], got {tokens.shape}")
    if pooling == "mean":
        return tokens.mean(dim=1)
    if pooling == "max":
        return tokens.max(dim=1).values
    if pooling == "foreground_mean":
        if image is None:
            raise ValueError("foreground_mean pooling requires the input image tensor")
        mask = patch_foreground_mask(image, patch_size, min_foreground_fraction)
        if mask.shape != tokens.shape[:2]:
            raise ValueError(
                f"Patch mask shape {tuple(mask.shape)} does not match tokens {tuple(tokens.shape[:2])}"
            )
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (tokens * weights).sum(dim=1) / denominator
    raise ValueError(f"Unsupported pooling mode: {pooling}")


def resolve_target_size(args: argparse.Namespace, checkpoint_info: Mapping[str, Any]) -> tuple[int, int, int]:
    native_size = tuple(int(value) for value in checkpoint_info["model"]["img_size"])
    if args.target_size is None:
        return native_size
    target_size = parse_target_size(args.target_size)
    if target_size != native_size and not args.allow_size_mismatch:
        raise ValueError(
            f"Target size {target_size} does not match checkpoint image size {native_size}; "
            "pass --allow-size-mismatch only if you know the architecture supports it"
        )
    return target_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV from build_dlbs_manifest.py")
    parser.add_argument("--checkpoint", required=True, help="Neuro-JEPA checkpoint, e.g. ckpt/latest.pt")
    parser.add_argument("--config", required=True, help="Neuro-JEPA YAML config used to build the encoder")
    parser.add_argument("--neurojepa-repo", required=True, help="Path to the Neuro-JEPA repository")
    parser.add_argument("--output", required=True, help="Output feature CSV")
    parser.add_argument(
        "--checkpoint-key",
        default="encoder",
        help="Checkpoint state-dict key to load, usually encoder or target_encoder",
    )
    parser.add_argument(
        "--pooling",
        choices=("mean", "foreground_mean", "max"),
        default="mean",
        help="How to pool Neuro-JEPA patch tokens",
    )
    parser.add_argument(
        "--min-foreground-fraction",
        type=float,
        default=0.1,
        help="Patch foreground threshold used only for foreground_mean pooling",
    )
    parser.add_argument("--image-column", default="image_path", help="Manifest NIfTI path column")
    parser.add_argument("--modalities", default="T1w", help="T1w,T2w,dwi,pet or all")
    parser.add_argument("--target-size", help="Optional H,W,D override; defaults to checkpoint image size")
    parser.add_argument("--allow-size-mismatch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--amp", action="store_true", help="Use bfloat16 autocast on CUDA")
    parser.add_argument("--strict", action="store_true", help="Strict checkpoint loading")
    parser.add_argument("--limit", type=int, help="Extract only the first N selected rows")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if Path(args.output).exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {args.output}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if not 0 <= args.min_foreground_fraction <= 1:
        raise ValueError("--min-foreground-fraction must be in [0, 1]")


def write_metadata(output_path: str | Path, metadata: dict[str, Any]) -> None:
    output_path = Path(output_path)
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    backbone, checkpoint_info = load_frozen_backbone(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        neurojepa_repo=args.neurojepa_repo,
        device=device,
        checkpoint_key=args.checkpoint_key,
        strict=args.strict,
    )
    target_size = resolve_target_size(args, checkpoint_info)
    patch_size = tuple(int(value) for value in checkpoint_info["model"]["patch_size"])

    modalities = parse_modalities(args.modalities)
    rows = load_manifest(args.manifest, modalities, args.image_column, args.limit)

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
            tokens, _ = backbone(image)
            features = pool_neurojepa_tokens(
                tokens,
                image=image,
                patch_size=patch_size,
                pooling=args.pooling,
                min_foreground_fraction=args.min_foreground_fraction,
            )
            pat_ids.extend(batch["pat_id"])
            feature_batches.append(features.float().cpu().numpy())
            print(f"Extracted {len(pat_ids)}/{len(dataset)}", end="\r", flush=True)
    print()

    feature_matrix = np.concatenate(feature_batches, axis=0)
    write_features(args.output, pat_ids, feature_matrix)
    metadata = {
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "output": str(Path(args.output).expanduser().resolve()),
        "image_column": args.image_column,
        "modalities": sorted(modalities),
        "target_size": list(target_size),
        "preprocessing": "asparagus_clsreg_volume_normalization_and_pad_center_crop",
        "pooling": args.pooling,
        "min_foreground_fraction": args.min_foreground_fraction,
        "n_scans": len(pat_ids),
        "n_features": int(feature_matrix.shape[1]),
        "device": str(device),
        "amp_bfloat16": bool(args.amp and device.type == "cuda"),
        "checkpoint_info": checkpoint_info,
    }
    write_metadata(args.output, metadata)
    print(f"Wrote {args.output} ({feature_matrix.shape[0]} x {feature_matrix.shape[1]})")


if __name__ == "__main__":
    main()
