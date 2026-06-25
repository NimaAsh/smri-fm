"""Frozen-feature wrapper for a Neuro-JEPA ViT encoder.

Loads the (frozen) Neuro-JEPA encoder from a pretraining checkpoint + YAML config and
returns one pooled patch-token embedding per scan. Preprocessing matches the asparagus
classification/regression val/test pipeline (see ``_preprocess``).

The Neuro-JEPA package is external: ``repo_path`` must point at a checkout, whose
``src/`` is prepended to ``sys.path`` when the model is built.
"""

from __future__ import annotations

import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor

from evaluation.models._preprocess import AsparagusClsregTransform
from evaluation.models.registry import register_model

_logger = logging.getLogger(__name__)


def _to_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _backbone_kwargs(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    model_cfg = config.get("model") or {}
    meta_cfg = config.get("meta") or {}
    if not isinstance(model_cfg, Mapping) or not isinstance(meta_cfg, Mapping):
        raise ValueError("Neuro-JEPA config must contain mapping-valued model/meta sections")

    name = model_cfg.get("backbone_name", model_cfg.get("model_name", "vit_base"))
    kwargs: dict[str, Any] = {
        "device": device,
        "model_name": name,
        "img_size": model_cfg.get("img_size", [224, 224, 64]),
        "patch_size": model_cfg.get("patch_size", [16, 16, 4]),
        "in_chans": model_cfg.get("in_chans", 1),
        "out_layers": model_cfg.get("out_layers", None),
        "uniform_power": model_cfg.get("uniform_power", False),
        "use_sdpa": meta_cfg.get("use_sdpa", model_cfg.get("use_sdpa", False)),
        "use_silu": model_cfg.get("use_silu", False),
        "wide_silu": model_cfg.get("wide_silu", False),
        "use_rope": model_cfg.get("use_rope", False),
        "use_activation_checkpointing": False,
    }
    if model_cfg.get("use_moe", False):
        kwargs["use_moe"] = True
        kwargs["moe_params"] = _to_namespace(model_cfg.get("moe_params"))
    return kwargs


def _patch_foreground_mask(
    image: Tensor, patch_size: tuple[int, int, int], min_fraction: float
) -> Tensor:
    if image.ndim != 5:
        raise ValueError(f"Expected image [B, C, H, W, D], got {tuple(image.shape)}")
    if any(dim % patch != 0 for dim, patch in zip(image.shape[2:], patch_size)):
        raise ValueError(f"Image {tuple(image.shape[2:])} not divisible by patch {patch_size}")
    foreground = image.abs().amax(dim=1, keepdim=True) > 0
    fraction = F.avg_pool3d(foreground.float(), kernel_size=patch_size, stride=patch_size)
    return fraction.flatten(1) >= min_fraction


def _pool_tokens(
    tokens: Tensor,
    image: Tensor,
    patch_size: tuple[int, int, int],
    pooling: str,
    min_fraction: float,
) -> Tensor:
    if tokens.ndim != 3 or tokens.shape[1] == 0:
        raise ValueError(f"Expected token embeddings [B, L, D], got {tuple(tokens.shape)}")
    if pooling == "mean":
        return tokens.mean(dim=1)
    if pooling == "max":
        return tokens.max(dim=1).values
    if pooling == "foreground_mean":
        mask = _patch_foreground_mask(image, patch_size, min_fraction)
        if mask.shape != tokens.shape[:2]:
            raise ValueError(f"mask {tuple(mask.shape)} != tokens {tuple(tokens.shape[:2])}")
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    raise ValueError(f"Unsupported pooling mode: {pooling}")


class NeuroJepaBackbone(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        patch_size: tuple[int, int, int],
        pooling: str = "mean",
        min_foreground_fraction: float = 0.1,
        amp: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.patch_size = tuple(int(p) for p in patch_size)
        self.pooling = pooling
        self.min_foreground_fraction = min_foreground_fraction
        self.amp = amp

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        image = batch["image"]
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.amp and image.is_cuda
            else nullcontext()
        )
        with ctx:
            tokens, _ = self.backbone(image)
        return _pool_tokens(
            tokens, image, self.patch_size, self.pooling, self.min_foreground_fraction
        ).float()


@register_model
def neurojepa(
    ckpt_path: str,
    config_path: str,
    repo_path: str,
    checkpoint_key: str = "encoder",
    pooling: str = "mean",
    min_foreground_fraction: float = 0.1,
    target_size: tuple[int, int, int] | None = None,
    strict: bool = False,
    amp: bool = False,
):
    """Frozen Neuro-JEPA ViT encoder: pooled patch-token embeddings.

    ``checkpoint_key`` selects the online ``encoder`` (default) or the EMA
    ``target_encoder``. ``pooling`` is ``mean`` (all patches), ``max``, or
    ``foreground_mean`` (patches with nonzero voxels after preprocessing).
    ``target_size`` defaults to the checkpoint's native ``img_size``.
    """
    repo = Path(repo_path).expanduser().resolve()
    src = repo / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"Neuro-JEPA repo src not found: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from neurojepa.utils.checkpoint_loader import robust_checkpoint_loader
    from neurojepa.utils.init_utils import (
        _clean_backbone_state_dict,
        _extract_backbone_state_dict,
        init_backbone,
    )

    with Path(config_path).expanduser().open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    # Build on CPU; main_linear moves the model to the run device.
    kwargs = _backbone_kwargs(config, torch.device("cpu"))
    backbone = init_backbone(**kwargs)

    ckpt = robust_checkpoint_loader(
        str(Path(ckpt_path).expanduser().resolve()), map_location=torch.device("cpu")
    )
    state_dict = _clean_backbone_state_dict(
        _extract_backbone_state_dict(ckpt, checkpoint_key=checkpoint_key)
    )
    matched = sorted(set(state_dict) & set(backbone.state_dict()))
    if not matched:
        raise ValueError(
            f"No checkpoint keys matched the backbone (checkpoint_key={checkpoint_key!r}). "
            "Check the key and the YAML architecture."
        )
    incompatible = backbone.load_state_dict(state_dict, strict=strict)
    _logger.info(
        "loaded neurojepa backbone: %d keys (missing=%d, unexpected=%d)",
        len(matched),
        len(incompatible.missing_keys),
        len(incompatible.unexpected_keys),
    )
    backbone.requires_grad_(False)
    backbone.eval()

    native_size = tuple(int(s) for s in kwargs["img_size"])
    if target_size is None:
        resolved = native_size
    else:
        resolved = tuple(int(s) for s in target_size)
        if resolved != native_size:
            _logger.warning("target_size %s != checkpoint img_size %s", resolved, native_size)

    model = NeuroJepaBackbone(
        backbone,
        patch_size=kwargs["patch_size"],
        pooling=pooling,
        min_foreground_fraction=min_foreground_fraction,
        amp=amp,
    )
    transform = AsparagusClsregTransform(resolved)
    return model, transform
