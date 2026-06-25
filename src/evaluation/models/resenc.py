"""Frozen-feature wrapper for an asparagus ResEnc-B (clsreg) encoder.

Extracts final-stage, globally averaged encoder features. Preprocessing matches the
asparagus classification/regression val/test pipeline (see ``_preprocess``).
"""

from __future__ import annotations

import logging
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from evaluation.models._preprocess import AsparagusClsregTransform
from evaluation.models.registry import register_model

_logger = logging.getLogger(__name__)


def _load_state_dict(ckpt_path: str) -> dict[str, Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        _logger.info(
            "loaded resenc weights (step=%s, epoch=%s)",
            ckpt.get("global_step", "?"),
            ckpt.get("epoch", "?"),
        )
        return ckpt["state_dict"]
    if "network_weights" in ckpt:
        _logger.info("loaded resenc weights from external checkpoint (network_weights)")
        return ckpt["network_weights"]
    raise ValueError("Unsupported checkpoint: expected a 'state_dict' or 'network_weights' key")


def _pool(encoded) -> Tensor:
    """Global-average-pool the deepest encoder stage -> (B, C)."""
    deepest = encoded[-1] if isinstance(encoded, (list, tuple)) else encoded
    if deepest.ndim != 5:
        raise ValueError(f"Expected ResEnc features [B, C, X, Y, Z], got {tuple(deepest.shape)}")
    return F.adaptive_avg_pool3d(deepest, output_size=1).flatten(1)


class ResEncBackbone(nn.Module):
    def __init__(self, module: nn.Module, amp: bool = False):
        super().__init__()
        self.module = module
        self.amp = amp

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        image = batch["image"]
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.amp and image.is_cuda
            else nullcontext()
        )
        with ctx:
            encoded = self.module.model._encode(image)
        return _pool(encoded).float()


@register_model
def resenc(
    ckpt_path: str,
    target_size: tuple[int, int, int] = (128, 128, 128),
    amp: bool = False,
):
    """Frozen asparagus ResEnc-B encoder.

    ``ckpt_path`` accepts an asparagus ``.ckpt`` (``state_dict`` key) or an external
    checkpoint (``network_weights`` key). Only the encoder is loaded; the decoder and
    stem-repeat logic are disabled to keep the single-channel input contract.
    """
    from asparagus.modules.lightning_modules.base_module import BaseModule
    from asparagus.modules.networks.resenc_unet import resenc_unet_b_clsreg

    class _FrozenModule(BaseModule):
        def training_step(self, *args, **kwargs):  # pragma: no cover - inference only
            raise RuntimeError("ResEnc frozen-feature module is inference-only")

        def validation_step(self, *args, **kwargs):  # pragma: no cover - inference only
            raise RuntimeError("ResEnc frozen-feature module is inference-only")

    net = resenc_unet_b_clsreg(
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
    module = _FrozenModule(
        model=net,
        weights=_load_state_dict(ckpt_path),
        load_decoder=False,
        repeat_stem_weights=False,
    )
    module.requires_grad_(False)
    module.eval()

    backbone = ResEncBackbone(module, amp=amp)
    transform = AsparagusClsregTransform(target_size)
    return backbone, transform
