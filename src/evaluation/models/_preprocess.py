"""Shared frozen-feature preprocessing for asparagus-based eval models.

Mirrors the asparagus classification/regression *val/test* pipeline
(``CPU_clsreg_val_test_transforms_crop``): per-volume intensity normalization plus a
fixed pad/center-crop to ``target_size``. The ResEnc and Neuro-JEPA model wrappers
both build their transform from here so frozen-feature probes are directly
comparable across checkpoints.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import torch
from torch import Tensor


class AsparagusClsregTransform:
    """``nib.Nifti1Image -> {"image": Tensor[C, X, Y, Z]}`` via the asparagus clsreg preset.

    The asparagus preset expects a sample dict ``{"image", "transforms_applied"}`` with a
    channel-first volume, so we load the nifti, add a channel dim, run the preset, and
    return only the (batch-collatable) image tensor.
    """

    def __init__(self, target_size: tuple[int, int, int] = (128, 128, 128)):
        self.target_size = tuple(int(s) for s in target_size)
        # asparagus is heavy; import only when a transform is actually constructed
        # (i.e. when a model is built), not at model-registry import time.
        from asparagus.modules.transforms.presets import CPU_clsreg_val_test_transforms_crop

        self._transform = CPU_clsreg_val_test_transforms_crop(target_size=self.target_size)

    def __call__(self, img: nib.Nifti1Image) -> dict[str, Tensor]:
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D image, got shape {data.shape}")
        if not np.all(np.isfinite(data)):
            raise ValueError("Image contains non-finite values")

        sample = {"image": torch.from_numpy(data.copy()).unsqueeze(0), "transforms_applied": {}}
        sample = self._transform(sample)
        return {"image": sample["image"]}
