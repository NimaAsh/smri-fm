from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from extract_smri_mae_features import pool_mae_features, validate_args  # noqa: E402


def test_pool_mae_features_returns_cls_and_patch_mean() -> None:
    cls = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    patches = torch.tensor(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ]
    )

    pooled = pool_mae_features((cls, None, patches))

    assert torch.equal(pooled["cls"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(pooled["mean"], torch.tensor([[3.0, 5.0], [4.0, 6.0]]))


def test_pool_mae_features_requires_class_token() -> None:
    with pytest.raises(ValueError, match="class token"):
        pool_mae_features((None, None, torch.ones(1, 2, 3)))


def test_validate_args_requires_output_and_valid_mask_ratio() -> None:
    args = SimpleNamespace(cls_output=None, mean_output=None, mask_ratio=None, overwrite=False)
    with pytest.raises(ValueError, match="At least one"):
        validate_args(args)

    args.cls_output = "features.csv"
    args.mask_ratio = 1.0
    with pytest.raises(ValueError, match="mask-ratio"):
        validate_args(args)
