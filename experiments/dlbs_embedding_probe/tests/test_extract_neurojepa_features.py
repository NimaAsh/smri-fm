from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from extract_neurojepa_features import (  # noqa: E402
    build_backbone_kwargs,
    patch_foreground_mask,
    pool_neurojepa_tokens,
    resolve_target_size,
    validate_args,
)


def test_pool_neurojepa_tokens_mean_and_max() -> None:
    tokens = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 8.0], [6.0, 4.0]],
        ]
    )

    assert torch.equal(
        pool_neurojepa_tokens(tokens, None, (2, 2, 2), "mean"),
        torch.tensor([[2.0, 3.0], [4.0, 6.0]]),
    )
    assert torch.equal(
        pool_neurojepa_tokens(tokens, None, (2, 2, 2), "max"),
        torch.tensor([[3.0, 4.0], [6.0, 8.0]]),
    )


def test_foreground_mean_pools_only_nonzero_patches() -> None:
    tokens = torch.tensor([[[1.0, 10.0], [5.0, 50.0]]])
    image = torch.zeros(1, 1, 4, 2, 2)
    image[:, :, :2] = 1.0

    pooled = pool_neurojepa_tokens(
        tokens,
        image,
        patch_size=(2, 2, 2),
        pooling="foreground_mean",
        min_foreground_fraction=0.1,
    )

    assert torch.equal(patch_foreground_mask(image, (2, 2, 2)), torch.tensor([[True, False]]))
    assert torch.equal(pooled, torch.tensor([[1.0, 10.0]]))


def test_resolve_target_size_defaults_to_checkpoint_geometry() -> None:
    checkpoint_info = {"model": {"img_size": [208, 240, 208]}}

    assert resolve_target_size(SimpleNamespace(target_size=None), checkpoint_info) == (208, 240, 208)

    args = SimpleNamespace(target_size="128,128,128", allow_size_mismatch=False)
    with pytest.raises(ValueError, match="does not match"):
        resolve_target_size(args, checkpoint_info)

    args.allow_size_mismatch = True
    assert resolve_target_size(args, checkpoint_info) == (128, 128, 128)


def test_build_backbone_kwargs_uses_pretrain_config_fields() -> None:
    kwargs = build_backbone_kwargs(
        {
            "meta": {"use_sdpa": True},
            "model": {
                "backbone_name": "vit_base",
                "img_size": [208, 240, 208],
                "patch_size": [16, 16, 16],
                "in_chans": 1,
                "uniform_power": True,
                "use_rope": True,
                "wide_silu": True,
            },
        },
        torch.device("cpu"),
    )

    assert kwargs["model_name"] == "vit_base"
    assert kwargs["img_size"] == [208, 240, 208]
    assert kwargs["patch_size"] == [16, 16, 16]
    assert kwargs["use_sdpa"] is True
    assert kwargs["use_activation_checkpointing"] is False


def test_validate_args_rejects_bad_extraction_options(tmp_path: Path) -> None:
    output = tmp_path / "features.csv"
    output.write_text("pat_id,Feature_0\n")

    args = SimpleNamespace(
        output=str(output),
        overwrite=False,
        batch_size=1,
        num_workers=0,
        min_foreground_fraction=0.1,
    )
    with pytest.raises(FileExistsError):
        validate_args(args)

    args.output = str(tmp_path / "new.csv")
    args.batch_size = 0
    with pytest.raises(ValueError, match="batch-size"):
        validate_args(args)
