from __future__ import annotations

import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from extract_resenc_features import (  # noqa: E402
    DlbsNiftiDataset,
    load_checkpoint_state_dict,
    load_manifest,
    parse_modalities,
    parse_target_size,
    pool_resenc_features,
    write_features,
)


def test_manifest_filtering_and_parsers(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "pat_id": ["scan-t1", "scan-t2", "scan-pet"],
            "modality": ["T1w", "T2w", "pet"],
            "model_input": ["/t1.nii.gz", "/t2.nii.gz", "/pet.nii.gz"],
        }
    ).to_csv(manifest, index=False)

    rows = load_manifest(manifest, parse_modalities("t1w,T2W"), "model_input")

    assert [row["pat_id"] for row in rows] == ["scan-t1", "scan-t2"]
    assert [row["image_path"] for row in rows] == ["/t1.nii.gz", "/t2.nii.gz"]
    assert parse_target_size("128, 128,128") == (128, 128, 128)
    with pytest.raises(ValueError, match="three positive integers"):
        parse_target_size("128,128")


def test_pooling_uses_deepest_encoder_stage() -> None:
    shallow = torch.zeros(2, 2, 4, 4, 4)
    deepest = torch.arange(2 * 3 * 2 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2, 2)

    pooled = pool_resenc_features([shallow, deepest])

    assert pooled.shape == (2, 3)
    assert torch.allclose(pooled, deepest.mean(dim=(2, 3, 4)))


def test_nifti_dataset_applies_asparagus_crop(tmp_path: Path) -> None:
    from asparagus.modules.transforms.presets import CPU_clsreg_val_test_transforms_crop

    image_path = tmp_path / "scan.nii.gz"
    nib.save(
        nib.Nifti1Image(np.random.default_rng(1).normal(size=(20, 24, 28)), np.eye(4)), image_path
    )
    dataset = DlbsNiftiDataset(
        [{"pat_id": "scan", "modality": "T1w", "image_path": str(image_path)}],
        CPU_clsreg_val_test_transforms_crop(target_size=(16, 16, 16)),
    )

    sample = dataset[0]

    assert sample["image"].shape == (1, 16, 16, 16)
    assert torch.isfinite(sample["image"]).all()


def test_load_checkpoint_state_dict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    expected = {"model.encoder.weight": torch.ones(2, 2)}
    torch.save({"state_dict": expected, "global_step": 10, "epoch": 2}, checkpoint)

    actual = load_checkpoint_state_dict(checkpoint)

    assert torch.equal(actual["model.encoder.weight"], expected["model.encoder.weight"])


def test_write_features_matches_probe_contract(tmp_path: Path) -> None:
    output = tmp_path / "features.csv"
    write_features(output, ["scan-a", "scan-b"], np.asarray([[1.0, 2.0], [3.0, 4.0]]))

    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0]) == ["pat_id", "Feature_0", "Feature_1"]
    assert rows[1] == {"pat_id": "scan-b", "Feature_0": "3", "Feature_1": "4"}
