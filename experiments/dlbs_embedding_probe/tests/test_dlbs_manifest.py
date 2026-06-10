from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_dlbs_manifest import build_manifest, summarize_manifest  # noqa: E402


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_manifest_includes_all_modalities_and_pet_age_sources(tmp_path: Path) -> None:
    t1_dir = tmp_path / "images"
    extra_dir = tmp_path / "extra"
    participants = tmp_path / "participants.tsv"
    touch(t1_dir / "sub-1_ses-wave1_acq-MPRAGE_run-1_T1w.nii.gz")
    touch(extra_dir / "sub-1/ses-wave1/anat/sub-1_ses-wave1_acq-FLAIR_run-1_T2w.nii.gz")
    touch(extra_dir / "sub-1/ses-wave1/dwi/sub-1_ses-wave1_acq-DTI_run-1_dwi.nii.gz")
    touch(extra_dir / "sub-1/ses-wave1/pet/sub-1_ses-wave1_trc-18FAV45_run-1_pet.nii.gz")
    pd.DataFrame(
        {
            "participant_id": ["sub-1"],
            "AgeMRI_W1": [50],
            "AgePETAmy_W1": [51],
        }
    ).to_csv(participants, sep="\t", index=False)

    rows = build_manifest(t1_dir, extra_dir, participants)
    summary = summarize_manifest(rows)

    assert {row["modality"] for row in rows} == {"T1w", "T2w", "dwi", "pet"}
    assert next(row for row in rows if row["modality"] == "pet")["age"] == 51.0
    assert summary["n_files"] == 4
    assert summary["paired_sessions"]["T1w_T2w_dwi_pet"] == 1
