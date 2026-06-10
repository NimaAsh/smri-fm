from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from evaluate_dlbs_age import (  # noqa: E402
    age_column_for_scan,
    load_dlbs_age_data,
    parse_modalities,
    parse_pat_id,
)


def test_parse_pat_id() -> None:
    assert parse_pat_id("sub-1003_ses-wave2_acq-MPRAGE_run-1_T1w") == {
        "participant_id": "sub-1003",
        "wave": "wave2",
        "modality": "T1w",
        "tracer": "",
    }
    assert parse_pat_id("sub-1003_ses-wave3_trc-18FAV1451_run-1_pet") == {
        "participant_id": "sub-1003",
        "wave": "wave3",
        "modality": "pet",
        "tracer": "18FAV1451",
    }
    assert parse_pat_id("not-a-dlbs-scan") is None


def test_modality_and_age_source_parsing() -> None:
    assert parse_modalities("T1w,t2w,DWI") == {"T1w", "T2w", "dwi"}
    assert parse_modalities("all") == {"T1w", "T2w", "dwi", "pet"}
    dwi = parse_pat_id("sub-1_ses-wave2_acq-DTI_run-1_dwi")
    tau = parse_pat_id("sub-1_ses-wave3_trc-18FAV1451_run-1_pet")
    assert dwi is not None and age_column_for_scan(dwi) == "AgeMRI_W2"
    assert tau is not None and age_column_for_scan(tau) == "AgePETTau_W3"


def test_adapter_filters_modalities_and_joins_acquisition_age(tmp_path: Path) -> None:
    participants = tmp_path / "participants.tsv"
    features = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "participant_id": ["sub-1"],
            "AgeMRI_W1": [50],
            "AgeMRI_W2": [55],
            "AgeMRI_W3": [60],
            "AgePETAmy_W1": [51],
            "AgePETTau_W3": [61],
        }
    ).to_csv(participants, sep="\t", index=False)
    pd.DataFrame(
        {
            "pat_id": [
                "sub-1_ses-wave1_acq-MPRAGE_run-1_T1w",
                "sub-1_ses-wave3_acq-MPRAGE_run-1_T1w",
                "sub-1_ses-wave1_acq-FLAIR_run-1_T2w",
                "sub-1_ses-wave3_trc-18FAV1451_run-1_pet",
            ],
            "Feature_0": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(features, index=False)

    data = load_dlbs_age_data(features, participants, include_modalities={"T2w", "pet"})

    assert data["groups"] == ["sub-1", "sub-1"]
    assert data["y"].tolist() == [50.0, 61.0]
    assert [row["modality"] for row in data["metadata"]] == ["T2w", "pet"]
    assert [row["age_source"] for row in data["metadata"]] == [
        "AgeMRI_W1",
        "AgePETTau_W3",
    ]
    assert len(data["filtered"]) == 2
