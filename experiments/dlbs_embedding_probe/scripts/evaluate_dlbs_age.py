#!/usr/bin/env python
"""Run a modality-aware frozen-feature age probe on DLBS."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from linear_probe import nested_group_ridge, refit_group_ridge, regression_metrics


MRI_AGE_COLUMN = {"wave1": "AgeMRI_W1", "wave2": "AgeMRI_W2", "wave3": "AgeMRI_W3"}
PET_AGE_PREFIX = {"18FAV45": "AgePETAmy", "18FAV1451": "AgePETTau"}
PAT_ID_RE = re.compile(
    r"^(?P<participant>sub-[^_]+)_ses-(?P<wave>wave[123])_.*_"
    r"(?P<modality>T1w|T2w|dwi|pet)(?:_0000)?$"
)
DEFAULT_ALPHAS = "0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000,3000,10000"
MODALITIES = {"T1w", "T2w", "dwi", "pet"}


def parse_pat_id(pat_id: str) -> dict[str, str] | None:
    match = PAT_ID_RE.match(pat_id)
    if match is None:
        return None
    tracer_match = re.search(r"(?:^|_)trc-([^_]+)", pat_id)
    return {
        "participant_id": match.group("participant"),
        "wave": match.group("wave"),
        "modality": match.group("modality"),
        "tracer": tracer_match.group(1) if tracer_match else "",
    }


def clean_float(value: object) -> float | None:
    text = str(value).strip()
    if text == "" or text.lower() in {"n/a", "na", "nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def detect_feature_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = [column for column in frame.columns if column.startswith(prefix)]
    if not columns:
        raise ValueError(f"No feature columns start with {prefix!r}")
    return columns


def parse_modalities(value: str) -> set[str]:
    if value.strip().lower() == "all":
        return set(MODALITIES)
    aliases = {"t1w": "T1w", "t2w": "T2w", "dwi": "dwi", "pet": "pet"}
    requested = {aliases.get(item.strip().lower(), item.strip()) for item in value.split(",")}
    unknown = requested - MODALITIES
    if not requested or unknown:
        raise ValueError(f"Unknown modalities: {sorted(unknown)}")
    return requested


def age_column_for_scan(scan: dict[str, str]) -> str | None:
    if scan["modality"] != "pet":
        return MRI_AGE_COLUMN[scan["wave"]]
    prefix = PET_AGE_PREFIX.get(scan["tracer"])
    return f"{prefix}_W{scan['wave'][-1]}" if prefix else None


def load_dlbs_age_data(
    features_path: str | Path,
    participants_path: str | Path,
    feature_prefix: str = "Feature_",
    include_modalities: set[str] | None = None,
    exclude_modalities: set[str] | None = None,
) -> dict[str, object]:
    """Join precomputed features to the acquisition-appropriate DLBS age."""
    participants_frame = pd.read_csv(participants_path, sep="\t", dtype=str)
    if "participant_id" not in participants_frame.columns:
        raise ValueError("participants.tsv must contain participant_id")
    participants = participants_frame.set_index("participant_id").to_dict("index")

    frame = pd.read_csv(features_path)
    if "pat_id" not in frame.columns:
        raise ValueError("Feature CSV must contain pat_id")
    if frame["pat_id"].duplicated().any():
        raise ValueError("Feature CSV contains duplicate pat_id values")
    feature_columns = detect_feature_columns(frame, feature_prefix)
    include_modalities = set(MODALITIES) if include_modalities is None else include_modalities
    exclude_modalities = set() if exclude_modalities is None else exclude_modalities

    metadata = []
    feature_rows = []
    labels = []
    groups = []
    skipped = []
    filtered = []
    for _, record in frame.iterrows():
        pat_id = str(record["pat_id"])
        parsed = parse_pat_id(pat_id)
        if parsed is None:
            skipped.append(f"{pat_id}: invalid DLBS scan id")
            continue
        modality = parsed["modality"]
        if modality not in include_modalities or modality in exclude_modalities:
            filtered.append(pat_id)
            continue
        participant = participants.get(parsed["participant_id"])
        if participant is None:
            skipped.append(f"{pat_id}: participant not found")
            continue
        age_source = age_column_for_scan(parsed)
        age = clean_float(participant.get(age_source)) if age_source else None
        if age is None:
            skipped.append(f"{pat_id}: acquisition age missing")
            continue
        feature_vector = record[feature_columns].to_numpy(dtype=float)
        if not np.all(np.isfinite(feature_vector)):
            skipped.append(f"{pat_id}: non-finite feature value")
            continue

        metadata.append({"pat_id": pat_id, **parsed, "age_source": age_source})
        feature_rows.append(feature_vector)
        labels.append(age)
        groups.append(parsed["participant_id"])

    if not metadata:
        raise ValueError("No usable rows after joining DLBS ages and filtering modalities")
    return {
        "x": np.asarray(feature_rows, dtype=float),
        "y": np.asarray(labels, dtype=float),
        "groups": groups,
        "metadata": metadata,
        "feature_columns": feature_columns,
        "skipped": skipped,
        "filtered": filtered,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", required=True, help="Wide CSV with pat_id and Feature_* columns"
    )
    parser.add_argument("--participants", default="DLBS/participants.tsv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-prefix", default="Feature_")
    parser.add_argument(
        "--modalities",
        default="all",
        help="Comma-separated modalities to include: T1w,T2w,dwi,pet (default: all)",
    )
    parser.add_argument(
        "--exclude-modalities",
        default="",
        help="Comma-separated modalities to remove after --modalities",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alphas = [float(value) for value in args.alphas.split(",") if value.strip()]
    include_modalities = parse_modalities(args.modalities)
    exclude_modalities = (
        parse_modalities(args.exclude_modalities) if args.exclude_modalities else set()
    )
    data = load_dlbs_age_data(
        args.features,
        args.participants,
        args.feature_prefix,
        include_modalities,
        exclude_modalities,
    )
    x = data["x"]
    y = data["y"]
    groups = data["groups"]

    result = nested_group_ridge(
        x,
        y,
        groups,
        alphas,
        outer_folds=min(args.outer_folds, len(set(groups))),
        inner_folds=args.inner_folds,
        seed=args.seed,
    )
    refit = refit_group_ridge(x, y, groups, alphas, args.inner_folds, args.seed + 1000)

    with (output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pat_id",
                "participant_id",
                "wave",
                "modality",
                "tracer",
                "age_source",
                "outer_fold",
                "alpha",
                "age",
                "prediction",
            ]
        )
        for row, age, prediction, fold, alpha in zip(
            data["metadata"],
            y,
            result["predictions"],
            result["fold_numbers"],
            result["row_alphas"],
        ):
            writer.writerow(
                [
                    row["pat_id"],
                    row["participant_id"],
                    row["wave"],
                    row["modality"],
                    row["tracer"],
                    row["age_source"],
                    int(fold),
                    f"{alpha:.8g}",
                    f"{age:.8g}",
                    f"{prediction:.8g}",
                ]
            )

    coefficients = [
        {
            "feature": feature,
            "coefficient_standardized": float(beta),
            "training_mean": float(mean),
            "training_std": float(std),
        }
        for feature, beta, mean, std in zip(
            data["feature_columns"],
            refit["beta"],
            refit["feature_mean"],
            refit["feature_std"],
        )
    ]
    coefficients.sort(key=lambda row: abs(row["coefficient_standardized"]), reverse=True)
    with (output_dir / "coefficients.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coefficients[0]))
        writer.writeheader()
        writer.writerows(coefficients)

    by_wave = {}
    for wave in sorted({row["wave"] for row in data["metadata"]}):
        indices = np.asarray(
            [index for index, row in enumerate(data["metadata"]) if row["wave"] == wave]
        )
        by_wave[wave] = {
            **regression_metrics(y[indices], result["predictions"][indices]),
            "n": int(len(indices)),
        }

    by_modality = {}
    for modality in sorted({row["modality"] for row in data["metadata"]}):
        indices = np.asarray(
            [index for index, row in enumerate(data["metadata"]) if row["modality"] == modality]
        )
        by_modality[modality] = {
            **regression_metrics(y[indices], result["predictions"][indices]),
            "n": int(len(indices)),
            "n_subjects": int(
                len({data["metadata"][index]["participant_id"] for index in indices})
            ),
        }

    selected_modalities = include_modalities - exclude_modalities
    observed_modalities = {row["modality"] for row in data["metadata"]}
    metrics = {
        "settings": {
            "features": str(args.features),
            "participants": str(args.participants),
            "outer_folds": min(args.outer_folds, len(set(groups))),
            "inner_folds": args.inner_folds,
            "alphas": alphas,
            "seed": args.seed,
            "n_features": int(x.shape[1]),
            "modalities": sorted(selected_modalities),
            "observed_modalities": sorted(observed_modalities),
        },
        "data": {
            "n_scans": int(len(y)),
            "n_subjects": int(len(set(groups))),
            "n_skipped": int(len(data["skipped"])),
            "n_filtered": int(len(data["filtered"])),
            "n_scans_by_modality": dict(
                sorted(Counter(row["modality"] for row in data["metadata"]).items())
            ),
        },
        "nested_group_cv": regression_metrics(y, result["predictions"]),
        "outer_train_mean_baseline": regression_metrics(y, result["baseline_predictions"]),
        "by_wave": by_wave,
        "by_modality": by_modality,
        "outer_folds": result["folds"],
        "final_refit_alpha": refit["alpha"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    (output_dir / "skipped.txt").write_text(
        "\n".join(data["skipped"]) + ("\n" if data["skipped"] else "")
    )

    cv = metrics["nested_group_cv"]
    print(f"Scans: {len(y)} from {len(set(groups))} subjects")
    print(f"Modalities present: {', '.join(sorted(observed_modalities))}")
    print(
        f"Nested grouped CV: MAE={cv['mae']:.3f}  RMSE={cv['rmse']:.3f}  "
        f"R2={cv['r2']:.3f}  r={cv['pearson_r']:.3f}"
    )
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
