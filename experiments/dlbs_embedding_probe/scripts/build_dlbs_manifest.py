#!/usr/bin/env python
"""Build a modality-aware manifest for the local DLBS image collection."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


IMAGE_RE = re.compile(
    r"^(?P<participant>sub-[^_]+)_ses-(?P<wave>wave[123])_"
    r"(?P<body>.+)_(?P<modality>T1w|T2w|dwi|pet)$"
)
PET_AGE_PREFIX = {"18FAV45": "AgePETAmy", "18FAV1451": "AgePETTau"}


def strip_image_suffix(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else name[:-4]


def clean_float(value: object) -> float | None:
    text = str(value).strip()
    if text == "" or text.lower() in {"n/a", "na", "nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_image(path: Path) -> dict[str, str] | None:
    pat_id = strip_image_suffix(path)
    match = IMAGE_RE.match(pat_id)
    if match is None:
        return None
    body = match.group("body")
    acquisition_match = re.search(r"(?:^|_)acq-([^_]+)", body)
    tracer_match = re.search(r"(?:^|_)trc-([^_]+)", body)
    return {
        "pat_id": pat_id,
        "participant_id": match.group("participant"),
        "wave": match.group("wave"),
        "modality": match.group("modality"),
        "acquisition": acquisition_match.group(1) if acquisition_match else "",
        "tracer": tracer_match.group(1) if tracer_match else "",
        "image_path": path.as_posix(),
    }


def age_column(modality: str, tracer: str, wave: str) -> str | None:
    wave_number = wave[-1]
    if modality != "pet":
        return f"AgeMRI_W{wave_number}"
    prefix = PET_AGE_PREFIX.get(tracer)
    return f"{prefix}_W{wave_number}" if prefix else None


def load_participants(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return {row["participant_id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def build_manifest(
    t1_dir: str | Path,
    extra_dir: str | Path,
    participants_path: str | Path,
) -> list[dict[str, object]]:
    """Return one manifest row per NIfTI file across all available modalities."""
    participants = load_participants(participants_path)
    paths = list(Path(t1_dir).glob("*.nii*"))
    paths.extend(Path(extra_dir).glob("sub-*/ses-*/*/*.nii*"))
    rows = []
    for path in sorted(paths):
        parsed = parse_image(path)
        if parsed is None:
            continue
        column = age_column(parsed["modality"], parsed["tracer"], parsed["wave"])
        participant = participants.get(parsed["participant_id"], {})
        age = clean_float(participant.get(column)) if column else None
        rows.append({**parsed, "age": age, "age_source": column or ""})

    if not rows:
        raise ValueError("No DLBS NIfTI images found")
    pat_ids = [str(row["pat_id"]) for row in rows]
    if len(pat_ids) != len(set(pat_ids)):
        raise ValueError("DLBS image inventory contains duplicate pat_id values")
    return rows


def summarize_manifest(rows: list[dict[str, object]]) -> dict[str, object]:
    """Summarize files, subjects, waves, sessions, tracers, and paired coverage."""
    modality_summary = {}
    session_sets = {}
    for modality in ("T1w", "T2w", "dwi", "pet"):
        selected = [row for row in rows if row["modality"] == modality]
        ages = [float(row["age"]) for row in selected if row["age"] is not None]
        sessions = {(row["participant_id"], row["wave"]) for row in selected}
        session_sets[modality] = sessions
        modality_summary[modality] = {
            "n_files": len(selected),
            "n_sessions": len(sessions),
            "n_subjects": len({row["participant_id"] for row in selected}),
            "files_by_wave": dict(sorted(Counter(row["wave"] for row in selected).items())),
            "n_with_age": len(ages),
            "age_min": min(ages) if ages else None,
            "age_max": max(ages) if ages else None,
            "age_mean": sum(ages) / len(ages) if ages else None,
        }

    tracer_summary = {}
    for tracer in sorted({str(row["tracer"]) for row in rows if row["tracer"]}):
        selected = [row for row in rows if row["tracer"] == tracer]
        ages = [float(row["age"]) for row in selected if row["age"] is not None]
        tracer_summary[tracer] = {
            "n_files": len(selected),
            "n_sessions": len({(row["participant_id"], row["wave"]) for row in selected}),
            "n_subjects": len({row["participant_id"] for row in selected}),
            "files_by_wave": dict(sorted(Counter(row["wave"] for row in selected).items())),
            "n_with_age": len(ages),
            "age_min": min(ages) if ages else None,
            "age_max": max(ages) if ages else None,
            "age_mean": sum(ages) / len(ages) if ages else None,
        }

    all_sessions = set().union(*session_sets.values())
    return {
        "n_files": len(rows),
        "n_subjects": len({row["participant_id"] for row in rows}),
        "n_subject_wave_sessions": len(all_sessions),
        "modalities": modality_summary,
        "pet_tracers": tracer_summary,
        "paired_sessions": {
            "T1w_T2w": len(session_sets["T1w"] & session_sets["T2w"]),
            "T1w_T2w_dwi": len(
                session_sets["T1w"] & session_sets["T2w"] & session_sets["dwi"]
            ),
            "T1w_T2w_dwi_pet": len(set.intersection(*session_sets.values())),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1-dir", default="DLBS/images")
    parser.add_argument("--extra-dir", default="DLBS/openneuro_extra")
    parser.add_argument("--participants", default="DLBS/participants.tsv")
    parser.add_argument("--output", default="DLBS/dlbs_image_manifest.csv")
    parser.add_argument("--summary", default="DLBS/dlbs_image_manifest_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_manifest(args.t1_dir, args.extra_dir, args.participants)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_manifest(rows)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output_path} ({summary['n_files']} images)")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
