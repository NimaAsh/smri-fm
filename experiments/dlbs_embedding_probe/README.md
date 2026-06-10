# DLBS Frozen-Feature Age Probe

This experiment explores how much age information a frozen representation carries
across the imaging modalities available in the Dallas Lifespan Brain Study (DLBS).
Feature extraction happens once; the probe then operates only on feature vectors,
targets, and subject identifiers without loading or updating the source model.

## DLBS overview

The local collection contains **3,625 NIfTI images from 464 participants** across
three longitudinal waves and 969 distinct participant-wave sessions. File counts
can exceed session counts because some sessions have repeated runs or multiple PET
tracers.

| Modality | Files | Sessions | Subjects | Wave 1 | Wave 2 | Wave 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T1w MPRAGE | 967 | 957 | 464 | 465 | 301 | 201 |
| T2w FLAIR | 968 | 954 | 464 | 468 | 303 | 197 |
| DWI | 955 | 955 | 464 | 464 | 299 | 192 |
| PET | 735 | 612 | 331 | 295 | 240 | 200 |

Coverage is strongly paired: 954 sessions have both T1w and T2w, 953 have T1w,
T2w, and DWI, and 597 have all three MRI modalities plus at least one PET scan.
MRI acquisition ages span 21-97 years (mean about 60.6); PET acquisition ages
span 31-98 years (mean 67.5), so modality comparisons should account for the older
PET subset.

PET contains two tracers:

| Tracer | Interpretation | Files | Subjects | Waves |
| --- | --- | ---: | ---: | --- |
| `18FAV45` | Florbetapir, amyloid | 551 | 295 | W1: 295, W2: 180, W3: 76 |
| `18FAV1451` | Flortaucipir, tau | 184 | 154 | W2: 60, W3: 124 |

The complete inventory comes from `DLBS/images` (T1w) and
`DLBS/openneuro_extra` (T2w/FLAIR, DWI, and PET). Recreate it with:

```bash
uv run python experiments/dlbs_embedding_probe/scripts/build_dlbs_manifest.py
```

This writes `DLBS/dlbs_image_manifest.csv` plus a JSON summary. Each row records
the image path, participant, wave, modality, PET tracer when applicable, and the
age at that acquisition.

## Age labels

- T1w, T2w, and DWI use the wave-matched `AgeMRI_W1/W2/W3` value.
- Amyloid PET (`18FAV45`) uses `AgePETAmy_W1/W2/W3`.
- Tau PET (`18FAV1451`) uses `AgePETTau_W2/W3`.

PET acquisition age is supported for dataset exploration, but PET is not a
structural-MRI input. Amyloid/tau positivity should be implemented as separate
classification targets using curated threshold labels; those labels are not in
the local `participants.tsv`.

## Feature contract

The reusable ridge probe accepts `X` (one feature vector per scan), `y` (age), and
`groups` (participant identifiers). Model wrappers, preprocessing, and feature
extraction remain outside this experiment.

The feature CSV has one row per extracted image:

```text
pat_id,Feature_0,Feature_1,...,Feature_D
sub-1003_ses-wave1_acq-MPRAGE_run-1_T1w,0.12,-0.07,...,1.31
sub-1003_ses-wave1_acq-FLAIR_run-1_T2w,0.08,-0.03,...,1.17
```

`pat_id` identifies the participant, wave, modality, and PET tracer. All selected
rows must use the same feature dimensionality. The manifest is an extraction
inventory; raw images do not enter the probe until a feature extractor has produced
these vectors.

## Asparagus ResEnc-B features

Extract final-stage, globally averaged frozen features from an Asparagus ResEnc-B
checkpoint. The extractor applies the same per-volume normalization and
`128 x 128 x 128` pad/center-crop used by the Asparagus classification/regression
evaluation pipeline.

```bash
uv run python experiments/dlbs_embedding_probe/scripts/extract_resenc_features.py \
    --manifest DLBS/dlbs_image_manifest.csv \
    --checkpoint /path/to/resenc_unet_b.ckpt \
    --modalities T1w \
    --device cuda \
    --amp \
    --output DLBS/features/resenc_b_t1w.csv
```

`--image-column` defaults to the manifest's raw `image_path`. To evaluate a
different preprocessing recipe, add a column containing those NIfTI paths and pass
its name explicitly. Use the same image column, target size, modalities, and probe
seed for every checkpoint comparison.

Inputs must be scalar 3D volumes. Raw 4D DWI series need to be converted to a
defined scalar input such as a b0 or b1000 image first, then referenced through a
separate manifest column.

## Native sMRI-MAE ViT features

Native `smri_mae` checkpoints contain the architecture and encoder weights in the
`.pth` file. This extractor loads that format directly, uses the checkpoint's native
image size, and writes both CLS-token and mean visible-patch embeddings in one pass.
It applies the Asparagus classification/regression volume normalization and
pad/center-crop, then passes only nonzero brain-containing patches to the encoder.

```bash
uv run python experiments/dlbs_embedding_probe/scripts/extract_smri_mae_features.py \
    --manifest DLBS/dlbs_image_manifest.csv \
    --checkpoint /path/to/checkpoint-last.pth \
    --modalities T1w,T2w \
    --device cuda \
    --amp \
    --cls-output DLBS/features/smri_mae_cls.csv \
    --mean-output DLBS/features/smri_mae_mean.csv
```

Omitting `--mask-ratio` evaluates all brain-containing patches. Set it explicitly
only for a controlled masked-input experiment; the value and random seed are written
to metadata. Extraction uses batch size one so variable brain masks do not trim
tokens from other scans in the same batch.

## Run

Evaluate every modality present in a feature CSV:

```bash
uv run python experiments/dlbs_embedding_probe/scripts/evaluate_dlbs_age.py \
    --features DLBS/features.csv \
    --participants DLBS/participants.tsv \
    --modalities all \
    --output-dir DLBS/qc/dlbs_age_all
```

Select or exclude modalities explicitly:

```bash
# Structural MRI only
--modalities T1w,T2w

# All MRI acquisitions, excluding PET
--modalities all --exclude-modalities pet

# PET age exploration, reported separately
--modalities pet
```

Use separate runs when comparing representation quality by modality. A pooled
`--modalities all` run is useful for coverage and robustness checks, but it mixes
repeated participant sessions and modality-specific age distributions.

Outputs include out-of-fold predictions, standardized coefficients, overall and
per-modality metrics, per-wave metrics, skipped rows, and the exact settings used.

## Method and scope

- Subject-grouped outer folds produce out-of-fold predictions.
- Subject-grouped inner folds select ridge regularization by MAE.
- Standardization and the mean baseline use training-fold data only.
- Every scan from a participant stays in the same fold, even when modalities and
  waves are pooled.

The generic implementation is `scripts/linear_probe.py` (206 lines). Dataset
inventory and DLBS label joining are separate dataset-specific modules. Baselines
are valid comparisons only when they use the same cohort, targets, and folds; they
are not dependencies of this probe.
