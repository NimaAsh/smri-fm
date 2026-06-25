# Evaluation

Internal evaluation suite. Currently only supporting frozen-feature sklearn linear probe.

## Run

```bash
uv run python -m evaluation.main_linear <model> <task> [--config cfg.yaml] [--overrides key=value ...]
# e.g.
uv run python -m evaluation.main_linear smri_mae dlbs_age --overrides model_kwargs.ckpt_path=/path/to/ckpt.pth
```

`model` and `task` are registered names (the CLI `--help` lists them). Run-level
settings come from [config/default_linear.yaml](config/default_linear.yaml),
overridden by an optional `--config` and then dot-list `--overrides`.

Outputs save in `<output_root>/<name>/` (default name `<model>__<task>`):

- `summary.csv`: one row of `model, task, tput, <metric>, <metric>_std`
- `metrics.json`: the summary plus per-fold scores
- `predictions.csv`: out-of-fold `sample_index, y_true, y_pred` rows
- `scatter.png`: regression-only predicted-vs-true scatter plot
- `config.yaml`: the fully resolved config
- `log.txt`: run log

## SLURM Matrix

Submit the standard internal checkpoint matrix from the repo root:

```bash
scripts/internal_evals/submit_linear_matrix.sh
```

By default this evaluates:

- ResEnc checkpoints: official FOMO-MRI AMAES ResEnc-B and Nima's PDF-1M ResEnc-B
- Neuro-JEPA checkpoints: FOMO300 `e60`, `latest`, `cooldown/latest`,
  `faithful_pre96/latest`, and `faithful_pre96/cooldown/latest`
- Tasks: `dlbs_age`, `adni_age`, `adni_sex`, `adni_ad_cn`, `adni_ad_cn_bag`

Useful overrides:

```bash
TASKS="dlbs_age adni_age" scripts/internal_evals/submit_linear_matrix.sh
MODEL_SET=resenc scripts/internal_evals/submit_linear_matrix.sh
OUT_ROOT=/data/$USER/internal_smri_evals_test scripts/internal_evals/submit_linear_matrix.sh
```

## Architecture

- [main_linear.py](main_linear.py) is the main entrypoint
- [models/](models/) contains model wrappers, e.g. [models/smri_mae.py](models/smri_mae.py). Each model defines a transform (`nib.Nifti1Image -> sample dict`) as well as the model itself (`batch dict -> embeddings`).
- [tasks/](tasks/) contains defined tasks, e.g. [tasks/fomo.py](tasks/fomo.py). Each task consists of a dataset as well as defined targets, splits, and scoring metrics.

## Adding things

Tasks and models share a registry: a builder decorated with `@register_task` /
`@register_model`, discovered automatically and constructed by name.

- **Task**: implement the `Task` protocol and decorate a builder with
  `@register_task`. For predicting a column of an HF dataset, use `ColumnTask`
  ([tasks/column.py](tasks/column.py)) with a sklearn splitter. See
  [tasks/dlbs.py](tasks/dlbs.py) and [tasks/fomo.py](tasks/fomo.py).
- **Model**: write a `(Model, Transform)` pair and decorate the builder with
  `@register_model`. See [models/smri_mae.py](models/smri_mae.py).
- **Dataset**: add a reproducible builder returning an HF `Dataset` of niftis +
  metadata next to its task (see `load_dlbs_t1w` / `load_fomo_task3`).
