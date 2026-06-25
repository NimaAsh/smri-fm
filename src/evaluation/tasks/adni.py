import numpy as np
import datasets as hfds
from datasets import Dataset, load_dataset
from sklearn.model_selection import KFold, StratifiedKFold

from evaluation.tasks.brain_age_gap import BrainAgeGapTask
from evaluation.tasks.column import ColumnTask
from evaluation.tasks.registry import register_task

REPO_ID = "medarc/adni_eval"


def load_adni_eval() -> Dataset:
    # The published dataset ships a single split with no train/val/test partition,
    # so concatenate whatever splits exist and keep one scan per subject. Tasks then
    # do subject-exclusive cross-validation (one row per subject => leakage-free folds).
    dataset_dict = load_dataset(REPO_ID)
    dataset = hfds.concatenate_datasets(list(dataset_dict.values()))
    dataset = drop_duplicates(dataset, key="participant_id")
    return dataset


def drop_duplicates(dataset: Dataset, key: str) -> Dataset:
    values = np.asarray(dataset[key])
    _, indices = np.unique(values, return_index=True)
    if len(indices) < len(values):
        dataset = dataset.select(np.sort(indices))
    return dataset


@register_task
def adni_age(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="adni_age",
        kind="regression",
        data=load_adni_eval(),
        splitter=KFold(n_splits=n_splits, shuffle=True, random_state=seed),
        image_column="nifti",
        target_column="age",
    )


@register_task
def adni_sex(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="adni_sex",
        kind="classification",
        data=load_adni_eval(),
        splitter=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        image_column="nifti",
        target_column="sex",
    )


@register_task
def adni_ad_cn(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    # NB: target is the 3-class diagnosis (CN/MCI/AD); this is not yet a binary AD-vs-CN task.
    return ColumnTask(
        name="adni_ad_cn",
        kind="classification",
        data=load_adni_eval(),
        splitter=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        image_column="nifti",
        target_column="diagnosis",
    )


@register_task
def adni_synthseg_volumes(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="adni_synthseg_volumes",
        kind="regression",
        data=load_adni_eval(),
        splitter=KFold(n_splits=n_splits, shuffle=True, random_state=seed),
        image_column="nifti",
        target_column="synthseg_volumes",
    )


@register_task
def adni_ad_cn_bag() -> BrainAgeGapTask:
    return BrainAgeGapTask(
        name="adni_ad_cn_bag",
        data=load_adni_eval(),
        age_column="age",
        dx_column="diagnosis",
        control_label=0,
        case_label=1,
        image_column="nifti",
    )
