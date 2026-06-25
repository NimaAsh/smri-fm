import numpy as np
from datasets import Dataset
from sklearn.model_selection import KFold

from evaluation.tasks.column import ColumnTask


def _data(n=20):
    return Dataset.from_dict(
        {
            "image": [[float(i)] for i in range(n)],
            "age": [40 + i for i in range(n)],
            "sex": ["M", "F"] * (n // 2),
        }
    )


def _task(target="age", kind="regression", n=20, splitter=None):
    return ColumnTask(
        name="t",
        kind=kind,
        data=_data(n),
        splitter=splitter or KFold(n_splits=5, shuffle=True, random_state=0),
        target_column=target,
    )


def test_missing_numeric_targets_dropped():
    data = Dataset.from_dict({"image": [1, 2, 3], "y": [1.0, float("nan"), 3.0]})
    task = ColumnTask(name="t", kind="regression", data=data, splitter=KFold(2), target_column="y")
    assert len(task.data) == 2


def test_missing_string_targets_dropped():
    data = Dataset.from_dict({"image": [1, 2, 3], "y": ["M", None, "F"]})
    task = ColumnTask(
        name="t", kind="classification", data=data, splitter=KFold(2), target_column="y"
    )
    assert len(task.data) == 2


def test_dataset_returns_canonical_columns():
    sample = _task(target="age").dataset()[0]
    assert set(sample) == {"image", "target"}
    assert sample["target"] == 40


def test_split_covers_all_and_is_disjoint():
    folds = list(_task().split())
    assert len(folds) == 5
    assert sum(len(test) for _, test in folds) == 20
    for train, test in folds:
        assert set(train).isdisjoint(test)


def test_fixed_split_list_yields_once():
    split = (np.arange(10), np.arange(10, 20))
    folds = list(_task(splitter=[split]).split())
    assert len(folds) == 1
    assert np.array_equal(folds[0][0], split[0])
    assert np.array_equal(folds[0][1], split[1])


def test_metrics_dispatch_on_kind():
    idx = np.arange(4)
    reg = _task(target="age", kind="regression", n=4)
    cls = _task(target="sex", kind="classification", n=4)
    assert set(reg.metrics(np.zeros(4), np.zeros(4), idx)) == {"mae", "rmse", "r2"}
    assert "balanced_accuracy" in cls.metrics(["M", "F", "M", "F"], ["M", "F", "M", "F"], idx)
