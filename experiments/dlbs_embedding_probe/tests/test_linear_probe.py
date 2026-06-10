from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from linear_probe import (  # noqa: E402
    fit_ridge,
    make_group_folds,
    nested_group_ridge,
    predict_ridge,
    standardize_train_test,
)


def test_group_folds_are_deterministic_and_keep_subjects_together() -> None:
    groups = [f"subject-{index // 3}" for index in range(30)]
    first = make_group_folds(groups, n_folds=5, seed=13)
    second = make_group_folds(groups, n_folds=5, seed=13)

    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert sorted(np.concatenate(first).tolist()) == list(range(len(groups)))
    for subject in set(groups):
        subject_rows = {index for index, group in enumerate(groups) if group == subject}
        assert sum(bool(subject_rows & set(fold)) for fold in first) == 1


def test_standardization_uses_training_statistics_only() -> None:
    train = np.asarray([[0.0], [2.0]])
    test = np.asarray([[101.0]])
    train_z, test_z, mean, std = standardize_train_test(train, test)

    np.testing.assert_allclose(mean, [1.0])
    np.testing.assert_allclose(std, [1.0])
    np.testing.assert_allclose(train_z[:, 0], [-1.0, 1.0])
    np.testing.assert_allclose(test_z[:, 0], [100.0])


def test_ridge_matches_sklearn_reference() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 6))
    y = rng.normal(size=80)
    x_z, _, _, _ = standardize_train_test(x, x)

    prediction = predict_ridge(x_z, fit_ridge(x_z, y, alpha=3.0))
    reference = Ridge(alpha=3.0).fit(x_z, y).predict(x_z)

    np.testing.assert_allclose(prediction, reference, atol=1e-10)


def test_nested_probe_uses_grouped_folds_and_train_only_baseline() -> None:
    rng = np.random.default_rng(4)
    groups = [f"subject-{index // 2}" for index in range(24)]
    x = rng.normal(size=(24, 4))
    y = 5.0 + 2.0 * x[:, 0] - x[:, 1]
    result = nested_group_ridge(x, y, groups, [0.01, 0.1, 1.0], 3, 2, 13)

    assert np.all(np.isfinite(result["predictions"]))
    for subject in set(groups):
        rows = [index for index, group in enumerate(groups) if group == subject]
        assert len(set(result["fold_numbers"][rows])) == 1
    for fold in np.unique(result["fold_numbers"]):
        test = result["fold_numbers"] == fold
        np.testing.assert_allclose(result["baseline_predictions"][test], y[~test].mean())
