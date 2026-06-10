#!/usr/bin/env python
"""Generic subject-grouped nested-CV ridge probe for precomputed features."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence

import numpy as np


def make_group_folds(groups: Sequence[str], n_folds: int, seed: int) -> list[np.ndarray]:
    """Assign every group to one size-balanced fold."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    group_list = list(groups)
    unique_groups = sorted(set(group_list))
    if len(unique_groups) < n_folds:
        raise ValueError(f"Need at least {n_folds} groups, found {len(unique_groups)}")

    random.Random(seed).shuffle(unique_groups)
    group_sizes = Counter(group_list)
    fold_groups: list[list[str]] = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    for group in sorted(unique_groups, key=group_sizes.get, reverse=True):
        fold_index = min(range(n_folds), key=fold_sizes.__getitem__)
        fold_groups[fold_index].append(group)
        fold_sizes[fold_index] += group_sizes[group]

    group_array = np.asarray(group_list, dtype=object)
    return [np.flatnonzero(np.isin(group_array, fold)) for fold in fold_groups]


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize both arrays using training statistics only."""
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0.0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def fit_ridge(x_train_z: np.ndarray, y_train: np.ndarray, alpha: float):
    """Fit ridge to standardized features with an unpenalized intercept."""
    y_mean = float(y_train.mean())
    xtx = x_train_z.T @ x_train_z
    system = xtx + alpha * np.eye(xtx.shape[0], dtype=float)
    rhs = x_train_z.T @ (y_train - y_mean)
    try:
        beta = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(system, rhs, rcond=None)[0]
    return beta, y_mean


def predict_ridge(x_z: np.ndarray, model) -> np.ndarray:
    beta, intercept = model
    return x_z @ beta + intercept


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    if len(y_true) < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        pearson_r = math.nan
    else:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": 1.0 - float(np.sum(error**2)) / denominator if denominator else math.nan,
        "pearson_r": pearson_r,
        "bias": float(np.mean(error)),
    }


def select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    alphas: Sequence[float],
    n_folds: int,
    seed: int,
) -> tuple[float, dict[str, float]]:
    """Choose ridge alpha by grouped inner-CV MAE."""
    folds = make_group_folds(groups, min(n_folds, len(set(groups))), seed)
    absolute_error = {float(alpha): 0.0 for alpha in alphas}
    n_scored = {float(alpha): 0 for alpha in alphas}
    all_indices = np.arange(len(y))

    for validation_indices in folds:
        train_indices = np.setdiff1d(all_indices, validation_indices, assume_unique=True)
        x_train_z, x_validation_z, _, _ = standardize_train_test(
            x[train_indices], x[validation_indices]
        )
        for alpha in alphas:
            alpha = float(alpha)
            model = fit_ridge(x_train_z, y[train_indices], alpha)
            prediction = predict_ridge(x_validation_z, model)
            absolute_error[alpha] += float(np.abs(prediction - y[validation_indices]).sum())
            n_scored[alpha] += len(validation_indices)

    scores = {str(alpha): absolute_error[alpha] / n_scored[alpha] for alpha in absolute_error}
    best_alpha = min(absolute_error, key=lambda alpha: (scores[str(alpha)], alpha))
    return best_alpha, scores


def _validate_inputs(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    alphas: Sequence[float],
) -> None:
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != len(y) or len(y) != len(groups):
        raise ValueError("Expected x[N,D], y[N], and one group per row")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Features and labels must be finite")
    if len(set(groups)) < 2:
        raise ValueError("At least two groups are required")
    if len(alphas) == 0 or any(alpha < 0 for alpha in alphas):
        raise ValueError("alphas must be a non-empty sequence of non-negative values")


def nested_group_ridge(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    alphas: Sequence[float],
    outer_folds: int = 5,
    inner_folds: int = 5,
    seed: int = 13,
) -> dict[str, object]:
    """Return unbiased predictions from grouped nested cross-validation."""
    _validate_inputs(x, y, groups, alphas)
    folds = make_group_folds(groups, outer_folds, seed)
    predictions = np.full(len(y), np.nan)
    baseline_predictions = np.full(len(y), np.nan)
    fold_numbers = np.full(len(y), -1, dtype=int)
    row_alphas = np.full(len(y), np.nan)
    fold_records = []
    all_indices = np.arange(len(y))

    for fold_number, test_indices in enumerate(folds, start=1):
        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
        train_groups = [groups[index] for index in train_indices]
        alpha, inner_scores = select_alpha(
            x[train_indices],
            y[train_indices],
            train_groups,
            alphas,
            inner_folds,
            seed + fold_number,
        )
        x_train_z, x_test_z, _, _ = standardize_train_test(x[train_indices], x[test_indices])
        fold_prediction = predict_ridge(x_test_z, fit_ridge(x_train_z, y[train_indices], alpha))
        predictions[test_indices] = fold_prediction
        baseline_predictions[test_indices] = y[train_indices].mean()
        fold_numbers[test_indices] = fold_number
        row_alphas[test_indices] = alpha
        fold_records.append(
            {
                "fold": fold_number,
                "alpha": alpha,
                "n_train": int(len(train_indices)),
                "n_test": int(len(test_indices)),
                "n_train_groups": int(len(set(train_groups))),
                "n_test_groups": int(len(set(groups[index] for index in test_indices))),
                "metrics": regression_metrics(y[test_indices], fold_prediction),
                "inner_mae": inner_scores,
            }
        )

    return {
        "predictions": predictions,
        "baseline_predictions": baseline_predictions,
        "fold_numbers": fold_numbers,
        "row_alphas": row_alphas,
        "folds": fold_records,
    }


def refit_group_ridge(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    alphas: Sequence[float],
    inner_folds: int = 5,
    seed: int = 1013,
) -> dict[str, object]:
    """Select alpha and refit on all rows for deployment or inspection."""
    _validate_inputs(x, y, groups, alphas)
    alpha, inner_scores = select_alpha(x, y, groups, alphas, inner_folds, seed)
    x_z, _, mean, std = standardize_train_test(x, x)
    beta, intercept = fit_ridge(x_z, y, alpha)
    return {
        "alpha": alpha,
        "beta": beta,
        "intercept": intercept,
        "feature_mean": mean,
        "feature_std": std,
        "inner_mae": inner_scores,
    }
