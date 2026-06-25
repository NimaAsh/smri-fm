import argparse
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from evaluation.models.base import Model, Transform
from evaluation.models.registry import create_model, list_models
from evaluation.tasks.base import Task
from evaluation.tasks.registry import create_task, list_tasks

# Large datasets + multi-worker DataLoaders exhaust file descriptors under torch's
# default "file_descriptor" sharing strategy ("Too many open files"); the file-system
# strategy shares worker tensors by name instead of by fd.
torch.multiprocessing.set_sharing_strategy("file_system")

DEFAULT_CONFIG = Path(__file__).parent / "config/default_linear.yaml"

logger = logging.getLogger(__name__)


# Estimators, keyed by task kind. Hyperparameters are selected by inner CV.
def fit_ridge(X: np.ndarray, y: np.ndarray, seed: int) -> Pipeline:
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model = Pipeline([("scaler", StandardScaler()), ("ridge", ridge)])
    return model.fit(X, y)


def fit_logistic(X: np.ndarray, y: np.ndarray, seed: int) -> Pipeline:
    clf = LogisticRegressionCV(Cs=10, scoring="balanced_accuracy", max_iter=1000, random_state=seed)
    model = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return model.fit(X, y)


ESTIMATORS = {"regression": fit_ridge, "classification": fit_logistic}


class TransformDataset(Dataset):
    """Applies the model transform to each canonical sample for batched loading."""

    def __init__(self, dataset: Dataset, transform: Transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        img = sample["image"]
        target = sample["target"]
        sample = self.transform(img)
        sample["target"] = target
        return sample


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


@torch.inference_mode()
def extract_features(
    model: Model,
    dataset: Dataset,
    transform: Transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TransformDataset(dataset, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model.eval()

    features = []
    targets = []
    for batch in loader:
        targets.extend(batch.pop("target"))
        batch = to_device(batch, device)
        embeddings = model(batch)
        features.append(embeddings.cpu().float())

    X = torch.cat(features).numpy()
    y = np.asarray(targets)
    return X, y


def run_linear(
    task: Task,
    model: Model,
    transform: Transform,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    logger.info("extracting features...")
    dataset = task.dataset()
    start = time.perf_counter()
    X, y = extract_features(model, dataset, transform, device, batch_size, num_workers)
    tput = len(y) / (time.perf_counter() - start)
    logger.info(f"features: {tuple(X.shape)} ({tput:.2f} samples/s)")

    fit = ESTIMATORS[task.kind]

    fold_metrics = []
    prediction_rows = []
    for fold, (train_idx, test_idx) in enumerate(task.split()):
        estimator = fit(X[train_idx], y[train_idx], seed)
        pred = estimator.predict(X[test_idx])
        scores = task.metrics(y[test_idx], pred, test_idx)
        fold_metrics.append(scores)
        for sample_index, y_true, y_pred in zip(test_idx, y[test_idx], pred):
            prediction_rows.append(
                {
                    "fold": fold,
                    "sample_index": int(sample_index),
                    "y_true": y_true.item() if hasattr(y_true, "item") else y_true,
                    "y_pred": y_pred.item() if hasattr(y_pred, "item") else y_pred,
                }
            )
        logger.info(f"fold {fold}: " + " ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    metrics = {
        "tput": tput,
        "summary": aggregate_folds(fold_metrics),
        "folds": fold_metrics,
        "predictions": prediction_rows,
    }
    return metrics


def aggregate_folds(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    summary = {}
    for key in fold_metrics[0]:
        values = np.array([fold[key] for fold in fold_metrics])
        summary[key] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    return summary


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_logging(run_dir: Path) -> None:
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "log.txt")]
    formatter = logging.Formatter("%(message)s")

    root = logging.getLogger("evaluation")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.propagate = False


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_regression_scatter(predictions: pd.DataFrame, summary: dict[str, float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = pd.to_numeric(predictions["y_true"], errors="coerce")
    y_pred = pd.to_numeric(predictions["y_pred"], errors="coerce")
    valid = y_true.notna() & y_pred.notna()
    y_true = y_true[valid].to_numpy()
    y_pred = y_pred[valid].to_numpy()
    if len(y_true) == 0:
        return

    lower = float(min(y_true.min(), y_pred.min()))
    upper = float(max(y_true.max(), y_pred.max()))
    margin = 0.05 * max(upper - lower, 1.0)
    lower -= margin
    upper += margin

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=20, alpha=0.65)
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1, linestyle="--")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    title_bits = []
    for key in ("mae", "rmse", "r2"):
        if key in summary:
            title_bits.append(f"{key.upper()}={summary[key]:.3f}")
    if title_bits:
        ax.set_title("  ".join(title_bits))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(
    model_name: str,
    task_name: str,
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict:
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if config_path:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(config_path))
    if overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(overrides))

    set_seed(cfg.seed)

    cfg.model = model_name
    cfg.task = task_name
    cfg.name = cfg.name or f"{cfg.model}__{cfg.task}"

    run_dir = Path(cfg.output_root) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)

    logger.info(f"run: {cfg.name}")
    logger.info(f"git sha: {git_sha()}")
    logger.info(f"config:\n{OmegaConf.to_yaml(cfg).rstrip()}")
    OmegaConf.save(cfg, run_dir / "config.yaml")

    device = torch.device(cfg.device)

    task = create_task(cfg.task)
    model, transform = create_model(cfg.model, **(cfg.model_kwargs or {}))
    model.to(device)

    logger.info(f"task: {cfg.task} ({task.kind})")
    logger.info(f"model: {cfg.model}")
    logger.info(f"dataset: {len(task.dataset())} samples")

    metrics = run_linear(
        task,
        model,
        transform,
        device=device,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    predictions = pd.DataFrame(metrics.pop("predictions"))
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    if task.kind == "regression":
        write_regression_scatter(predictions, metrics["summary"], run_dir / "scatter.png")

    metrics = {"model": cfg.model, "task": cfg.task, **metrics}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    summary = pd.DataFrame(
        [{"model": cfg.model, "task": cfg.task, "tput": metrics["tput"], **metrics["summary"]}]
    )
    summary.to_csv(run_dir / "summary.csv", index=False)
    logger.info(f"summary:\n\n{summary.to_markdown(index=False, floatfmt='.4f')}")
    return metrics


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help=f"[{', '.join(list_models())}]")
    parser.add_argument("task", type=str, help=f"[{', '.join(list_tasks())}]")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    main(args.model, args.task, args.config, args.overrides)


if __name__ == "__main__":
    cli()
