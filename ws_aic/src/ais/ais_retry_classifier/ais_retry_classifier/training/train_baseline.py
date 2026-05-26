from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np

try:
    from ..core.schema import FEATURE_COLUMNS
except ImportError:
    from ais_retry_classifier.core.schema import FEATURE_COLUMNS


DEFAULT_FEATURES = tuple(FEATURE_COLUMNS)


def _to_float(value: str | int | float | None) -> float:
    if value is None or value == "":
        return 0.0
    try:
        number = float(value)
    except ValueError:
        return 0.0
    return 0.0 if math.isnan(number) or math.isinf(number) else number


def load_dataset(path: Path, feature_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("binary_success", "") == "":
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"no labeled rows found in {path}")
    x = np.asarray(
        [[_to_float(row.get(name)) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([int(float(row["binary_success"])) for row in rows], dtype=np.float64)
    return x, y, rows


def train_val_split(n: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_n = max(1, int(round(n * val_ratio))) if n > 1 else 0
    val_idx = np.asarray(indices[:val_n], dtype=np.int64)
    train_idx = np.asarray(indices[val_n:], dtype=np.int64)
    if len(train_idx) == 0:
        train_idx = val_idx
    return train_idx, val_idx


def fit_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-9] = 1.0
    x_norm = (x_train - mean) / std
    weights = np.zeros(x_norm.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(epochs):
        logits = x_norm @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        err = probs - y_train
        grad_w = (x_norm.T @ err) / len(x_norm) + l2 * weights
        grad_b = float(err.mean())
        weights -= lr * grad_w
        bias -= lr * grad_b
    return weights, bias, mean, std


def predict_proba(x: np.ndarray, weights: np.ndarray, bias: float, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    logits = ((x - mean) / std) @ weights + bias
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float | int]:
    pred = (probs >= 0.5).astype(np.float64)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    total = max(1, len(y_true))
    return {
        "accuracy": float((tp + tn) / total),
        "success_precision": float(tp / max(1, tp + fp)),
        "success_recall": float(tp / max(1, tp + fn)),
        "failure_recall": float(tn / max(1, tn + fp)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tabular retry success baseline.")
    parser.add_argument("--csv", required=True, help="Path to features.csv")
    parser.add_argument("--output", default="retry_classifier_model.json")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--features", nargs="*", default=list(DEFAULT_FEATURES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_names = tuple(args.features)
    x, y, _rows = load_dataset(Path(args.csv), feature_names)
    train_idx, val_idx = train_val_split(len(y), args.val_ratio, args.seed)
    weights, bias, mean, std = fit_logistic_regression(
        x[train_idx],
        y[train_idx],
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
    )
    train_metrics = metrics(y[train_idx], predict_proba(x[train_idx], weights, bias, mean, std))
    val_metrics = metrics(y[val_idx], predict_proba(x[val_idx], weights, bias, mean, std)) if len(val_idx) else {}
    model = {
        "model_type": "standardized_logistic_regression",
        "feature_names": feature_names,
        "weights": weights.tolist(),
        "bias": bias,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")
    print(f"model written: {output}")
    print(f"train_metrics={train_metrics}")
    print(f"val_metrics={val_metrics}")


if __name__ == "__main__":
    main()
