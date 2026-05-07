from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def _ws_aic_root() -> Path:
    return Path(__file__).resolve().parents[4]


DEFAULT_WEIGHT_ROOT = _ws_aic_root() / "weight" / "ais_distance_prediction"


def _progress_bar(
    iterable,
    *,
    enabled: bool,
    desc: str,
    leave: bool,
):
    if not enabled:
        return iterable
    return tqdm(iterable, desc=desc, leave=leave, dynamic_ncols=True)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def split_samples_by_episode(
    samples: Sequence[Mapping[str, Any]],
    *,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records by episode name to reduce frame-level leakage."""
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to < 1.")

    episodes = sorted({sample["episode_name"] for sample in samples})
    rng = random.Random(seed)
    rng.shuffle(episodes)

    n_total = len(episodes)
    n_test = round(n_total * test_ratio)
    n_val = round(n_total * val_ratio)
    test_episodes = set(episodes[:n_test])
    val_episodes = set(episodes[n_test : n_test + n_val])
    train_episodes = set(episodes[n_test + n_val :])

    train_samples: list[dict[str, Any]] = []
    val_samples: list[dict[str, Any]] = []
    test_samples: list[dict[str, Any]] = []
    for sample in samples:
        episode_name = sample["episode_name"]
        if episode_name in test_episodes:
            test_samples.append(dict(sample))
        elif episode_name in val_episodes:
            val_samples.append(dict(sample))
        elif episode_name in train_episodes:
            train_samples.append(dict(sample))
    return train_samples, val_samples, test_samples


def compute_target_stats(
    samples: Sequence[Mapping[str, Any]],
    target_keys: Sequence[str] = ("x_mm", "y_mm", "z_mm"),
) -> dict[str, torch.Tensor]:
    values = []
    for sample in samples:
        label = sample["label"]["plug_tip_to_port"]
        values.append([float(label[key]) for key in target_keys])
    targets = torch.tensor(values, dtype=torch.float32)
    std = targets.std(dim=0, unbiased=False).clamp_min(1e-6)
    return {"mean": targets.mean(dim=0), "std": std}


def _to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    targets = batch["target"].to(device, non_blocking=True)
    return images, targets


def _denormalize(
    values: torch.Tensor,
    target_mean: torch.Tensor | Sequence[float] | None,
    target_std: torch.Tensor | Sequence[float] | None,
) -> torch.Tensor:
    if target_mean is None or target_std is None:
        return values
    mean = torch.as_tensor(target_mean, dtype=values.dtype, device=values.device)
    std = torch.as_tensor(target_std, dtype=values.dtype, device=values.device)
    return values * std + mean


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    loss_fn: nn.Module | None = None,
    grad_clip_norm: float | None = None,
    show_progress: bool = True,
    progress_desc: str = "train",
) -> dict[str, float]:
    model.train()
    device = torch.device(device)
    loss_fn = loss_fn or nn.SmoothL1Loss()
    total_loss = 0.0
    total_items = 0

    progress = _progress_bar(
        dataloader,
        enabled=show_progress,
        desc=progress_desc,
        leave=False,
    )
    for batch in progress:
        images, targets = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = loss_fn(predictions, targets)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        batch_size = images.shape[0]
        batch_loss = float(loss.detach().cpu())
        total_loss += batch_loss * batch_size
        total_items += batch_size
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{batch_loss:.4f}", avg=f"{total_loss / total_items:.4f}")

    return {"loss": total_loss / max(total_items, 1)}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
    *,
    loss_fn: nn.Module | None = None,
    target_mean: torch.Tensor | Sequence[float] | None = None,
    target_std: torch.Tensor | Sequence[float] | None = None,
    show_progress: bool = True,
    progress_desc: str = "val",
) -> dict[str, float]:
    model.eval()
    device = torch.device(device)
    loss_fn = loss_fn or nn.SmoothL1Loss()

    total_loss = 0.0
    total_items = 0
    absolute_errors = []
    squared_errors = []
    euclidean_errors = []

    progress = _progress_bar(
        dataloader,
        enabled=show_progress,
        desc=progress_desc,
        leave=False,
    )
    for batch in progress:
        images, targets = _to_device(batch, device)
        predictions = model(images)
        loss = loss_fn(predictions, targets)

        raw_targets = batch.get("raw_target", targets).to(device, non_blocking=True)
        raw_predictions = _denormalize(predictions, target_mean, target_std)
        error = raw_predictions - raw_targets

        batch_size = images.shape[0]
        batch_loss = float(loss.detach().cpu())
        total_loss += batch_loss * batch_size
        total_items += batch_size
        absolute_errors.append(error.abs().detach().cpu())
        squared_errors.append(error.square().detach().cpu())
        euclidean_errors.append(torch.linalg.vector_norm(error, dim=1).detach().cpu())
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{batch_loss:.4f}", avg=f"{total_loss / total_items:.4f}")

    metrics = {"loss": total_loss / max(total_items, 1)}
    if absolute_errors:
        abs_error = torch.cat(absolute_errors, dim=0)
        sq_error = torch.cat(squared_errors, dim=0)
        euclidean_error = torch.cat(euclidean_errors, dim=0)
        metrics.update(
            {
                "mae": float(abs_error.mean()),
                "rmse": float(sq_error.mean().sqrt()),
                "euclidean": float(euclidean_error.mean()),
            }
        )
    return metrics


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int,
    metrics: Mapping[str, float],
    config: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": dict(metrics),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if config is not None:
        payload["config"] = dict(config)
    torch.save(payload, path)
    return path


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    epochs: int,
    weight_dir: str | Path = DEFAULT_WEIGHT_ROOT,
    run_name: str = "vision_offset_resnet50",
    loss_fn: nn.Module | None = None,
    target_mean: torch.Tensor | Sequence[float] | None = None,
    target_std: torch.Tensor | Sequence[float] | None = None,
    scheduler: Any | None = None,
    grad_clip_norm: float | None = 1.0,
    config: Mapping[str, Any] | None = None,
    show_progress: bool = True,
) -> list[dict[str, float]]:
    """Train and save best/last checkpoints under ``ws_aic/weight``."""
    device = torch.device(device)
    model.to(device)
    history: list[dict[str, float]] = []
    best_metric = float("inf")
    output_dir = Path(weight_dir) / run_name

    epoch_progress = _progress_bar(
        range(1, epochs + 1),
        enabled=show_progress,
        desc="epochs",
        leave=True,
    )
    for epoch in epoch_progress:
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_fn=loss_fn,
            grad_clip_norm=grad_clip_norm,
            show_progress=show_progress,
            progress_desc=f"train {epoch}/{epochs}",
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            loss_fn=loss_fn,
            target_mean=target_mean,
            target_std=target_std,
            show_progress=show_progress,
            progress_desc=f"val {epoch}/{epochs}",
        )
        if scheduler is not None:
            scheduler.step(val_metrics["loss"])

        row = {
            "epoch": float(epoch),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)

        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_metrics,
            config=config,
        )
        monitor_value = val_metrics.get("euclidean", val_metrics["loss"])
        if monitor_value < best_metric:
            best_metric = monitor_value
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
            )

        if hasattr(epoch_progress, "set_postfix"):
            epoch_progress.set_postfix(
                train_loss=f"{train_metrics['loss']:.4f}",
                val_loss=f"{val_metrics['loss']:.4f}",
                val_euclidean=f"{val_metrics.get('euclidean', float('nan')):.3f}",
            )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_mae={val_metrics.get('mae', float('nan')):.3f} "
            f"val_euclidean={val_metrics.get('euclidean', float('nan')):.3f}"
        )

    return history
