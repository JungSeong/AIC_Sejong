from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import build_pose_model
from .model.dataset import (
    DEFAULT_DATASET_ROOT,
    PosePredictionDataset,
    load_samples,
    pose_targets_from_sample,
)


def _ws_aic_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _stats(samples):
    names = ("port0_position", "port1_position", "dyaw")
    values = {name: [] for name in names}
    for sample in samples:
        targets = pose_targets_from_sample(sample)
        for name in names:
            values[name].append(targets[name])
    stats = {}
    for name, tensors in values.items():
        stacked = torch.stack(tensors)
        stats[name] = {
            "mean": stacked.mean(dim=0).tolist() if stacked.ndim > 1 else float(stacked.mean()),
            "std": stacked.std(dim=0, unbiased=False).clamp_min(1e-6).tolist()
            if stacked.ndim > 1
            else float(stacked.std(unbiased=False).clamp_min(1e-6)),
        }
    return stats


def _loss(output, batch):
    return (
        F.smooth_l1_loss(output["port0_position"], batch["port0_position"])
        + F.smooth_l1_loss(output["port1_position"], batch["port1_position"])
        + F.smooth_l1_loss(output["dyaw"], batch["dyaw"])
    )


def _move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train unified AIS pose prediction model.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--versions", nargs="+", default=("v4.0",))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ws_aic_root() / "model" / "ais_pose_prediction" / "pose_resnet50_v4.0",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    train_samples = load_samples(args.dataset_root, versions=args.versions, splits=("train",))
    val_samples = load_samples(args.dataset_root, versions=args.versions, splits=("val",))
    stats = _stats(train_samples)
    target_mean = {name: stats[name]["mean"] for name in stats}
    target_std = {name: stats[name]["std"] for name in stats}
    train_ds = PosePredictionDataset(
        train_samples,
        image_size=args.image_size,
        target_mean=target_mean,
        target_std=target_std,
    )
    val_ds = PosePredictionDataset(
        val_samples,
        image_size=args.image_size,
        target_mean=target_mean,
        target_std=target_std,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = build_pose_model(
        backbone_name=args.backbone,
        pretrained=True,
        hidden_dim=args.hidden_dim,
        num_views=3,
        aggregation="concat",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "versions": list(args.versions),
        "backbone": args.backbone,
        "hidden_dim": args.hidden_dim,
        "image_size": args.image_size,
        "cameras": ["left", "center", "right"],
        "aggregation": "concat",
        "num_views": 3,
        "position_dim": 2,
        "force_torque_dim": 6,
        "joint_dim": 12,
        "target_mean": target_mean,
        "target_std": target_std,
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["image"], batch["force_torque"], batch["joint_positions"])
            loss = _loss(output, batch)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * int(batch["image"].shape[0])
            train_count += int(batch["image"].shape[0])

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.inference_mode():
            for batch in val_loader:
                batch = _move_batch(batch, device)
                output = model(batch["image"], batch["force_torque"], batch["joint_positions"])
                loss = _loss(output, batch)
                val_loss += float(loss.item()) * int(batch["image"].shape[0])
                val_count += int(batch["image"].shape[0])
        train_mean = train_loss / max(train_count, 1)
        val_mean = val_loss / max(val_count, 1)
        print(f"epoch={epoch:03d} train_loss={train_mean:.5f} val_loss={val_mean:.5f}")
        payload = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "metrics": {"train_loss": train_mean, "val_loss": val_mean, "epoch": epoch},
        }
        torch.save(payload, args.output_dir / "last.pt")
        if val_mean < best_val:
            best_val = val_mean
            torch.save(payload, args.output_dir / "best.pt")
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
