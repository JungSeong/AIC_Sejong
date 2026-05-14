from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from ..core.config import SRC_ROOT, SfpGrvsConfig

AIS_ROOT = SRC_ROOT / "ais"
if str(AIS_ROOT) not in sys.path:
    sys.path.append(str(AIS_ROOT))
AIS_TRANSFORM_ROOT = SRC_ROOT / "ais" / "ais_transform"
if str(AIS_TRANSFORM_ROOT) not in sys.path:
    sys.path.append(str(AIS_TRANSFORM_ROOT))

from ..data.rotation_dataset import SfpRotationDataset  # noqa: E402
from .train_distance import (  # noqa: E402
    DEFAULT_BACKBONE_CHECKPOINT,
    _image_transform,
    _infer_hidden_dim,
    _infer_num_port_heads,
)
from ais_distance_prediction.model.position_model import (  # noqa: E402
    build_resnet_position_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an SFP rotation residual model on GRVS approach samples."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=SfpGrvsConfig.ROTATION_DATASET_DIR,
        help="GRVS rotation dataset root containing samples.jsonl.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("AIC_DISTANCE_MODEL_PATH", DEFAULT_BACKBONE_CHECKPOINT)),
        help="Backbone checkpoint to initialize from.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SfpGrvsConfig.ROTATION_MODEL_DIR / "best.pt",
        help="Output checkpoint path.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> Path:
    dataset_root = Path(args.data).expanduser().resolve()
    samples_path = dataset_root / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing GRVS rotation samples: {samples_path}")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing backbone checkpoint: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload["model_state_dict"]
    checkpoint_config = dict(payload.get("config", {}))
    cameras = tuple(checkpoint_config.get("cameras", ("left", "center", "right")))
    dataset = SfpRotationDataset(
        dataset_root,
        cameras=cameras,
        transform=_image_transform(checkpoint_config.get("image_size", 224)),
        validate_files=True,
    )
    if args.dry_run:
        print(
            "dry-run rotation: "
            f"samples={len(dataset)}, data={dataset_root}, "
            f"checkpoint={checkpoint_path}, output={args.output}"
        )
        return Path(args.output)

    val_size = max(1, int(len(dataset) * float(args.val_ratio)))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    hidden_dim = _infer_hidden_dim(state_dict)
    num_port_heads = _infer_num_port_heads(state_dict, checkpoint_config)
    model = build_resnet_position_model(
        backbone_name=checkpoint_config.get("backbone", "resnet50"),
        pretrained=False,
        output_dim=3,
        hidden_dim=hidden_dim,
        dropout=float(checkpoint_config.get("dropout", 0.1)),
        aggregation=checkpoint_config.get("aggregation", "mean"),
        num_port_heads=num_port_heads,
        num_views=int(checkpoint_config.get("num_views", len(cameras))),
    )
    model.load_state_dict(state_dict)
    model.to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    best_val = float("inf")
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            image = batch["image"].to(args.device)
            port_id = batch["port_id"].to(args.device)
            target = batch["raw_target"].to(args.device)
            pred = model(image, port_id)
            loss = torch.linalg.vector_norm(pred - target, dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * image.shape[0]
        train_loss /= max(1, len(train_ds))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(args.device)
                port_id = batch["port_id"].to(args.device)
                target = batch["raw_target"].to(args.device)
                pred = model(image, port_id)
                loss = torch.linalg.vector_norm(pred - target, dim=1).mean()
                val_loss += float(loss.item()) * image.shape[0]
        val_loss /= max(1, len(val_ds))

        if val_loss < best_val:
            best_val = val_loss
            save_config = dict(checkpoint_config)
            save_config.update(
                {
                    "cameras": cameras,
                    "output": "plug_to_port_rotation_rotvec_rad",
                    "target_mean": [0.0, 0.0, 0.0],
                    "target_std": [1.0, 1.0, 1.0],
                    "grvs_data": str(dataset_root),
                    "grvs_version": SfpGrvsConfig.VERSION,
                }
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": save_config,
                    "metrics": {
                        "val_rotvec_rad": float(val_loss),
                        "train_rotvec_rad": float(train_loss),
                        "epoch": epoch,
                    },
                },
                output_path,
            )
        print(
            f"epoch={epoch:03d} train_rotvec_rad={train_loss:.4f} "
            f"val_rotvec_rad={val_loss:.4f} best={best_val:.4f}"
        )

    return output_path


def main() -> None:
    args = parse_args()
    output = run_training(args)
    print(f"rotation checkpoint: {output}")


if __name__ == "__main__":
    main()
