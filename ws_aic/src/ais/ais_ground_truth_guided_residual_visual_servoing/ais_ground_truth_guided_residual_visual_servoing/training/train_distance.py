from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from ..core.config import SRC_ROOT, SfpGrvsConfig, WS_ROOT


AIS_ROOT = SRC_ROOT / "ais"
if str(AIS_ROOT) not in sys.path:
    sys.path.append(str(AIS_ROOT))

from ais_distance_prediction.model.dataset import VisionOffsetDataset  # noqa: E402
from ais_distance_prediction.model.position_model import (  # noqa: E402
    build_resnet_position_model,
)


DEFAULT_BACKBONE_CHECKPOINT = (
    WS_ROOT
    / "model"
    / "ais_distance_prediction"
    / "sfp_distance_resnet50_left_center_right_concat"
    / "best.pt"
)


def _infer_hidden_dim(state_dict: dict[str, torch.Tensor]) -> int:
    if "head.0.weight" in state_dict:
        return int(state_dict["head.0.weight"].shape[0])
    indices = []
    for key in state_dict:
        if key.startswith("heads.") and key.endswith(".0.weight"):
            try:
                indices.append(int(key.split(".")[1]))
            except (IndexError, ValueError):
                continue
    if not indices:
        raise ValueError("Checkpoint has no recognizable regression head weights.")
    return int(state_dict[f"heads.{min(indices)}.0.weight"].shape[0])


def _infer_num_port_heads(
    state_dict: dict[str, torch.Tensor],
    checkpoint_config: dict,
) -> int:
    if "head.0.weight" in state_dict:
        return int(checkpoint_config.get("num_port_heads", 1))
    indices = []
    for key in state_dict:
        if key.startswith("heads.") and key.endswith(".0.weight"):
            try:
                indices.append(int(key.split(".")[1]))
            except (IndexError, ValueError):
                continue
    return int(checkpoint_config.get("num_port_heads", max(indices) + 1))


def _image_transform(image_size: int | list[int] | tuple[int, int] | None):
    steps = []
    if image_size is not None:
        if isinstance(image_size, int):
            size = (image_size, image_size)
        else:
            size = (int(image_size[0]), int(image_size[1]))
        steps.append(transforms.Resize(size))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the SFP distance prediction backbone on GRVS samples."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=SfpGrvsConfig.DISTANCE_DATASET_DIR,
        help="GRVS distance dataset root containing samples.jsonl.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("AIC_DISTANCE_MODEL_PATH", DEFAULT_BACKBONE_CHECKPOINT)),
        help="Existing distance prediction checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SfpGrvsConfig.DISTANCE_MODEL_DIR / "best.pt",
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
        raise FileNotFoundError(f"Missing GRVS distance samples: {samples_path}")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing distance checkpoint: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload["model_state_dict"]
    checkpoint_config = dict(payload.get("config", {}))
    target_mean = torch.as_tensor(
        checkpoint_config.get("target_mean", [0.0, 0.0, 0.0]),
        dtype=torch.float32,
        device=args.device,
    )
    target_std = torch.as_tensor(
        checkpoint_config.get("target_std", [1.0, 1.0, 1.0]),
        dtype=torch.float32,
        device=args.device,
    )

    cameras = tuple(checkpoint_config.get("cameras", ("left", "center", "right")))
    dataset = VisionOffsetDataset(
        dataset_root,
        cameras=cameras,
        target_keys=("x_mm", "y_mm", "z_mm"),
        transform=_image_transform(checkpoint_config.get("image_size", 224)),
        expand_all_ports=False,
        validate_files=True,
    )
    if args.dry_run:
        print(
            "dry-run: "
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
        output_dim=int(target_mean.numel()),
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
            target_mm = batch["raw_target"].to(args.device)
            pred_norm = model(image, port_id)
            pred_mm = pred_norm * target_std + target_mean
            loss = torch.linalg.vector_norm(pred_mm - target_mm, dim=1).mean()
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
                target_mm = batch["raw_target"].to(args.device)
                pred_norm = model(image, port_id)
                pred_mm = pred_norm * target_std + target_mean
                loss = torch.linalg.vector_norm(pred_mm - target_mm, dim=1).mean()
                val_loss += float(loss.item()) * image.shape[0]
        val_loss /= max(1, len(val_ds))

        if val_loss < best_val:
            best_val = val_loss
            save_config = dict(checkpoint_config)
            save_config.update(
                {
                    "cameras": cameras,
                    "target_mean": target_mean.detach().cpu().tolist(),
                    "target_std": target_std.detach().cpu().tolist(),
                    "grvs_data": str(dataset_root),
                    "grvs_version": SfpGrvsConfig.VERSION,
                }
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": save_config,
                    "metrics": {
                        "val_euclidean_mm": float(val_loss),
                        "train_euclidean_mm": float(train_loss),
                        "epoch": epoch,
                    },
                },
                output_path,
            )
        print(
            f"epoch={epoch:03d} train_euclidean_mm={train_loss:.4f} "
            f"val_euclidean_mm={val_loss:.4f} best={best_val:.4f}"
        )

    return output_path


def main() -> None:
    args = parse_args()
    output = run_training(args)
    print(f"distance checkpoint: {output}")


if __name__ == "__main__":
    main()
