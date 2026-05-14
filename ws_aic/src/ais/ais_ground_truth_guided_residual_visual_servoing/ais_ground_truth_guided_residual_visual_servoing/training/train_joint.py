from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from ..core.config import SfpGrvsConfig
from .train_distance import (
    DEFAULT_BACKBONE_CHECKPOINT,
    run_training as run_distance_training,
)
from .train_rotation import run_training as run_rotation_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GRVS YOLO pose, distance, and rotation models."
    )
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-distance", action="store_true")
    parser.add_argument("--skip-rotation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yolo-base-model", default="yolo11n-pose.pt")
    parser.add_argument("--yolo-epochs", type=int, default=50)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--distance-epochs", type=int, default=20)
    parser.add_argument("--distance-batch-size", type=int, default=64)
    parser.add_argument("--distance-lr", type=float, default=1e-5)
    parser.add_argument("--rotation-epochs", type=int, default=20)
    parser.add_argument("--rotation-batch-size", type=int, default=64)
    parser.add_argument("--rotation-lr", type=float, default=1e-5)
    return parser.parse_args()


def train_yolo(args: argparse.Namespace) -> None:
    data_yaml = SfpGrvsConfig.YOLO_DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing GRVS YOLO dataset: {data_yaml}")
    if args.dry_run:
        print(
            "dry-run yolo: "
            f"data={data_yaml}, output_dir={SfpGrvsConfig.YOLO_MODEL_DIR}, "
            f"base={args.yolo_base_model}"
        )
        return

    from ultralytics import YOLO

    SfpGrvsConfig.YOLO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.yolo_base_model)
    model.train(
        data=str(data_yaml),
        epochs=int(args.yolo_epochs),
        imgsz=int(args.yolo_imgsz),
        project=str(SfpGrvsConfig.YOLO_MODEL_DIR.parent),
        name=SfpGrvsConfig.YOLO_MODEL_DIR.name,
        exist_ok=True,
        task="pose",
    )


def train_distance(args: argparse.Namespace) -> None:
    distance_args = argparse.Namespace(
        data=SfpGrvsConfig.DISTANCE_DATASET_DIR,
        checkpoint=Path(
            os.environ.get(
                "AIC_DISTANCE_MODEL_PATH",
                str(DEFAULT_BACKBONE_CHECKPOINT),
            )
        ),
        output=SfpGrvsConfig.DISTANCE_MODEL_DIR / "best.pt",
        epochs=int(args.distance_epochs),
        batch_size=int(args.distance_batch_size),
        lr=float(args.distance_lr),
        weight_decay=1e-4,
        val_ratio=0.15,
        num_workers=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dry_run=bool(args.dry_run),
    )
    run_distance_training(distance_args)


def train_rotation(args: argparse.Namespace) -> None:
    rotation_args = argparse.Namespace(
        data=SfpGrvsConfig.ROTATION_DATASET_DIR,
        checkpoint=Path(
            os.environ.get(
                "AIC_DISTANCE_MODEL_PATH",
                str(DEFAULT_BACKBONE_CHECKPOINT),
            )
        ),
        output=SfpGrvsConfig.ROTATION_MODEL_DIR / "best.pt",
        epochs=int(args.rotation_epochs),
        batch_size=int(args.rotation_batch_size),
        lr=float(args.rotation_lr),
        weight_decay=1e-4,
        val_ratio=0.15,
        num_workers=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dry_run=bool(args.dry_run),
    )
    run_rotation_training(rotation_args)


def main() -> None:
    args = parse_args()
    print(
        "GRVS paths: "
        f"data_yolo={SfpGrvsConfig.YOLO_DATASET_DIR}, "
        f"data_distance={SfpGrvsConfig.DISTANCE_DATASET_DIR}, "
        f"data_rotation={SfpGrvsConfig.ROTATION_DATASET_DIR}, "
        f"model_yolo={SfpGrvsConfig.YOLO_MODEL_DIR}, "
        f"model_distance={SfpGrvsConfig.DISTANCE_MODEL_DIR}, "
        f"model_rotation={SfpGrvsConfig.ROTATION_MODEL_DIR}"
    )
    if not args.skip_yolo:
        train_yolo(args)
    if not args.skip_distance:
        train_distance(args)
    if not args.skip_rotation:
        train_rotation(args)


if __name__ == "__main__":
    main()
