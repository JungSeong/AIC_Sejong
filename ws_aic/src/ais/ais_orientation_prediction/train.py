from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


sys.path.insert(0, str(_repo_root() / "src" / "ais"))

from ais_orientation_prediction.model import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_WEIGHT_ROOT,
    JOINT_NAMES,
    OrientationDeltaDataset,
    attach_wrench_lpf_features,
    build_orientation_model,
    compute_ee_pose_target_stats,
    compute_input_stats,
    compute_target_stats,
    fit,
    load_samples,
    sample_has_force_torque,
    sample_has_ee_pose_target,
    sample_has_joint_positions,
    sample_has_target,
    seed_everything,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multimodal ais_orientation_prediction model.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--versions", nargs="+", default=["v3.2"])
    parser.add_argument("--cameras", nargs="+", default=["left", "center", "right"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--weight-root", type=Path, default=DEFAULT_WEIGHT_ROOT)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--wrench-lpf-alpha", type=float, default=0.2)
    parser.add_argument("--aux-ee-pose-loss-weight", type=float, default=0.1)
    parser.add_argument("--direction-loss-weight", type=float, default=0.1)
    parser.add_argument("--magnitude-loss-weight", type=float, default=0.1)
    parser.add_argument("--direction-eps", type=float, default=1e-3)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-source", choices=["actual", "collect"], default="actual")
    parser.add_argument("--target-mode", choices=["correction", "applied"], default="correction")
    parser.add_argument("--target-unit", choices=["rad", "deg"], default="rad")
    parser.add_argument(
        "--joint-source",
        choices=["insertion_wrist", "ik_result", "home"],
        default="insertion_wrist",
    )
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _complete_multimodal_samples(
    samples: list[dict],
    *,
    cameras: tuple[str, ...],
    target_source: str,
    target_mode: str,
    joint_source: str,
) -> list[dict]:
    return [
        sample
        for sample in samples
        if all(camera in sample.get("images", {}) for camera in cameras)
        and sample_has_target(
            sample,
            target_source=target_source,
            target_mode=target_mode,
        )
        and sample_has_force_torque(sample)
        and sample_has_joint_positions(
            sample,
            joint_source=joint_source,
            joint_names=JOINT_NAMES,
        )
        and sample_has_ee_pose_target(sample)
    ]


def main() -> None:
    args = _parse_args()
    seed_everything(args.seed)

    cameras = tuple(args.cameras)
    samples = load_samples(
        args.dataset_root,
        versions=args.versions,
        splits=("train", "val"),
    )
    samples = _complete_multimodal_samples(
        samples,
        cameras=cameras,
        target_source=args.target_source,
        target_mode=args.target_mode,
        joint_source=args.joint_source,
    )
    samples = attach_wrench_lpf_features(samples, alpha=args.wrench_lpf_alpha)
    train_samples = [sample for sample in samples if sample.get("_split") == "train"]
    val_samples = [sample for sample in samples if sample.get("_split") == "val"]
    if not train_samples or not val_samples:
        raise RuntimeError(
            f"Need non-empty train and val samples, got train={len(train_samples)} "
            f"val={len(val_samples)}."
        )

    target_stats = compute_target_stats(
        train_samples,
        target_source=args.target_source,
        target_mode=args.target_mode,
        target_unit=args.target_unit,
    )
    input_stats = compute_input_stats(
        train_samples,
        joint_source=args.joint_source,
        joint_names=JOINT_NAMES,
    )
    ee_pose_target_stats = compute_ee_pose_target_stats(train_samples)

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    dataset_kwargs = {
        "dataset_root": args.dataset_root,
        "cameras": cameras,
        "target_source": args.target_source,
        "target_mode": args.target_mode,
        "target_unit": args.target_unit,
        "target_mean": target_stats["mean"],
        "target_std": target_stats["std"],
        "force_torque_mean": input_stats["force_torque"]["mean"],
        "force_torque_std": input_stats["force_torque"]["std"],
        "joint_mean": input_stats["joint"]["mean"],
        "joint_std": input_stats["joint"]["std"],
        "ee_pose_target_mean": ee_pose_target_stats["mean"],
        "ee_pose_target_std": ee_pose_target_stats["std"],
        "joint_source": args.joint_source,
        "joint_names": JOINT_NAMES,
        "always_return_views": True,
        "require_all_cameras": True,
    }
    train_dataset = OrientationDeltaDataset(
        samples=train_samples,
        transform=train_transform,
        **dataset_kwargs,
    )
    val_dataset = OrientationDeltaDataset(
        samples=val_samples,
        transform=val_transform,
        **dataset_kwargs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = build_orientation_model(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        aggregation="concat" if len(cameras) > 1 else "mean",
        num_views=len(cameras),
        force_torque_dim=18,
        joint_dim=len(JOINT_NAMES) * 2,
        aux_ee_pose_dim=9,
        freeze_backbone=args.freeze_backbone,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    version_tag = "_".join(args.versions)
    camera_tag = "_".join(cameras)
    run_name = (
        args.run_name
        or f"rpy_delta_multimodal_{args.backbone}_{camera_tag}_{version_tag}"
    )

    config = {
        "architecture": "image_resnet + force_torque(raw,lpf,raw_minus_lpf)_mlp + joint_sin_cos_mlp -> concat -> mlp",
        "backbone": args.backbone,
        "pretrained": args.pretrained,
        "versions": args.versions,
        "cameras": list(cameras),
        "target_source": args.target_source,
        "target_mode": args.target_mode,
        "target_unit": args.target_unit,
        "joint_source": args.joint_source,
        "joint_names": list(JOINT_NAMES),
        "joint_encoding": "sin_cos",
        "main_head": "pred_rot = softplus(pred_magnitude) * normalize(pred_direction)",
        "aux_ee_pose_loss_weight": args.aux_ee_pose_loss_weight,
        "direction_loss_weight": args.direction_loss_weight,
        "magnitude_loss_weight": args.magnitude_loss_weight,
        "direction_eps": args.direction_eps,
        "ema_decay": args.ema_decay,
        "wrench_lpf_alpha": args.wrench_lpf_alpha,
        "wrench_delta": "raw_minus_lpf",
        "force_torque_dim": 18,
        "joint_dim": len(JOINT_NAMES) * 2,
        "target_mean": target_stats["mean"].tolist(),
        "target_std": target_stats["std"].tolist(),
        "force_torque_mean": input_stats["force_torque"]["mean"].tolist(),
        "force_torque_std": input_stats["force_torque"]["std"].tolist(),
        "joint_mean": input_stats["joint"]["mean"].tolist(),
        "joint_std": input_stats["joint"]["std"].tolist(),
        "ee_pose_target": "actual.plug_reference xyz+rotation_6d, auxiliary loss only",
        "ee_pose_target_mean": ee_pose_target_stats["mean"].tolist(),
        "ee_pose_target_std": ee_pose_target_stats["std"].tolist(),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }

    print(f"dataset_root: {args.dataset_root}")
    print(f"samples: train={len(train_dataset)} val={len(val_dataset)}")
    print(f"device: {device}")
    print(f"run_name: {run_name}")
    print(f"force_torque_dim: 18 = raw(6) + lpf(6) + raw_minus_lpf(6)")
    print(f"joint_dim: {len(JOINT_NAMES) * 2} = sin(6) + cos(6)")
    print(f"ema_decay: {args.ema_decay}")

    history = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs=args.epochs,
        weight_dir=args.weight_root,
        run_name=run_name,
        target_mean=target_stats["mean"],
        target_std=target_stats["std"],
        aux_loss_weight=args.aux_ee_pose_loss_weight,
        direction_loss_weight=args.direction_loss_weight,
        magnitude_loss_weight=args.magnitude_loss_weight,
        direction_eps=args.direction_eps,
        ema_decay=args.ema_decay,
        config=config,
        show_progress=not args.no_progress,
    )

    output_dir = args.weight_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


if __name__ == "__main__":
    main()
