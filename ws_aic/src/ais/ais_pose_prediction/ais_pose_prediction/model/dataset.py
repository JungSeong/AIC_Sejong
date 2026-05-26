from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


CAMERAS = ("left", "center", "right")
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def _ws_aic_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    return Path(__file__).resolve().parents[5]


DEFAULT_DATASET_ROOT = _ws_aic_root() / "data" / "ais_rpy_randomization"


def _episode_name(sample_id: str) -> str:
    match = re.match(r"(.+)_collect_\d+$", sample_id)
    return match.group(1) if match else sample_id


def load_samples(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    versions: Sequence[str] | None = None,
    splits: Sequence[str] = ("train", "val"),
) -> list[dict[str, Any]]:
    root = Path(dataset_root).expanduser().resolve()
    version_roots = [root / version for version in versions] if versions else [
        path for path in sorted(root.iterdir()) if path.is_dir() and (path / "metadata").is_dir()
    ]
    grouped: dict[str, dict[str, Any]] = {}
    for version_root in version_roots:
        for split in splits:
            for path in sorted((version_root / "metadata" / split).glob("*.json")):
                record = json.loads(path.read_text(encoding="utf-8"))
                camera = str(record.get("camera", ""))
                if camera not in CAMERAS:
                    continue
                sample_id = str(record["sample_id"])
                sample = grouped.get(sample_id)
                if sample is None:
                    sample = dict(record)
                    sample["_dataset_root"] = str(version_root)
                    sample["_version"] = version_root.name
                    sample["_split"] = split
                    sample["episode_name"] = _episode_name(sample_id)
                    sample["images"] = {}
                    grouped[sample_id] = sample
                sample["images"][camera] = str(record["image"])
    return list(grouped.values())


def port_id_from_sample(sample: Mapping[str, Any]) -> int:
    scenario = sample.get("scenario", {})
    if "sfp_port_idx" in scenario:
        return int(scenario["sfp_port_idx"])
    text = str(sample.get("task", {}).get("port_name", ""))
    match = re.search(r"_(\d+)$", text)
    return int(match.group(1)) if match else 0


def _rpy_from_quat_xyzw(record: Mapping[str, Any]) -> torch.Tensor:
    qx, qy, qz, qw = (float(record[k]) for k in ("qx", "qy", "qz", "qw"))
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        return torch.zeros(3, dtype=torch.float32)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    matrix = torch.tensor(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=torch.float32,
    )
    sy = torch.linalg.vector_norm(matrix[:2, 0]).clamp_min(1e-9)
    return torch.stack(
        [
            torch.atan2(matrix[2, 1], matrix[2, 2]),
            torch.atan2(-matrix[2, 0], sy),
            torch.atan2(matrix[1, 0], matrix[0, 0]),
        ]
    )


def pose_targets_from_sample(sample: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    ports = sample["label"]["ports"]
    task = sample["task"]
    prefix = str(task["target_module_name"])
    values = {}
    dyaw = None
    for port_id in (0, 1):
        port_key = f"{prefix}/sfp_port_{port_id}"
        port_record = ports[port_key]
        offset = port_record["plug_tip_in_port"]
        values[f"port{port_id}_position"] = torch.tensor(
            [float(offset["x_mm"]), float(offset["y_mm"])],
            dtype=torch.float32,
        )
        if bool(port_record.get("is_target", False)):
            dyaw = _rpy_from_quat_xyzw(port_record["port_in_plug_tip"])[2]
    values["dyaw"] = torch.tensor(0.0 if dyaw is None else float(dyaw), dtype=torch.float32)
    return values


def force_torque_from_sample(sample: Mapping[str, Any]) -> torch.Tensor:
    wrench = sample["wrench"]
    force = wrench["force"]
    torque = wrench["torque"]
    return torch.tensor(
        [force["x"], force["y"], force["z"], torque["x"], torque["y"], torque["z"]],
        dtype=torch.float32,
    )


def joint_positions_from_sample(sample: Mapping[str, Any]) -> torch.Tensor:
    record = sample["label"]["insertion_wrist"]["joint_positions"]
    angles = torch.tensor([float(record[name]) for name in JOINT_NAMES], dtype=torch.float32)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=0)


class PosePredictionDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        *,
        cameras: Sequence[str] = CAMERAS,
        image_size: int | tuple[int, int] = 224,
        target_mean: Mapping[str, Sequence[float] | float] | None = None,
        target_std: Mapping[str, Sequence[float] | float] | None = None,
    ) -> None:
        self.samples = [dict(sample) for sample in samples if all(c in sample.get("images", {}) for c in cameras)]
        self.cameras = tuple(cameras)
        self.image_size = image_size
        self.target_mean = target_mean or {}
        self.target_std = target_std or {}

    def __len__(self) -> int:
        return len(self.samples)

    def _image(self, sample: Mapping[str, Any], camera: str) -> torch.Tensor:
        path = Path(sample["_dataset_root"]) / str(sample["images"][camera])
        image = Image.open(path).convert("RGB")
        size = (self.image_size, self.image_size) if isinstance(self.image_size, int) else self.image_size
        image = TF.resize(image, list(size))
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def _norm(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if name not in self.target_mean or name not in self.target_std:
            return value
        mean = torch.as_tensor(self.target_mean[name], dtype=torch.float32)
        std = torch.as_tensor(self.target_std[name], dtype=torch.float32).clamp_min(1e-6)
        return (value - mean) / std

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        targets = pose_targets_from_sample(sample)
        return {
            "image": torch.stack([self._image(sample, camera) for camera in self.cameras], dim=0),
            "force_torque": force_torque_from_sample(sample),
            "joint_positions": joint_positions_from_sample(sample),
            "port_id": torch.tensor(port_id_from_sample(sample), dtype=torch.long),
            "port0_position": self._norm("port0_position", targets["port0_position"]),
            "port1_position": self._norm("port1_position", targets["port1_position"]),
            "dyaw": self._norm("dyaw", targets["dyaw"]),
            "raw_port0_position": targets["port0_position"],
            "raw_port1_position": targets["port1_position"],
            "raw_dyaw": targets["dyaw"],
        }
