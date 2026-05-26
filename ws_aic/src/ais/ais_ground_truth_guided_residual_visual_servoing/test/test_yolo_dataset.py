from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from ais_ground_truth_guided_residual_visual_servoing.data.yolo_dataset import (
    SfpYoloFailureRecorder,
)


def _image_msg(width: int = 100, height: int = 100):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    return SimpleNamespace(
        width=width,
        height=height,
        encoding="bgr8",
        step=width * 3,
        data=image.tobytes(),
    )


def _camera_info(width: int = 100, height: int = 100):
    return SimpleNamespace(k=[100.0, 0.0, width / 2, 0.0, 100.0, height / 2, 0.0, 0.0, 1.0])


def _transform(x: float, y: float, z: float):
    return SimpleNamespace(
        translation=SimpleNamespace(x=x, y=y, z=z),
        rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def test_yolo_failure_recorder_writes_sfp_pose_dataset(tmp_path):
    obs = SimpleNamespace()
    for camera in ("left", "center", "right"):
        setattr(obs, f"{camera}_image", _image_msg())
        setattr(obs, f"{camera}_camera_info", _camera_info())

    recorder = SfpYoloFailureRecorder(tmp_path, val_ratio=0.0, bbox_margin=0.08)
    saved = recorder.record(
        observation=obs,
        port_transforms=[_transform(-0.02, 0.0, 1.0), _transform(0.02, 0.0, 1.0)],
        base_to_camera={
            "left": np.eye(4),
            "center": np.eye(4),
            "right": np.eye(4),
        },
        episode_id=1,
        timestep_index=7,
        sample_id="sample_000001",
        stem_prefix="yolo_failed",
    )

    assert saved == 3
    assert (tmp_path / "data.yaml").exists()
    labels = sorted((tmp_path / "labels" / "train").glob("*.txt"))
    images = sorted((tmp_path / "images" / "train").glob("*.jpg"))
    assert len(labels) == 3
    assert len(images) == 3
    assert all(path.name.startswith("sample_000001_") for path in labels + images)
    records = [
        json.loads(line)
        for line in (tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 3
    assert {record["camera"] for record in records} == {"left", "center", "right"}
    assert {record["reason"] for record in records} == {"yolo_failed"}
    assert {record["timestep_index"] for record in records} == {7}
    first_values = labels[0].read_text(encoding="utf-8").strip().split()
    assert first_values[0] == "0"
    assert len(first_values) == 21
