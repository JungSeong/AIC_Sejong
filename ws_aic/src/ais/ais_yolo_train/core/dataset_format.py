from pathlib import Path

import cv2
import numpy as np
import yaml


WS_ROOT = Path(__file__).resolve().parents[4]
APPROACH_DATA_ROOT = WS_ROOT / "data" / "yolo" / "approach"

CAMERAS = [
    ("left", "left_camera/optical"),
    ("center", "center_camera/optical"),
    ("right", "right_camera/optical"),
]

SFP_KEYPOINT_NAMES = [
    "port0_top_left",
    "port0_top_right",
    "port0_bottom_right",
    "port0_bottom_left",
    "port1_top_left",
    "port1_top_right",
    "port1_bottom_right",
    "port1_bottom_left",
]

SC_KEYPOINT_NAMES = [
    "sc_top_left",
    "sc_top_right",
    "sc_bottom_right",
    "sc_bottom_left",
]

# Visible SC port opening face corners in the sc_port_link frame. The SC port
# entrance is normal to local +Y; using the old local X-Y plane labels a thin
# side/top face instead of the magenta port opening.
SC_PORT_CORNER_POINTS_M = [
    (-0.0173355, 0.0137160, -0.0046355),
    (0.0173355, 0.0137160, -0.0046355),
    (0.0173355, 0.0137160, 0.0046355),
    (-0.0173355, 0.0137160, 0.0046355),
]

TARGET_CONFIGS = {
    "SFP": {
        "task": "pose",
        "class_id": 0,
        "class_name": "port_pair",
        "keypoint_names": SFP_KEYPOINT_NAMES,
        "port_size_m": (0.014, 0.010),
    },
    "SC": {
        "task": "pose",
        "class_id": 0,
        "class_name": "sc_port",
        "keypoint_names": SC_KEYPOINT_NAMES,
        "corner_points_m": SC_PORT_CORNER_POINTS_M,
    },
}

TARGET_CHOICES = tuple(TARGET_CONFIGS)
DEFAULT_TARGET = "SFP"
DEFAULT_OUTPUT = APPROACH_DATA_ROOT / DEFAULT_TARGET
MOUNT_CANDIDATE_COUNT = 5
SC_PORT_CANDIDATE_COUNT = 2


def normalize_target(target: str) -> str:
    normalized = target.upper()
    if normalized not in TARGET_CONFIGS:
        raise ValueError(f"Unknown target {target!r}. Use one of: {', '.join(TARGET_CHOICES)}")
    return normalized


def get_target_config(target: str) -> dict:
    return TARGET_CONFIGS[normalize_target(target)]


def default_output_for_target(target: str) -> Path:
    return APPROACH_DATA_ROOT / normalize_target(target)


def draw_label(image: np.ndarray, label: str, target: str = DEFAULT_TARGET) -> np.ndarray:
    config = get_target_config(target)
    keypoint_names = config["keypoint_names"]
    annotated = image.copy()
    h, w = annotated.shape[:2]
    values = [float(v) for v in label.split()]
    x_c, y_c, box_w, box_h = values[1:5]
    x1 = int((x_c - box_w / 2) * w)
    y1 = int((y_c - box_h / 2) * h)
    x2 = int((x_c + box_w / 2) * w)
    y2 = int((y_c + box_h / 2) * h)

    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if not keypoint_names:
        cv2.putText(
            annotated,
            config["class_name"],
            (x1 + 4, max(18, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
        return annotated

    kpts = np.array(values[5:], dtype=np.float64).reshape(-1, 2)
    kpts[:, 0] *= w
    kpts[:, 1] *= h
    if len(kpts) == 4:
        groups = [(0, 4, (0, 255, 255))]
    else:
        groups = [(0, 4, (0, 255, 255)), (4, 8, (255, 0, 255))]

    for start, end, color in groups:
        if start >= len(kpts):
            continue
        pts = kpts[start:end].astype(int)
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)
        for idx, (x, y) in enumerate(pts, start=start):
            cv2.circle(annotated, (int(x), int(y)), 4, color, -1)
            cv2.putText(
                annotated,
                str(idx),
                (int(x) + 4, int(y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
    return annotated


def split_for_episode(episode_id: int, val_ratio: float) -> str:
    if val_ratio <= 0:
        return "train"
    period = max(1, round(1.0 / val_ratio))
    return "val" if episode_id % period == 0 else "train"


def write_data_yaml(output_dir: Path, target: str = DEFAULT_TARGET) -> None:
    config = get_target_config(target)
    cfg = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {config["class_id"]: config["class_name"]},
    }
    if config["task"] == "pose":
        keypoint_names = config["keypoint_names"]
        cfg.update(
            {
                "kpt_shape": [len(keypoint_names), 2],
                "flip_idx": list(range(len(keypoint_names))),
                "kpt_names": {config["class_id"]: keypoint_names},
            }
        )

    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
