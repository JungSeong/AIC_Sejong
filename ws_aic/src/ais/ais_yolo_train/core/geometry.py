import numpy as np


def euler_to_quat(roll: float, pitch: float, yaw: float):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return w, x, y, z


def make_viewpoints(n: int, board_center=(-0.38, 0.22, 0.13)):
    bx, by, bz = board_center
    viewpoints = [
        (bx, by, bz + 0.40, np.pi, 0.0, 0.0),
        (bx, by, bz + 0.55, np.pi, 0.0, 0.0),
        (bx, by, bz + 0.30, np.pi, 0.0, 0.0),
    ]

    for dx, dy, roll_off, pitch_off in [
        (0.10, 0.00, 0.0, 0.3),
        (-0.10, 0.00, 0.0, -0.3),
        (0.00, 0.10, 0.3, 0.0),
        (0.00, -0.10, -0.3, 0.0),
    ]:
        viewpoints.append((bx + dx, by + dy, bz + 0.40, np.pi + roll_off, pitch_off, 0.0))

    for dx, dy in [(0.08, 0.08), (0.08, -0.08), (-0.08, 0.08), (-0.08, -0.08)]:
        viewpoints.append((bx + dx, by + dy, bz + 0.40, np.pi, 0.0, np.arctan2(dy, dx)))

    rng = np.random.default_rng(42)
    while len(viewpoints) < n:
        dx = rng.uniform(-0.12, 0.12)
        dy = rng.uniform(-0.12, 0.12)
        dz = rng.uniform(0.30, 0.55)
        roll = np.pi + rng.uniform(-0.3, 0.3)
        pitch = rng.uniform(-0.3, 0.3)
        yaw = rng.uniform(-0.5, 0.5)
        viewpoints.append((bx + dx, by + dy, bz + dz, roll, pitch, yaw))

    return viewpoints[:n]


def project_to_camera(point_3d_base: np.ndarray, k: np.ndarray, base_to_cam: np.ndarray):
    point_cam = base_to_cam @ np.append(point_3d_base, 1.0)
    x, y, z = point_cam[:3]
    if z < 1e-6:
        return None
    u = k[0, 0] * x / z + k[0, 2]
    v = k[1, 1] * y / z + k[1, 2]
    return float(u), float(v), float(z)


def port_corners_in_frame(port_size_m: tuple[float, float]) -> np.ndarray:
    width, height = port_size_m
    half_w = width / 2.0
    half_h = height / 2.0
    return np.array(
        [
            [-half_w, -half_h, 0.0],
            [half_w, -half_h, 0.0],
            [half_w, half_h, 0.0],
            [-half_w, half_h, 0.0],
        ],
        dtype=np.float64,
    )


def order_image_corners(points: np.ndarray) -> np.ndarray:
    sums = points[:, 0] + points[:, 1]
    diffs = points[:, 0] - points[:, 1]
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmax(diffs)],
            points[np.argmax(sums)],
            points[np.argmin(diffs)],
        ],
        dtype=np.float64,
    )


def make_bbox_from_points(points: np.ndarray, image_w: int, image_h: int, margin: float):
    x_min = float(np.min(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    x_max = float(np.max(points[:, 0]))
    y_max = float(np.max(points[:, 1]))
    pad_x = (x_max - x_min) * margin
    pad_y = (y_max - y_min) * margin

    x_min = np.clip(x_min - pad_x, 0, image_w - 1)
    y_min = np.clip(y_min - pad_y, 0, image_h - 1)
    x_max = np.clip(x_max + pad_x, 0, image_w - 1)
    y_max = np.clip(y_max + pad_y, 0, image_h - 1)
    if x_max <= x_min or y_max <= y_min:
        return None

    x_center = ((x_min + x_max) / 2.0) / image_w
    y_center = ((y_min + y_max) / 2.0) / image_h
    width = (x_max - x_min) / image_w
    height = (y_max - y_min) / image_h
    return x_center, y_center, width, height
