"""
crop_insertion_roi.py
─────────────────────
URDF calibration 정보를 이용해 plug/port 3D 위치를
각 카메라 픽셀 좌표로 투영하고, 삽입 ROI를 크롭한다.

Kinematic chain (URDF ur_gz.urdf.xacro):
  world
   └─ tool0           ← transforms.gripper (steps.jsonl)
       └─ cam_mount   ← T_tool0_to_mount: xyz=(0,0,-0.0265)
           ├─ center  ← T_mount_to_cam[center]
           ├─ left    ← T_mount_to_cam[left]
           └─ right   ← T_mount_to_cam[right]
               └─ sensor_link ← T_cam_to_sensor
                   └─ optical ← T_sensor_to_optical

Intrinsics (basler_camera_macro.xacro):
  width=1152, height=1024, hfov=0.8718 rad
  fx = fy ≈ 1237.6, cx=576, cy=512, 왜곡 없음
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


# ── Intrinsics ────────────────────────────────────────────────────────────────
IMG_W, IMG_H = 1152, 1024
HFOV = 0.8718  # radians

FX = FY = IMG_W / (2.0 * math.tan(HFOV / 2.0))   # ≈ 1237.6
CX, CY = IMG_W / 2.0, IMG_H / 2.0                 # 576, 512

K = np.array([
    [FX,  0.0, CX],
    [0.0, FY,  CY],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

# ── Transform helpers ─────────────────────────────────────────────────────────

def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX 오일러 → 3×3 회전행렬 (URDF 규칙: rpy = roll pitch yaw)."""
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)

    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def make_T(xyz: tuple[float, float, float],
           rpy: tuple[float, float, float]) -> np.ndarray:
    """4×4 homogeneous transform from xyz + rpy."""
    T = np.eye(4)
    T[:3, :3] = rpy_to_rot(*rpy)
    T[:3, 3]  = xyz
    return T


def quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Unit quaternion (w,x,y,z) → 3×3 rotation matrix."""
    T = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    return T


def pose_to_T(translation: dict, rotation: dict) -> np.ndarray:
    """steps.jsonl transform dict → 4×4 homogeneous matrix."""
    t = np.array([translation['x'], translation['y'], translation['z']])
    R = quat_to_rot(rotation['w'], rotation['x'], rotation['y'], rotation['z'])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T


# ── Fixed transforms from URDF ────────────────────────────────────────────────

# tool0 → cam_mount
T_tool0_to_mount = make_T(
    xyz=(0.0, 0.0, -0.0265),
    rpy=(0.0, 0.0, 0.0),
)

# cam_mount → camera_link (per view)
T_mount_to_camera_link = {
    'center': make_T(
        xyz=(0.0, -0.1077, -0.00719),
        rpy=(0.0, -1.30899630, 1.57079623),
    ),
    'left': make_T(
        xyz=(-0.09326, -0.053843, -0.007188),
        rpy=(0.0, -1.30899630, 0.52359903),
    ),
    'right': make_T(
        xyz=(0.09326, -0.053843, -0.007188),
        rpy=(0.0, -1.30899630, 2.61799343),
    ),
}

# camera_link → sensor_link
T_camera_link_to_sensor = make_T(
    xyz=(0.02174, 0.0, 0.0145),
    rpy=(0.0, 0.0, 0.0),
)

# sensor_link → optical (Z-forward 카메라 프레임)
T_sensor_to_optical = make_T(
    xyz=(0.0, 0.0, 0.0),
    rpy=(-math.pi / 2, 0.0, -math.pi / 2),
)

# cam_mount → optical (precomputed fixed chain)
T_mount_to_optical: dict[str, np.ndarray] = {
    view: T_mount_to_camera_link[view]
          @ T_camera_link_to_sensor
          @ T_sensor_to_optical
    for view in ('center', 'left', 'right')
}

# tool0 → optical
T_tool0_to_optical: dict[str, np.ndarray] = {
    view: T_tool0_to_mount @ T_mount_to_optical[view]
    for view in ('center', 'left', 'right')
}


# ── Projection ────────────────────────────────────────────────────────────────

def world_to_pixel(
    point_world: np.ndarray,   # (3,) or (N,3) in world frame
    T_world_to_gripper: np.ndarray,  # 4×4, from steps.jsonl transforms.gripper
    view: str,
) -> np.ndarray:
    """
    3D world 좌표 → (u, v) 픽셀 좌표.

    NOTE: transforms.gripper ≈ tool0 가정.
    실제 gripper 프레임이 다르다면 T_gripper_to_tool0를 추가로 곱할 것.

    Returns
    -------
    uv : (2,) or (N,2) float  (u=col, v=row)
    """
    T_world_to_tool0 = T_world_to_gripper  # gripper ≈ tool0 가정

    # world → optical
    T_optical_to_world = T_world_to_tool0 @ T_tool0_to_optical[view]
    T_world_to_optical = np.linalg.inv(T_optical_to_world)

    pts = np.atleast_2d(point_world)  # (N, 3)
    # homogeneous
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])    # (N, 4)

    pts_cam = (T_world_to_optical @ pts_h.T).T[:, :3]  # (N, 3)

    # 카메라 뒤에 있으면 NaN
    Z = pts_cam[:, 2]
    valid = Z > 0.01

    uv = np.full((pts.shape[0], 2), np.nan)
    uv[valid, 0] = FX * pts_cam[valid, 0] / Z[valid] + CX   # u (col)
    uv[valid, 1] = FY * pts_cam[valid, 1] / Z[valid] + CY   # v (row)

    return uv.squeeze() if point_world.ndim == 1 else uv


# ── Crop helpers ──────────────────────────────────────────────────────────────

class CropResult(NamedTuple):
    crop: np.ndarray          # 크롭된 이미지 (H_crop × W_crop × 3)
    u: float                  # 원본 이미지에서 투영된 u (col)
    v: float                  # 원본 이미지에서 투영된 v (row)
    depth: float              # 카메라에서 포인트까지 Z 거리 (m)
    valid: bool               # 유효한 투영 여부


def crop_roi(
    img: np.ndarray,
    point_world: np.ndarray,
    T_world_to_gripper: np.ndarray,
    view: str,
    half_size_px: int = 160,
    pad_value: int = 0,
) -> CropResult:
    """
    주어진 3D world 포인트를 투영해 이미지를 크롭한다.

    Parameters
    ----------
    img          : (H, W, 3) uint8 RGB 이미지
    point_world  : (3,) 월드 좌표 (plug 또는 port 위치)
    T_world_to_gripper : 4×4 (steps.jsonl transforms.gripper → pose_to_T)
    view         : 'center' | 'left' | 'right'
    half_size_px : 크롭 반경 (픽셀). 전체 크롭 크기 = 2*half_size_px × 2*half_size_px
    pad_value    : 이미지 경계 밖을 채울 값

    Returns
    -------
    CropResult
    """
    uv = world_to_pixel(point_world, T_world_to_gripper, view)
    u, v = float(uv[0]), float(uv[1])

    if np.isnan(u) or np.isnan(v):
        blank = np.full((2 * half_size_px, 2 * half_size_px, 3),
                        pad_value, dtype=np.uint8)
        return CropResult(blank, u, v, float('nan'), False)

    # 카메라 Z (depth)
    T_world_to_tool0 = T_world_to_gripper
    T_optical_to_world = T_world_to_tool0 @ T_tool0_to_optical[view]
    T_world_to_optical = np.linalg.inv(T_optical_to_world)
    pt_h = np.append(point_world, 1.0)
    depth = float((T_world_to_optical @ pt_h)[2])

    # 적응형 크롭: 멀수록 크롭 영역을 줄여 물리 크기 일정하게 유지
    # half_size_px를 depth에 따라 정규화하고 싶다면 아래 주석 해제
    # half_size_px = max(32, int(half_size_px * 0.30 / max(depth, 0.01)))

    h, w = img.shape[:2]
    u_int, v_int = int(round(u)), int(round(v))

    # 패딩 후 크롭
    pad = half_size_px
    padded = cv2.copyMakeBorder(
        img, pad, pad, pad, pad,
        borderType=cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )
    u_p, v_p = u_int + pad, v_int + pad
    crop = padded[v_p - pad: v_p + pad, u_p - pad: u_p + pad]

    valid = (0 <= u_int < w) and (0 <= v_int < h) and depth > 0.01
    return CropResult(crop, u, v, depth, valid)


def adaptive_half_size(depth: float,
                       physical_radius_m: float = 0.025,
                       min_px: int = 64,
                       max_px: int = 256) -> int:
    """
    물리적 반경 physical_radius_m (m)에 해당하는 픽셀 수를 depth로 계산.
    삽입 구멍 반경 ≈ 2.5cm를 기본값으로 사용.
    """
    if depth <= 0:
        return max_px
    px = int(FX * physical_radius_m / depth)
    return int(np.clip(px, min_px, max_px))


# ── Batch crop pipeline ───────────────────────────────────────────────────────

def extract_insertion_crops(
    session_dir: Path,
    view: str = 'center',
    target: str = 'plug',    # 'plug' or 'port'
    phase_filter: list[str] | None = None,
    step_stride: int = 10,
    half_size_px: int | None = None,  # None → adaptive
    physical_radius_m: float = 0.025,
) -> list[dict]:
    """
    세션 전체에서 크롭 이미지를 추출한다.

    Returns
    -------
    list of dict:
        {
          'session': str,
          'step': int,
          'phase': str,
          'view': str,
          'crop': np.ndarray,   # (crop_H, crop_W, 3) uint8 RGB
          'u': float, 'v': float, 'depth': float, 'valid': bool
        }
    """
    if phase_filter is None:
        phase_filter = ['approach', 'insert']

    results = []

    with open(session_dir / 'steps.jsonl') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if i % step_stride != 0:
            continue
        d = json.loads(line)
        if d['phase'] not in phase_filter:
            continue

        # gripper transform
        T_gripper = pose_to_T(
            d['transforms']['gripper']['translation'],
            d['transforms']['gripper']['rotation'],
        )

        # target 3D position
        tf = d['transforms'].get(target, {})
        if not tf:
            continue
        pt_world = np.array([
            tf['translation']['x'],
            tf['translation']['y'],
            tf['translation']['z'],
        ])

        # 이미지 로드
        img_key = f'{view}_image'
        img_path = d.get('observation', {}).get(img_key, {}).get('path')
        if not img_path:
            continue
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 크롭 크기 결정
        if half_size_px is not None:
            hpx = half_size_px
        else:
            # 먼저 depth 계산
            T_w2tool0 = T_gripper
            T_optical_to_world = T_w2tool0 @ T_tool0_to_optical[view]
            T_world_to_optical = np.linalg.inv(T_optical_to_world)
            pt_h = np.append(pt_world, 1.0)
            depth_est = float((T_world_to_optical @ pt_h)[2])
            hpx = adaptive_half_size(depth_est, physical_radius_m)

        result = crop_roi(rgb, pt_world, T_gripper, view, half_size_px=hpx)
        results.append({
            'session': session_dir.name,
            'step': d['step'],
            'phase': d['phase'],
            'view': view,
            **result._asdict(),
        })

    return results


# ── Visualization helpers ─────────────────────────────────────────────────────

def draw_projection_on_image(
    img: np.ndarray,
    point_world: np.ndarray,
    T_world_to_gripper: np.ndarray,
    view: str,
    color: tuple[int, int, int] = (0, 255, 0),
    radius: int = 10,
    label: str = '',
) -> np.ndarray:
    """원본 이미지에 투영점 + 크롭 박스를 그려 반환한다."""
    uv = world_to_pixel(point_world, T_world_to_gripper, view)
    vis = img.copy()
    if np.isnan(uv[0]):
        return vis
    u, v = int(round(uv[0])), int(round(uv[1]))
    cv2.circle(vis, (u, v), radius, color, 2)
    if label:
        cv2.putText(vis, label, (u + 12, v - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pathlib import Path

    CAPTURES = Path('/home/vsc/LLM_TUNE/AIC_Sejong/aic_data/captures')
    sessions = sorted([p for p in CAPTURES.iterdir() if (p / 'steps.jsonl').exists()])
    sess = sessions[0]

    # 첫 insert 스텝 로드
    with open(sess / 'steps.jsonl') as f:
        for line in f:
            d = json.loads(line)
            if d['phase'] == 'insert':
                break

    T_grip = pose_to_T(
        d['transforms']['gripper']['translation'],
        d['transforms']['gripper']['rotation'],
    )
    plug_world = np.array([
        d['transforms']['plug']['translation']['x'],
        d['transforms']['plug']['translation']['y'],
        d['transforms']['plug']['translation']['z'],
    ])
    port_world = np.array([
        d['transforms']['port']['translation']['x'],
        d['transforms']['port']['translation']['y'],
        d['transforms']['port']['translation']['z'],
    ])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for col_i, view in enumerate(['left', 'center', 'right']):
        img_path = d['observation'][f'{view}_image']['path']
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 투영 시각화 (원본)
        vis = draw_projection_on_image(rgb, plug_world, T_grip, view,
                                       color=(0, 255, 0), label='plug')
        vis = draw_projection_on_image(vis, port_world, T_grip, view,
                                       color=(255, 80, 80), label='port')
        axes[0][col_i].imshow(vis)
        axes[0][col_i].set_title(f'{view} — 투영 확인')
        axes[0][col_i].axis('off')

        # adaptive crop (port 위치 기준)
        result = crop_roi(rgb, port_world, T_grip, view,
                          half_size_px=adaptive_half_size(result.depth
                                                          if col_i > 0 else 0.25))
        axes[1][col_i].imshow(result.crop)
        axes[1][col_i].set_title(
            f'{view} — port crop\n'
            f'u={result.u:.0f} v={result.v:.0f} depth={result.depth:.3f}m'
        )
        axes[1][col_i].axis('off')

    plt.suptitle(f'삽입 ROI 크롭 — {sess.name}\nstep={d["step"]} phase={d["phase"]}',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('crop_sanity_check.png', dpi=120)
    plt.show()
    print('Done. crop_sanity_check.png 저장됨')
