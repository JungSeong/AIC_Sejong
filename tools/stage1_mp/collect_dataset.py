#!/usr/bin/env python3
"""
YOLO 학습용 자동 데이터 수집 스크립트
========================================

ground_truth=true 시뮬레이션에서:
  - 3대 카메라 이미지 캡처
  - TF에서 포트 3D 좌표 조회
  - 3D → 이미지 투영으로 bounding box 자동 생성
  - YOLO 형식으로 저장

출력 구조 (YOLO dataset format):
  ~/aic_yolo_dataset/
  ├── images/
  │   ├── train/  episode_00001_left.jpg, ...
  │   └── val/
  ├── labels/
  │   ├── train/  episode_00001_left.txt  (YOLO 형식)
  │   └── val/
  └── data.yaml

YOLO label 형식 (정규화된 0~1):
  class_id  x_center  y_center  width  height

Classes:
  0: sfp_port
  1: sc_port

사용법:
  # 시뮬레이터 실행 (별도 터미널)
  distrobox enter -r aic_eval -- /entrypoint.sh \\
    spawn_task_board:=true spawn_cable:=true \\
    cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \\
    ground_truth:=true start_aic_engine:=false

  # 스크립트 실행
  cd ~/AIC_Sejong/ws_aic/src/aic
  pixi run python /home/sch24/collect_dataset.py --episodes 100
"""
import argparse
import os
import sys
import time
import json
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener, TransformException


# ───────────────────────────────────────────────
#  수집할 포트 프레임들 + 클래스 ID
# ───────────────────────────────────────────────
PORT_DEFINITIONS = [
    # (class_id, class_name, 가능한 TF 프레임들, 포트 실제 크기 (w, h) in meters)
    (0, "sfp_port", [
        "task_board/nic_card_mount_0/sfp_port_0_link",
        "task_board/nic_card_mount_0/sfp_port_1_link",
        "task_board/nic_card_mount_1/sfp_port_0_link",
        "task_board/nic_card_mount_1/sfp_port_1_link",
        "task_board/nic_card_mount_2/sfp_port_0_link",
        "task_board/nic_card_mount_2/sfp_port_1_link",
        "task_board/nic_card_mount_3/sfp_port_0_link",
        "task_board/nic_card_mount_3/sfp_port_1_link",
        "task_board/nic_card_mount_4/sfp_port_0_link",
        "task_board/nic_card_mount_4/sfp_port_1_link",
    ], (0.014, 0.010)),  # SFP 포트 대략 크기
    (1, "sc_port", [
        "task_board/sc_port_0/sc_port_base_link",
        "task_board/sc_port_1/sc_port_base_link",
    ], (0.012, 0.025)),  # SC 포트 대략 크기
]

CAMERAS = [
    ("left",   "left_camera/optical"),
    ("center", "center_camera/optical"),
    ("right",  "right_camera/optical"),
]


# ───────────────────────────────────────────────
#  수학 유틸리티
# ───────────────────────────────────────────────
def transform_to_matrix(t) -> np.ndarray:
    tx, ty, tz = t.translation.x, t.translation.y, t.translation.z
    qx, qy, qz, qw = t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w

    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [tx, ty, tz]
    return M


def project_to_camera(point_3d_base, K, T_base_to_cam):
    p_homo = np.append(point_3d_base, 1.0)
    p_cam = T_base_to_cam @ p_homo
    x, y, z = p_cam[:3]
    if z < 1e-6:
        return None, None, None
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return float(u), float(v), float(z)


def compute_bbox_from_size(u, v, depth, port_size_m, K, margin=1.2):
    """포트 실제 크기를 이미지 bbox로 변환 (핀홀 근사)."""
    w_m, h_m = port_size_m
    # bbox_pixel = focal * real_size / depth
    bbox_w = (K[0, 0] * w_m) / depth * margin
    bbox_h = (K[1, 1] * h_m) / depth * margin
    return bbox_w, bbox_h


# ───────────────────────────────────────────────
#  ROS 노드
# ───────────────────────────────────────────────
class DatasetCollector(Node):
    def __init__(self):
        super().__init__("dataset_collector")

        self._cam_info = {}
        self._latest_image = {}

        for name, _ in CAMERAS:
            self.create_subscription(
                CameraInfo, f"/{name}_camera/camera_info",
                lambda msg, n=name: self._on_info(n, msg), 10,
            )
            self.create_subscription(
                Image, f"/{name}_camera/image",
                lambda msg, n=name: self._on_image(n, msg), 10,
            )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.get_logger().info("DatasetCollector ready.")

    def _on_info(self, name, msg):
        self._cam_info[name] = msg

    def _on_image(self, name, msg):
        # RGB8 or BGR8 image
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )
        # Gazebo는 RGB로 publish, OpenCV는 BGR이라 변환
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self._latest_image[name] = img

    def has_all_data(self):
        return (len(self._cam_info) == len(CAMERAS)
                and len(self._latest_image) == len(CAMERAS))

    def discover_existing_ports(self):
        """현재 시뮬레이션에 존재하는 포트 프레임을 조회."""
        found = []
        for class_id, class_name, candidate_frames, size_m in PORT_DEFINITIONS:
            for frame in candidate_frames:
                try:
                    tf = self._tf_buffer.lookup_transform(
                        "base_link", frame, Time()
                    )
                    pos = np.array([
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        tf.transform.translation.z,
                    ])
                    found.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "frame": frame,
                        "pos_3d": pos,
                        "size_m": size_m,
                    })
                except TransformException:
                    continue
        return found

    def collect_one_frame(self, episode_id, out_dir, is_val=False, debug=False):
        """현재 장면의 이미지 + 라벨을 저장."""
        split = "val" if is_val else "train"

        # 1. 카메라 TF
        cam_T_in_base = {}
        for name, frame in CAMERAS:
            try:
                tf = self._tf_buffer.lookup_transform("base_link", frame, Time())
                cam_T_in_base[name] = transform_to_matrix(tf.transform)
            except TransformException as ex:
                self.get_logger().warn(f"카메라 {name} TF 실패: {ex}")
                return False

        # 2. 존재하는 포트 조회
        ports = self.discover_existing_ports()
        if not ports:
            if debug:
                self.get_logger().warn("포트 TF를 찾지 못함")
            return False

        if debug and episode_id == 0:
            self.get_logger().info(f"발견된 포트 {len(ports)}개:")
            for p in ports:
                self.get_logger().info(
                    f"  {p['class_name']:10s} @ {p['frame']}"
                    f"  pos=({p['pos_3d'][0]:+.3f}, {p['pos_3d'][1]:+.3f}, "
                    f"{p['pos_3d'][2]:+.3f})"
                )

        # 3. 각 카메라별로 저장
        saved = False
        for name, _ in CAMERAS:
            if name not in self._latest_image:
                if debug:
                    self.get_logger().warn(f"카메라 {name}: 이미지 없음")
                continue

            img = self._latest_image[name].copy()
            h, w = img.shape[:2]
            K = np.array(self._cam_info[name].k).reshape(3, 3)
            T_base_to_cam = np.linalg.inv(cam_T_in_base[name])

            # 각 포트를 이미지에 투영
            yolo_labels = []
            skipped_reasons = []
            for p in ports:
                u, v, depth = project_to_camera(p["pos_3d"], K, T_base_to_cam)
                if u is None:
                    skipped_reasons.append(f"{p['frame']}: behind camera")
                    continue
                # 이미지 영역 밖이면 스킵
                if u < 0 or u >= w or v < 0 or v >= h:
                    skipped_reasons.append(
                        f"{p['frame']}: out of image (u={u:.0f}, v={v:.0f}, "
                        f"img={w}x{h})"
                    )
                    continue
                if depth < 0.05 or depth > 2.0:
                    skipped_reasons.append(
                        f"{p['frame']}: depth {depth:.2f} out of range"
                    )
                    continue

                bbox_w, bbox_h = compute_bbox_from_size(
                    u, v, depth, p["size_m"], K
                )

                x_center_norm = u / w
                y_center_norm = v / h
                w_norm = bbox_w / w
                h_norm = bbox_h / h

                x_center_norm = np.clip(x_center_norm, 0, 1)
                y_center_norm = np.clip(y_center_norm, 0, 1)
                w_norm = np.clip(w_norm, 0.001, 1)
                h_norm = np.clip(h_norm, 0.001, 1)

                yolo_labels.append(
                    f"{p['class_id']} {x_center_norm:.6f} {y_center_norm:.6f} "
                    f"{w_norm:.6f} {h_norm:.6f}"
                )

            if debug and episode_id == 0:
                self.get_logger().info(
                    f"카메라 {name}: 유효 라벨 {len(yolo_labels)}개"
                )
                for r in skipped_reasons:
                    self.get_logger().info(f"    스킵: {r}")

            if not yolo_labels:
                continue

            # 저장
            stem = f"ep{episode_id:05d}_{name}"
            img_path = out_dir / "images" / split / f"{stem}.jpg"
            lbl_path = out_dir / "labels" / split / f"{stem}.txt"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            lbl_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(img_path), img)
            lbl_path.write_text("\n".join(yolo_labels))
            saved = True

        return saved


# ───────────────────────────────────────────────
#  메인
# ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100,
                        help="수집할 에피소드 수 (현재 장면에서 n개 스냅샷)")
    parser.add_argument("--output", type=str, default="~/aic_yolo_dataset",
                        help="출력 디렉토리")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="검증 세트 비율")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="프레임 간 대기 시간 (초) - 다양한 로봇 자세 위해")
    args = parser.parse_args()

    out_dir = Path(os.path.expanduser(args.output))
    out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = DatasetCollector()

    # 초기 데이터 수신 대기
    node.get_logger().info("카메라 데이터 수신 대기 (최대 15초)...")
    start = time.time()
    while not node.has_all_data() and (time.time() - start) < 15.0:
        rclpy.spin_once(node, timeout_sec=0.2)

    if not node.has_all_data():
        node.get_logger().error("카메라 데이터 수신 실패")
        sys.exit(1)

    # TF 버퍼가 충분히 쌓이도록 5초간 spin
    node.get_logger().info("TF 버퍼 누적 대기 (5초간 spin)...")
    tf_start = time.time()
    while (time.time() - tf_start) < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    # 포트 TF 확인 (찾을 때까지 추가로 최대 10초 더 대기)
    ports = node.discover_existing_ports()
    extra_start = time.time()
    while not ports and (time.time() - extra_start) < 10.0:
        rclpy.spin_once(node, timeout_sec=0.2)
        ports = node.discover_existing_ports()

    if not ports:
        node.get_logger().error(
            "포트 TF를 전혀 못 찾음. 다음을 확인:\n"
            "  1. 시뮬레이터에 nic_card_mount_0_present:=true 옵션 있는지\n"
            "  2. Gazebo 화면에 NIC 카드가 실제로 보이는지\n"
            "  3. check_frames.py 결과에 포트 프레임이 있는지"
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    node.get_logger().info(f"포트 {len(ports)}개 확인. 수집 시작.")

    # 수집
    saved_count = 0
    for i in range(args.episodes):
        rclpy.spin_once(node, timeout_sec=0.1)

        is_val = (i % int(1 / args.val_ratio) == 0) if args.val_ratio > 0 else False
        # 첫 프레임은 항상 디버그 모드
        ok = node.collect_one_frame(i, out_dir, is_val=is_val, debug=(i == 0))
        if ok:
            saved_count += 1

        if (i + 1) % 10 == 0:
            node.get_logger().info(
                f"진행: {i+1}/{args.episodes} (저장 {saved_count}개)"
            )
        time.sleep(args.interval)

    # data.yaml 생성
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: sfp_port\n"
        f"  1: sc_port\n"
    )

    node.get_logger().info(
        f"\n완료!\n"
        f"  저장 위치: {out_dir}\n"
        f"  저장된 프레임: {saved_count} / {args.episodes}\n"
        f"  총 이미지 파일: "
        f"{len(list((out_dir / 'images').rglob('*.jpg')))}\n"
        f"  data.yaml: {data_yaml}\n"
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
