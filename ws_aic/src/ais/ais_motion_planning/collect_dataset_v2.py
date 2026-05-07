#!/usr/bin/env python3
"""
YOLO 학습용 자동 데이터 수집 스크립트 v2
==========================================

v1 대비 추가:
  - 로봇 자세 자동 순회 (다양한 각도에서 수집)
  - 각 뷰포인트마다 이미지 3장(left/center/right) 저장
  - 정지 후 안정화 대기로 블러 방지

사용법:
  # 시뮬레이터 실행 (별도 터미널)
  distrobox enter -r aic_eval -- /entrypoint.sh \\
    spawn_task_board:=true \\
    nic_card_mount_0_present:=true \\
    sc_port_0_present:=true \\
    spawn_cable:=true cable_type:=sfp_sc_cable \\
    attach_cable_to_gripper:=true \\
    ground_truth:=true start_aic_engine:=false

  # 자동 수집 실행
  cd ~/AIC_Sejong/ws_aic/src/aic
  pixi run python /home/sch24/collect_dataset_v2.py --episodes 500

  # 뷰포인트 수 조정
  pixi run python /home/sch24/collect_dataset_v2.py \\
    --episodes 500 --n_viewpoints 20

라벨 형식 (YOLO): class_id x_center y_center width height
Classes: 0=sfp_port, 1=sc_port
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import Pose, Point, Quaternion, Wrench, Vector3
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener, TransformException

# aic 컨트롤러 인터페이스
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode


# ───────────────────────────────────────────────
#  수집 대상 포트 + 클래스
# ───────────────────────────────────────────────
PORT_DEFINITIONS = [
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
    ], (0.014, 0.010)),
    (1, "sc_port", [
        "task_board/sc_port_0/sc_port_base_link",
        "task_board/sc_port_1/sc_port_base_link",
    ], (0.012, 0.025)),
]

CAMERAS = [
    ("left",   "left_camera/optical"),
    ("center", "center_camera/optical"),
    ("right",  "right_camera/optical"),
]


# ───────────────────────────────────────────────
#  로봇 뷰포인트 정의
# ───────────────────────────────────────────────
# 각 뷰포인트는 task_board(보드 중심) 주변의 그리퍼 목표 pose
# base_link 기준 좌표 (x, y, z) + 오일러 각 (roll, pitch, yaw)
#
# 보드 대략 위치: base_link 기준 (-0.38, +0.22, +0.13) 근처
# 그래서 그리퍼 목표는 이 주변에서 위/옆/각도 다양하게.


def euler_to_quat(roll, pitch, yaw):
    """Roll-pitch-yaw (XYZ intrinsic) → Quaternion (w, x, y, z)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)


def make_viewpoints(n=10, board_center=(-0.38, 0.22, 0.13)):
    """다양한 뷰포인트 생성.

    보드 중심 위/주변에서 그리퍼가 보드를 바라보는 자세들.
    """
    bx, by, bz = board_center
    vps = []

    # 1. 중앙 위 (수직)
    vps.append((bx, by, bz + 0.40, np.pi, 0.0, 0.0))

    # 2. 중앙 위 (멀리)
    vps.append((bx, by, bz + 0.55, np.pi, 0.0, 0.0))

    # 3. 중앙 위 (가까이)
    vps.append((bx, by, bz + 0.30, np.pi, 0.0, 0.0))

    # 4~7. 네 방향 기울여 바라보기
    for dx, dy, roll_off, pitch_off in [
        (+0.10, 0.00, 0.0, +0.3),   # 앞에서 본다 (아래 기울임)
        (-0.10, 0.00, 0.0, -0.3),   # 뒤에서
        (0.00, +0.10, +0.3, 0.0),   # 오른쪽
        (0.00, -0.10, -0.3, 0.0),   # 왼쪽
    ]:
        vps.append((bx + dx, by + dy, bz + 0.40, np.pi + roll_off, pitch_off, 0.0))

    # 8~11. 대각선 뷰
    for dx, dy in [(+0.08, +0.08), (+0.08, -0.08),
                   (-0.08, +0.08), (-0.08, -0.08)]:
        vps.append((bx + dx, by + dy, bz + 0.40, np.pi, 0.0, np.arctan2(dy, dx)))

    # 12~. 추가 랜덤 변형
    np.random.seed(42)
    while len(vps) < n:
        dx = np.random.uniform(-0.12, 0.12)
        dy = np.random.uniform(-0.12, 0.12)
        dz = np.random.uniform(0.30, 0.55)
        roll = np.pi + np.random.uniform(-0.3, 0.3)
        pitch = np.random.uniform(-0.3, 0.3)
        yaw = np.random.uniform(-0.5, 0.5)
        vps.append((bx + dx, by + dy, bz + dz, roll, pitch, yaw))

    return vps[:n]


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
        [1-2*(yy+zz),   2*(xy-wz),   2*(xz+wy)],
        [  2*(xy+wz), 1-2*(xx+zz),   2*(yz-wx)],
        [  2*(xz-wy),   2*(yz+wx), 1-2*(xx+yy)],
    ])
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = [tx, ty, tz]
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
    w_m, h_m = port_size_m
    bbox_w = (K[0, 0] * w_m) / depth * margin
    bbox_h = (K[1, 1] * h_m) / depth * margin
    return bbox_w, bbox_h


# ───────────────────────────────────────────────
#  ROS 노드
# ───────────────────────────────────────────────
class DatasetCollector(Node):
    def __init__(self):
        super().__init__("dataset_collector_v2")

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

        # 로봇 제어 퍼블리셔
        self._motion_pub = self.create_publisher(
            MotionUpdate, "/aic_controller/pose_commands", 10
        )

        self.get_logger().info("DatasetCollector v2 ready.")

    def _on_info(self, name, msg):
        self._cam_info[name] = msg

    def _on_image(self, name, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self._latest_image[name] = img

    def has_all_data(self):
        return (len(self._cam_info) == 3
                and len(self._latest_image) == 3)

    def _lookup_port_tf(self, frame):
        entrance_frame = f"{frame}_entrance"
        try:
            tf = self._tf_buffer.lookup_transform(
                "base_link", entrance_frame, Time())
            return tf, entrance_frame
        except TransformException:
            tf = self._tf_buffer.lookup_transform("base_link", frame, Time())
            return tf, frame

    def discover_existing_ports(self):
        found = []
        for class_id, class_name, candidate_frames, size_m in PORT_DEFINITIONS:
            for frame in candidate_frames:
                try:
                    tf, resolved_frame = self._lookup_port_tf(frame)
                    pos = np.array([
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        tf.transform.translation.z,
                    ])
                    found.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "frame": resolved_frame,
                        "base_frame": frame,
                        "pos_3d": pos,
                        "size_m": size_m,
                    })
                except TransformException:
                    continue
        return found

    def move_robot_to(self, x, y, z, roll, pitch, yaw,
                      stiffness=None, damping=None):
        """로봇 그리퍼를 목표 자세로 이동 명령."""
        if stiffness is None:
            stiffness = [200.0, 200.0, 200.0, 50.0, 50.0, 50.0]
        if damping is None:
            damping = [80.0, 80.0, 80.0, 20.0, 20.0, 20.0]

        w, qx, qy, qz = euler_to_quat(roll, pitch, yaw)
        msg = MotionUpdate(
            header=Header(
                frame_id="base_link",
                stamp=self.get_clock().now().to_msg(),
            ),
            pose=Pose(
                position=Point(x=float(x), y=float(y), z=float(z)),
                orientation=Quaternion(w=w, x=qx, y=qy, z=qz),
            ),
            target_stiffness=np.diag(stiffness).flatten().tolist(),
            target_damping=np.diag(damping).flatten().tolist(),
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )
        self._motion_pub.publish(msg)

    def collect_one_frame(self, episode_id, out_dir, is_val=False, debug=False,
                          stem_prefix=""):
        split = "val" if is_val else "train"

        cam_T_in_base = {}
        for name, frame in CAMERAS:
            try:
                tf = self._tf_buffer.lookup_transform("base_link", frame, Time())
                cam_T_in_base[name] = transform_to_matrix(tf.transform)
            except TransformException:
                return False

        ports = self.discover_existing_ports()
        if not ports:
            return False

        saved = False
        for name, _ in CAMERAS:
            if name not in self._latest_image:
                continue

            img = self._latest_image[name].copy()
            h, w = img.shape[:2]
            K = np.array(self._cam_info[name].k).reshape(3, 3)
            T_base_to_cam = np.linalg.inv(cam_T_in_base[name])

            yolo_labels = []
            for p in ports:
                u, v, depth = project_to_camera(p["pos_3d"], K, T_base_to_cam)
                if u is None or u < 0 or u >= w or v < 0 or v >= h:
                    continue
                if depth < 0.05 or depth > 2.0:
                    continue

                bbox_w, bbox_h = compute_bbox_from_size(
                    u, v, depth, p["size_m"], K)

                x_c = np.clip(u / w, 0, 1)
                y_c = np.clip(v / h, 0, 1)
                w_n = np.clip(bbox_w / w, 0.001, 1)
                h_n = np.clip(bbox_h / h, 0.001, 1)

                yolo_labels.append(
                    f"{p['class_id']} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"
                )

            if not yolo_labels:
                continue

            stem = f"{stem_prefix}ep{episode_id:05d}_{name}"
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
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--n_viewpoints", type=int, default=15,
                        help="로봇 자세 뷰포인트 수")
    parser.add_argument("--output", type=str, default="../../data/yolo")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--move_settle_s", type=float, default=2.5,
                        help="로봇 이동 후 안정화 대기 시간")
    parser.add_argument("--frames_per_viewpoint", type=int, default=None,
                        help="뷰포인트당 수집 프레임 수 (기본: episodes/n_viewpoints)")
    parser.add_argument("--stem_prefix", type=str, default="",
                        help="저장 파일명 prefix (시나리오 반복 수집 시 덮어쓰기 방지)")
    args = parser.parse_args()

    date_dir = datetime.now().strftime("%Y%m%d")
    out_dir = Path(os.path.expanduser(args.output)) / date_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 뷰포인트 생성
    viewpoints = make_viewpoints(n=args.n_viewpoints)
    frames_per_vp = args.frames_per_viewpoint or max(
        1, args.episodes // len(viewpoints)
    )
    total_frames = len(viewpoints) * frames_per_vp

    print(f"수집 계획:")
    print(f"  뷰포인트 수: {len(viewpoints)}")
    print(f"  뷰포인트당 프레임: {frames_per_vp}")
    print(f"  총 프레임: {total_frames} (각 카메라 3장 = {total_frames*3}장)")
    print(f"  출력: {out_dir}\n")

    rclpy.init()
    node = DatasetCollector()

    # 데이터 수신 대기
    node.get_logger().info("카메라 데이터 수신 대기 (최대 15초)...")
    start = time.time()
    while not node.has_all_data() and (time.time() - start) < 15.0:
        rclpy.spin_once(node, timeout_sec=0.2)
    if not node.has_all_data():
        node.get_logger().error("카메라 데이터 수신 실패")
        sys.exit(1)

    # TF 누적 대기
    node.get_logger().info("TF 버퍼 누적 대기 (5초)...")
    t0 = time.time()
    while (time.time() - t0) < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    ports = node.discover_existing_ports()
    if not ports:
        node.get_logger().error("포트 TF를 찾지 못함")
        sys.exit(1)
    node.get_logger().info(f"포트 {len(ports)}개 확인")

    # 수집 루프
    saved_count = 0
    episode_id = 0

    try:
        for vp_idx, (x, y, z, roll, pitch, yaw) in enumerate(viewpoints):
            node.get_logger().info(
                f"\n━━ 뷰포인트 {vp_idx+1}/{len(viewpoints)} ━━\n"
                f"  pos=({x:+.3f}, {y:+.3f}, {z:+.3f})\n"
                f"  rot(rpy)=({roll:+.2f}, {pitch:+.2f}, {yaw:+.2f})"
            )

            # 로봇 이동
            for _ in range(5):  # 여러 번 publish해서 확실히 수신
                node.move_robot_to(x, y, z, roll, pitch, yaw)
                rclpy.spin_once(node, timeout_sec=0.05)
                time.sleep(0.1)

            # 이동 + 안정화 대기
            settle_start = time.time()
            while (time.time() - settle_start) < args.move_settle_s:
                rclpy.spin_once(node, timeout_sec=0.1)

            # 이 뷰포인트에서 여러 프레임 수집
            for k in range(frames_per_vp):
                rclpy.spin_once(node, timeout_sec=0.1)
                is_val = (episode_id % int(1 / args.val_ratio) == 0
                          if args.val_ratio > 0 else False)
                if node.collect_one_frame(episode_id, out_dir, is_val=is_val,
                                          debug=(episode_id == 0),
                                          stem_prefix=args.stem_prefix):
                    saved_count += 1
                episode_id += 1
                time.sleep(0.1)

            node.get_logger().info(
                f"  진행: {episode_id}/{total_frames} "
                f"(저장 {saved_count} 프레임)"
            )

    except KeyboardInterrupt:
        node.get_logger().info("수집 중단됨")

    # data.yaml 생성
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: sfp_port\n"
        f"  1: sc_port\n"
    )

    # 최종 통계
    train_imgs = len(list((out_dir / "images" / "train").glob("*.jpg"))) \
        if (out_dir / "images" / "train").exists() else 0
    val_imgs = len(list((out_dir / "images" / "val").glob("*.jpg"))) \
        if (out_dir / "images" / "val").exists() else 0

    node.get_logger().info(
        f"\n완료!\n"
        f"  총 프레임: {saved_count} / {total_frames}\n"
        f"  train 이미지: {train_imgs}장\n"
        f"  val 이미지: {val_imgs}장\n"
        f"  총 이미지: {train_imgs + val_imgs}장\n"
        f"  data.yaml: {out_dir / 'data.yaml'}\n"
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
