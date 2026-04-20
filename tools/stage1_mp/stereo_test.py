#!/usr/bin/env python3
"""
Stereo Triangulation Verification Test
=====================================

이 대회 환경에서 스테레오 삼각측량 방식이 얼마나 정확한지 검증.

절차:
  1. Ground truth TF에서 포트 3D 좌표 가져옴 (정답)
  2. 그 좌표를 각 카메라에 투영해서 (u, v) 픽셀 얻음
  3. 두 카메라 쌍으로 삼각측량 수행 → 3D 복원
  4. 원본과 복원값의 오차 출력

목적:
  - 스테레오 파이프라인(수학/TF 변환)이 맞는지 검증
  - 이 대회 환경에서 스테레오 복원 정확도 수치 확보
  - Vision 검출 모듈 완성 시 바로 활용 가능한 인프라 구축

실행:
  cd ~/AIC_Sejong/ws_aic/src/aic
  pixi run python /home/sch24/stereo_test.py
"""
import sys
import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformListener, TransformException


# ───────────────────────────────────────────────
#  테스트할 타겟 프레임 (존재하는 것부터 사용)
# ───────────────────────────────────────────────
TARGET_FRAMES = [
    # Trial 1 후보
    "task_board/nic_card_mount_0/sfp_port_0_link",
    "task_board/nic_card_mount_0/sfp_port_1_link",
    # Trial 2 후보
    "task_board/nic_card_mount_1/sfp_port_0_link",
    "task_board/nic_card_mount_1/sfp_port_1_link",
    # Trial 3 후보
    "task_board/sc_port_0/sc_port_base_link",
    "task_board/sc_port_1/sc_port_base_link",
    # Fallback (항상 존재)
    "gripper/tcp",
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
    """geometry_msgs/Transform → 4x4 homogeneous matrix."""
    tx, ty, tz = t.translation.x, t.translation.y, t.translation.z
    qx, qy, qz, qw = t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w

    # 쿼터니언 → 회전 행렬
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    R = np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy)],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)],
    ])

    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [tx, ty, tz]
    return M


def project_to_camera(
    point_3d_base: np.ndarray, K: np.ndarray, T_base_to_cam: np.ndarray
) -> tuple[float, float]:
    """3D base 좌표를 카메라 이미지 픽셀 (u, v)로 투영."""
    p_homo = np.append(point_3d_base, 1.0)
    p_cam = T_base_to_cam @ p_homo
    x, y, z = p_cam[:3]
    if z < 1e-6:
        return -1.0, -1.0
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return float(u), float(v)


def triangulate(
    u_a: float, v_a: float, K_a: np.ndarray, T_base_to_a: np.ndarray,
    u_b: float, v_b: float, K_b: np.ndarray, T_base_to_b: np.ndarray,
) -> np.ndarray:
    """두 카메라의 픽셀 좌표로 3D base 좌표 복원."""
    # P = K @ [R | t]  (3x4)  base → image
    P_a = K_a @ T_base_to_a[:3, :]
    P_b = K_b @ T_base_to_b[:3, :]

    pts_4d = cv2.triangulatePoints(
        P_a, P_b,
        np.array([[u_a], [v_a]], dtype=np.float64),
        np.array([[u_b], [v_b]], dtype=np.float64),
    )
    pts_3d = (pts_4d[:3] / pts_4d[3]).flatten()
    return pts_3d


# ───────────────────────────────────────────────
#  ROS2 노드
# ───────────────────────────────────────────────
class StereoTester(Node):
    def __init__(self):
        super().__init__("stereo_tester")

        self._cam_info: dict[str, CameraInfo] = {}
        for name, _ in CAMERAS:
            self.create_subscription(
                CameraInfo,
                f"/{name}_camera/camera_info",
                lambda msg, n=name: self._on_info(n, msg),
                10,
            )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.get_logger().info("StereoTester started.")

    def _on_info(self, name: str, msg: CameraInfo) -> None:
        self._cam_info[name] = msg

    def has_all_camera_info(self) -> bool:
        return len(self._cam_info) == len(CAMERAS)

    def run_test(self) -> None:
        # 1. 각 카메라의 base_link 기준 pose
        cam_T_in_base: dict[str, np.ndarray] = {}
        for name, frame in CAMERAS:
            try:
                tf = self._tf_buffer.lookup_transform("base_link", frame, Time())
                cam_T_in_base[name] = transform_to_matrix(tf.transform)
                self.get_logger().info(
                    f"  [camera] {name:6s}  pos="
                    f"({tf.transform.translation.x:+.3f}, "
                    f"{tf.transform.translation.y:+.3f}, "
                    f"{tf.transform.translation.z:+.3f})"
                )
            except TransformException as ex:
                self.get_logger().error(f"camera {name} TF 없음: {ex}")
                return

        # 2. 테스트할 타겟 프레임 찾기
        target_name = None
        target_3d = None
        for frame in TARGET_FRAMES:
            try:
                tf = self._tf_buffer.lookup_transform("base_link", frame, Time())
                target_name = frame
                target_3d = np.array([
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z,
                ])
                break
            except TransformException:
                continue

        if target_3d is None:
            self.get_logger().error("테스트할 프레임을 찾지 못함")
            return

        # 3. 각 카메라에 투영
        projections: dict[str, tuple[float, float]] = {}
        for name, _ in CAMERAS:
            K = np.array(self._cam_info[name].k).reshape(3, 3)
            T_base_to_cam = np.linalg.inv(cam_T_in_base[name])
            u, v = project_to_camera(target_3d, K, T_base_to_cam)
            projections[name] = (u, v)

        # 4. 각 카메라 쌍으로 삼각측량
        pairs = [("left", "center"), ("left", "right"), ("center", "right")]

        header = "=" * 70
        print()
        print(header)
        print(f"  Target frame: {target_name}")
        print(f"  Ground truth 3D (base_link): "
              f"({target_3d[0]:+.6f}, {target_3d[1]:+.6f}, {target_3d[2]:+.6f})")
        print(header)
        print(f"  Image size: {self._cam_info['left'].width} x "
              f"{self._cam_info['left'].height}")
        print()
        print("  투영 결과 (ground truth → 이미지 픽셀):")
        for name, _ in CAMERAS:
            u, v = projections[name]
            print(f"    [{name:6s}] u={u:7.2f}, v={v:7.2f}")
        print()
        print("  스테레오 삼각측량 (두 픽셀 → 3D 복원):")
        print(f"    {'pair':<18s} {'restored (x, y, z)':<42s} {'error':>10s}")
        print("    " + "-" * 74)

        all_errors = []
        for cam_a, cam_b in pairs:
            u_a, v_a = projections[cam_a]
            u_b, v_b = projections[cam_b]

            K_a = np.array(self._cam_info[cam_a].k).reshape(3, 3)
            K_b = np.array(self._cam_info[cam_b].k).reshape(3, 3)
            T_base_to_a = np.linalg.inv(cam_T_in_base[cam_a])
            T_base_to_b = np.linalg.inv(cam_T_in_base[cam_b])

            pts_3d = triangulate(
                u_a, v_a, K_a, T_base_to_a,
                u_b, v_b, K_b, T_base_to_b,
            )
            err = float(np.linalg.norm(pts_3d - target_3d))
            all_errors.append(err)

            pair_name = f"{cam_a}+{cam_b}"
            restored = f"({pts_3d[0]:+.6f}, {pts_3d[1]:+.6f}, {pts_3d[2]:+.6f})"
            print(f"    {pair_name:<18s} {restored:<42s} {err*1000:7.3f} mm")

        print()
        avg_err = float(np.mean(all_errors))
        print(f"  평균 복원 오차: {avg_err*1000:.3f} mm")
        print(header)

        # 5. 해석
        print()
        if avg_err < 1e-4:
            print("  ✅ 스테레오 파이프라인 완벽 동작 (수치 오차 = 반올림 수준)")
            print("     → 실제 검출 (u, v) 넣으면 동일 품질로 복원 가능")
        elif avg_err < 1e-3:
            print("  ✅ 스테레오 파이프라인 정상 (sub-mm 정확도)")
        else:
            print("  ⚠️  복원 오차가 예상보다 큼 — TF/intrinsic/변환 순서 점검 필요")

        print()
        print("  [다음 단계] 실제 검출 (u, v)를 쓸 때:")
        print("    오차 = (검출 픽셀 오차) × (depth / focal length)")
        print("    예: 픽셀 오차 2px, depth 0.5m, f=1236 → 3D 오차 ≈ 0.8mm")
        print()


# ───────────────────────────────────────────────
#  메인
# ───────────────────────────────────────────────
def main():
    rclpy.init()
    node = StereoTester()

    # 카메라 info 수신 대기
    node.get_logger().info("카메라 info 토픽 구독 대기 중 (최대 15초)...")
    start = time.time()
    while not node.has_all_camera_info() and (time.time() - start) < 15.0:
        rclpy.spin_once(node, timeout_sec=0.2)

    if not node.has_all_camera_info():
        missing = [n for n, _ in CAMERAS if n not in node._cam_info]
        node.get_logger().error(
            f"카메라 info 수신 실패: {missing}\n"
            "시뮬레이터가 실행 중인지, ground_truth=true 로 시작했는지 확인 필요."
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    node.get_logger().info("카메라 info 모두 수신. TF 안정화 대기 (2초)...")

    # TF가 누적될 시간 확보
    tf_start = time.time()
    while (time.time() - tf_start) < 2.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.run_test()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
