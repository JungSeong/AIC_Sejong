#!/usr/bin/env python3
"""
YOLO 포트 검출 + 스테레오 삼각측량 → 3D 좌표
================================================

실시간으로:
  1. 3대 카메라 이미지 구독
  2. 각 이미지에서 YOLO로 포트 검출
  3. 가장 잘 보이는 두 카메라 선택
  4. 스테레오 삼각측량으로 3D 좌표 계산
  5. 결과 출력 + 선택적으로 ground truth와 비교

사용법:
  cd ~/AIC_Sejong/ws_aic/src/aic
  pixi run python /home/sch24/port_detector_3d.py \\
    --model ~/aic_yolo_runs/port_detector/weights/best.pt

옵션:
  --compare: ground truth와 비교하여 오차 출력
  --continuous: 계속 추론 (Ctrl+C로 종료)
"""
import argparse
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener, TransformException


CAMERAS = [
    ("left",   "left_camera/optical"),
    ("center", "center_camera/optical"),
    ("right",  "right_camera/optical"),
]

CLASS_NAMES = {0: "sfp_port", 1: "sc_port"}


# ───────────────────────────────────────────────
#  유틸리티 (stereo_test.py와 공유)
# ───────────────────────────────────────────────
def transform_to_matrix(t):
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


def triangulate(u_a, v_a, K_a, T_base_to_a, u_b, v_b, K_b, T_base_to_b):
    P_a = K_a @ T_base_to_a[:3, :]
    P_b = K_b @ T_base_to_b[:3, :]
    pts_4d = cv2.triangulatePoints(
        P_a, P_b,
        np.array([[u_a], [v_a]], dtype=np.float64),
        np.array([[u_b], [v_b]], dtype=np.float64),
    )
    return (pts_4d[:3] / pts_4d[3]).flatten()


# ───────────────────────────────────────────────
#  포트 검출 + 3D 추정 노드
# ───────────────────────────────────────────────
class PortDetector3D(Node):
    def __init__(self, yolo_model_path, conf_thresh=0.25):
        super().__init__("port_detector_3d")

        from ultralytics import YOLO
        self.yolo = YOLO(yolo_model_path)
        self.conf_thresh = conf_thresh

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

        self.get_logger().info(f"YOLO 모델 로드: {yolo_model_path}")

    def _on_info(self, name, msg):
        self._cam_info[name] = msg

    def _on_image(self, name, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self._latest_image[name] = img

    def ready(self):
        return (len(self._cam_info) == 3 and len(self._latest_image) == 3)

    def detect_ports(self, image):
        """YOLO로 포트 검출 후 (class_id, confidence, u, v, bbox) 리스트 반환."""
        results = self.yolo(image, verbose=False, conf=self.conf_thresh)
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                u = (x1 + x2) / 2
                v = (y1 + y2) / 2
                detections.append({
                    "class_id": cls,
                    "class_name": CLASS_NAMES.get(cls, "unknown"),
                    "conf": conf,
                    "u": float(u),
                    "v": float(v),
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                })
        return detections

    def compute_3d(self, target_class_id=None):
        """모든 카메라에서 검출 후 같은 포트를 매칭하여 3D 계산.

        알고리즘:
          1. 각 카메라에서 YOLO 검출
          2. 카메라 쌍별로 같은 클래스의 검출끼리 매칭
          3. Epipolar 유사성 (rectified라 v 좌표가 비슷해야 함) 체크
          4. 가장 plausible한 매칭으로 삼각측량
        """
        # 1. 카메라 TF
        cam_T_in_base = {}
        for name, frame in CAMERAS:
            try:
                tf = self._tf_buffer.lookup_transform("base_link", frame, Time())
                cam_T_in_base[name] = transform_to_matrix(tf.transform)
            except TransformException:
                return None

        # 2. 각 카메라에서 검출
        all_detections = {}
        for name, _ in CAMERAS:
            img = self._latest_image.get(name)
            if img is None:
                continue
            dets = self.detect_ports(img)
            if target_class_id is not None:
                dets = [d for d in dets if d["class_id"] == target_class_id]
            all_detections[name] = dets

        # 카메라가 2대 이상 검출해야 스테레오 가능
        cameras_with_dets = [n for n, d in all_detections.items() if d]
        if len(cameras_with_dets) < 2:
            return {"error": "need detections from at least 2 cameras",
                    "detections": all_detections}

        # 3. 카메라 쌍 선택 (베이스라인 큰 순)
        pair_priority = [("left", "right"), ("left", "center"), ("center", "right")]
        pair = None
        for a, b in pair_priority:
            if a in cameras_with_dets and b in cameras_with_dets:
                pair = (a, b)
                break
        if pair is None:
            return {"error": "no valid camera pair"}

        cam_a, cam_b = pair
        dets_a = all_detections[cam_a]
        dets_b = all_detections[cam_b]

        K_a = np.array(self._cam_info[cam_a].k).reshape(3, 3)
        K_b = np.array(self._cam_info[cam_b].k).reshape(3, 3)
        T_base_to_a = np.linalg.inv(cam_T_in_base[cam_a])
        T_base_to_b = np.linalg.inv(cam_T_in_base[cam_b])

        # 4. 같은 클래스끼리 모든 조합으로 삼각측량 → 3D 타당성으로 매칭
        # (epipolar 제약은 카메라가 rectified stereo가 아니라 완화)
        board_center = np.array([-0.38, 0.22, 0.13])  # 대략 보드 위치 (base_link)
        all_matches = []

        for da in dets_a:
            for db in dets_b:
                if da["class_id"] != db["class_id"]:
                    continue

                port_3d = triangulate(
                    da["u"], da["v"], K_a, T_base_to_a,
                    db["u"], db["v"], K_b, T_base_to_b,
                )

                # 3D 타당성 검증 — 태스크 보드 근처여야 함
                dist_from_board = float(np.linalg.norm(port_3d - board_center))
                if dist_from_board > 0.5:
                    continue

                # 보드 높이 범위 (z: 0 ~ 0.3 정도)
                if port_3d[2] < -0.1 or port_3d[2] > 0.5:
                    continue

                all_matches.append({
                    "det_a": da, "det_b": db,
                    "port_3d": port_3d,
                    "dist_from_board": dist_from_board,
                    "conf_sum": da["conf"] + db["conf"],
                })

        if not all_matches:
            return {"error": "no valid matching between cameras",
                    "detections": all_detections,
                    "hint": "3D 복원 결과가 모두 보드 범위 밖"}

        # 가장 plausible한 매칭 선택:
        # 보드 중심에서 가깝고 + confidence 합이 큰 것
        best_match = min(
            all_matches,
            key=lambda m: m["dist_from_board"] - 0.1 * m["conf_sum"],
        )

        return {
            "port_3d": best_match["port_3d"],
            "pair": pair,
            "detections": all_detections,
            "selected": {
                cam_a: best_match["det_a"],
                cam_b: best_match["det_b"],
            },
            "dist_from_board": best_match["dist_from_board"],
            "all_matches": all_matches,
        }

    def compare_with_gt(self, port_3d, gt_frames=None):
        """Ground truth TF와 비교 (존재하는 모든 포트와 비교해서 가장 가까운 것)."""
        if gt_frames is None:
            # 모든 가능한 포트 프레임
            gt_frames = []
            for nic_idx in range(5):
                for port_idx in range(2):
                    gt_frames.append(
                        f"task_board/nic_card_mount_{nic_idx}/"
                        f"sfp_port_{port_idx}_link"
                    )
            for sc_idx in range(2):
                gt_frames.append(
                    f"task_board/sc_port_{sc_idx}/sc_port_base_link"
                )
        best_err = float("inf")
        best_frame = None
        best_gt = None
        for frame in gt_frames:
            try:
                tf = self._tf_buffer.lookup_transform("base_link", frame, Time())
                gt = np.array([
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z,
                ])
                err = np.linalg.norm(port_3d - gt)
                if err < best_err:
                    best_err = err
                    best_frame = frame
                    best_gt = gt
            except TransformException:
                continue
        return best_frame, best_gt, best_err


# ───────────────────────────────────────────────
#  메인
# ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="YOLO best.pt 경로")
    parser.add_argument("--target", type=str, default=None,
                        choices=[None, "sfp_port", "sc_port"],
                        help="특정 클래스만 검출")
    parser.add_argument("--compare", action="store_true",
                        help="ground truth와 비교")
    parser.add_argument("--continuous", action="store_true",
                        help="계속 추론 (Ctrl+C 종료)")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    target_class_id = None
    if args.target == "sfp_port":
        target_class_id = 0
    elif args.target == "sc_port":
        target_class_id = 1

    rclpy.init()
    node = PortDetector3D(args.model, conf_thresh=args.conf)

    node.get_logger().info("데이터 수신 대기 (최대 15초)...")
    start = time.time()
    while not node.ready() and (time.time() - start) < 15.0:
        rclpy.spin_once(node, timeout_sec=0.2)
    if not node.ready():
        node.get_logger().error("데이터 수신 실패")
        return

    # TF 버퍼 누적 대기 (5초간 spin)
    node.get_logger().info("TF 버퍼 누적 대기 (5초)...")
    tf_start = time.time()
    while (time.time() - tf_start) < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    try:
        iteration = 0
        while True:
            iteration += 1
            rclpy.spin_once(node, timeout_sec=0.1)
            result = node.compute_3d(target_class_id=target_class_id)

            print("\n" + "=" * 70)
            print(f"  [Iteration {iteration}] 추론 결과")
            print("=" * 70)

            if result is None or "error" in result:
                print(f"  ERROR: {result.get('error', result)}")
                if "hint" in result:
                    print(f"  HINT: {result['hint']}")
                # 디버그: 검출된 것들 요약
                for cam, dets in result.get("detections", {}).items():
                    print(f"    [{cam}] {len(dets)} detections")
            else:
                port_3d = result["port_3d"]
                pair = result["pair"]
                n_matches = len(result.get("all_matches", []))
                print(f"  사용한 카메라 쌍: {pair}  "
                      f"(유효 매칭 {n_matches}개 중 최선 선택)")
                for cam, det in result["selected"].items():
                    print(f"    [{cam:6s}] {det['class_name']:10s} "
                          f"conf={det['conf']:.3f}  u={det['u']:.1f}, v={det['v']:.1f}")
                print(f"  추정 3D 좌표 (base_link): "
                      f"({port_3d[0]:+.4f}, {port_3d[1]:+.4f}, {port_3d[2]:+.4f})")
                print(f"  보드 중심과의 거리: "
                      f"{result['dist_from_board']*100:.1f} cm")

                if args.compare:
                    frame, gt, err = node.compare_with_gt(port_3d)
                    if frame:
                        print(f"  Ground truth (가장 가까운 포트: {frame}):")
                        print(f"    GT:    ({gt[0]:+.4f}, {gt[1]:+.4f}, {gt[2]:+.4f})")
                        print(f"    오차:  {err*1000:.2f} mm")
                    else:
                        print(f"  Ground truth 비교 실패: 어떤 포트 프레임도 "
                              f"TF에서 찾을 수 없음")

            if not args.continuous:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
