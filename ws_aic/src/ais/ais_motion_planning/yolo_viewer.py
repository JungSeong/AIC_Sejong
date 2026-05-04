#!/usr/bin/env python3
"""
YOLO 포트 검출 실시간 시각화
==============================

3대 카메라(left/center/right) 영상에 YOLO bbox를 그려서 창으로 보여줌.
ground_truth=true/false 무관하게 동작 (카메라 토픽만 사용).

사용법:
  # 시뮬레이터가 돌고 있어야 함 (엔진 on/off 무관)
  cd ~/AIC_Sejong/ws_aic/src/aic
  pixi run python /home/sch24/yolo_viewer.py \\
    --model ../../weight/ais_yolo/weights/best.pt

옵션:
  --conf 0.3       : confidence threshold (기본 0.5)
  --save           : 창 대신 파일로 저장 (~/yolo_view/)
  --single         : 단일 프레임만 캡처 (헤드리스 환경용)
"""
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


CAMERAS = ["left", "center", "right"]
CLASS_COLORS = {
    0: (0, 255, 0),      # sfp_port → 녹색
    1: (255, 0, 255),    # sc_port → 마젠타
}
CLASS_NAMES = {0: "sfp_port", 1: "sc_port"}


class YoloViewer(Node):
    def __init__(self, model_path, conf_thresh):
        super().__init__("yolo_viewer")

        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

        self._latest = {}
        for name in CAMERAS:
            self.create_subscription(
                Image, f"/{name}_camera/image",
                lambda msg, n=name: self._on_image(n, msg),
                10,
            )

        self.get_logger().info(f"YOLO 뷰어 시작 (conf ≥ {conf_thresh})")

    def _on_image(self, name, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self._latest[name] = img

    def ready(self):
        return len(self._latest) == len(CAMERAS)

    def detect_and_draw(self, image):
        """YOLO 추론 후 bbox를 이미지에 그려서 반환."""
        results = self.model(image, verbose=False, conf=self.conf_thresh)
        annotated = image.copy()
        n_detections = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                color = CLASS_COLORS.get(cls, (255, 255, 255))
                label = f"{CLASS_NAMES.get(cls, 'unknown')} {conf:.2f}"

                # bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                # 중심점
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(annotated, (cx, cy), 4, color, -1)
                # 라벨 배경
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(
                    annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1),
                    color, -1)
                # 라벨 텍스트
                cv2.putText(
                    annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
                )
                n_detections += 1

        # 상단에 요약 정보
        cv2.putText(
            annotated, f"Detections: {n_detections}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        return annotated, n_detections

    def get_combined_view(self):
        """3대 카메라 뷰를 하나의 이미지로 합침."""
        if not self.ready():
            return None

        annotated_cams = []
        total_dets = 0
        for name in CAMERAS:
            img = self._latest[name]
            annotated, n = self.detect_and_draw(img)
            # 카메라 이름 표시
            cv2.putText(
                annotated, name.upper(),
                (annotated.shape[1] - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3,
            )
            # 크기 축소
            h, w = annotated.shape[:2]
            scaled = cv2.resize(annotated, (w // 2, h // 2))
            annotated_cams.append(scaled)
            total_dets += n

        # 가로로 이어 붙임
        combined = np.hstack(annotated_cams)
        return combined, total_dets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--save", action="store_true",
                        help="화면 대신 파일로 저장")
    parser.add_argument("--single", action="store_true",
                        help="한 프레임만 캡처 후 종료")
    parser.add_argument("--out", type=str, default="~/yolo_view")
    args = parser.parse_args()

    out_dir = Path(os.path.expanduser(args.out))
    if args.save or args.single:
        out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = YoloViewer(args.model, args.conf)

    print("카메라 데이터 수신 대기...")
    start = time.time()
    while not node.ready() and (time.time() - start) < 15.0:
        rclpy.spin_once(node, timeout_sec=0.2)

    if not node.ready():
        print("ERROR: 카메라 데이터 수신 실패")
        return

    print("뷰어 시작 (q 키로 종료)")

    try:
        frame_idx = 0
        while True:
            rclpy.spin_once(node, timeout_sec=0.05)
            result = node.get_combined_view()
            if result is None:
                continue
            combined, total_dets = result

            if args.save or args.single:
                out_path = out_dir / f"frame_{frame_idx:05d}.jpg"
                cv2.imwrite(str(out_path), combined)
                print(f"[{frame_idx}] 저장: {out_path}  "
                      f"(총 {total_dets}개 검출)")

                if args.single:
                    break
            else:
                cv2.imshow("YOLO Port Detection (left | center | right)",
                           combined)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    out_dir.mkdir(parents=True, exist_ok=True)
                    path = out_dir / f"snapshot_{frame_idx:05d}.jpg"
                    cv2.imwrite(str(path), combined)
                    print(f"스냅샷 저장: {path}")

            frame_idx += 1
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
