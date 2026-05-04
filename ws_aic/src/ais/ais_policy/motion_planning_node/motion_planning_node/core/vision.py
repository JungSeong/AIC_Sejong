"""YOLO and stereo-based port localization."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
from rclpy.time import Time
from tf2_ros import TransformException

from motion_planning_node.core.config import Stage1Config
from motion_planning_node.core.geometry import (
    _project_3d_to_pixel,
    transform_to_matrix,
)


# ═══════════════════════════════════════════════════════════
#  Vision 모듈 (`YOLO + Stereo)
# ═══════════════════════════════════════════════════════════

class VisionPortEstimator:
    """카메라 영상에서 포트 3D 좌표를 추정.

    첫 호출 시에만 YOLO 모델 로드 (지연 초기화).
    """

    CAMERAS = [("left", "left_camera/optical"),
               ("center", "center_camera/optical"),
               ("right", "right_camera/optical")]

    CLASS_NAMES = {0: "sfp_port", 1: "sc_port"}

    # 디버그 이미지 저장 디렉토리 (None이면 저장 안 함)
    # Docker: AIC_DEBUG_SAVE_DIR=/debug/yolo_detections (docker-compose에서 volume mount)
    # Native: ~/aic_debug/yolo_detections (기본값)
    DEBUG_SAVE_DIR: str = os.environ.get(
        "AIC_DEBUG_SAVE_DIR",
        str(Path.home() / "aic_debug" / "yolo_detections"),
    )

    def __init__(self, model_path: str, conf_thresh: float = 0.5, logger=None):
        self._model_path = model_path
        self._conf_thresh = conf_thresh
        self._logger = logger
        self._model = None
        self._loaded = False
        self._debug_call_count = 0  # 호출 횟수 (파일명 중복 방지)

        # 디버그 디렉토리 생성
        if self.DEBUG_SAVE_DIR:
            try:
                os.makedirs(self.DEBUG_SAVE_DIR, exist_ok=True)
                if logger:
                    logger.info(f"[Vision Debug] 저장 경로: {self.DEBUG_SAVE_DIR}")
            except Exception as e:
                if logger:
                    logger.error(f"[Vision Debug] 디렉토리 생성 실패: {e}")
                # 저장 실패해도 동작은 계속 — 디버그 비활성화
                self.__class__.DEBUG_SAVE_DIR = None

    def _ensure_loaded(self):
        if self._loaded:
            return
        if not os.path.isfile(self._model_path):
            if self._logger:
                self._logger.error(
                    f"YOLO 모델 파일 없음: {self._model_path}\n"
                    "  해결 방법:\n"
                    "  1) AIC_YOLO_MODEL_PATH 환경 변수로 경로 지정\n"
                    "  2) ws_aic/weight/ais_yolo/weights/best.pt 에 배치\n"
                    "  3) YOLO 학습 스크립트의 --output 경로 확인"
                )
            return
        try:
            from ultralytics import YOLO
            import cv2  # noqa: F401
        except ImportError as e:
            if self._logger:
                self._logger.error(f"Vision 의존성 없음: {e}")
            return
        try:
            self._model = YOLO(self._model_path)
            self._loaded = True
            if self._logger:
                self._logger.info(f"YOLO 모델 로드: {self._model_path}")
        except Exception as e:
            if self._logger:
                self._logger.error(f"YOLO 로드 실패: {e}")

    @staticmethod
    def _image_from_msg(img_msg):
        import cv2
        img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, 3)
        if img_msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def _detect(self, image: np.ndarray, cam_name: str = "") -> list:
        import cv2

        # conf=0.01로 raw 결과 전부 받아서 로깅 후, conf_thresh 적용
        results = self._model(image, verbose=False, conf=0.01)
        raw_dets = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                raw_dets.append({
                    "class_id": cls,
                    "class_name": self.CLASS_NAMES.get(cls, "unknown"),
                    "conf": conf,
                    "u": float((x1 + x2) / 2),
                    "v": float((y1 + y2) / 2),
                    "xyxy": (int(x1), int(y1), int(x2), int(y2)),
                })

        # 진단 로그
        if self._logger:
            if raw_dets:
                for d in raw_dets:
                    flag = "✓" if d["conf"] >= self._conf_thresh else "✗(low conf)"
                    self._logger.info(
                        f"  [{cam_name}] {flag} {d['class_name']} "
                        f"conf={d['conf']:.3f} (thresh={self._conf_thresh}) "
                        f"uv=({d['u']:.0f},{d['v']:.0f})"
                    )
            else:
                self._logger.warn(f"  [{cam_name}] 검출 결과 0개 (모델이 아무것도 못 찾음)")

        # ── 디버그 이미지 저장 ────────────────────────────────
        if self.DEBUG_SAVE_DIR:
            debug_img = image.copy()
            # 클래스별 색상: sfp_port=초록, sc_port=파랑, unknown=빨강
            COLOR = {0: (0, 255, 0), 1: (255, 100, 0), -1: (0, 0, 255)}

            for d in raw_dets:
                x1, y1, x2, y2 = d["xyxy"]
                color = COLOR.get(d["class_id"], COLOR[-1])
                passed = d["conf"] >= self._conf_thresh

                # 통과한 것: 실선 두껍게, 실패한 것: 점선 얇게
                thickness = 3 if passed else 1
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, thickness)

                # 라벨
                label = f"{d['class_name']} {d['conf']:.2f}"
                label += " ✓" if passed else " ✗"
                cv2.putText(
                    debug_img, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )

            # threshold 선 표시 (이미지 상단에 텍스트)
            cv2.putText(
                debug_img,
                f"thresh={self._conf_thresh}  cam={cam_name}  dets={len(raw_dets)}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

            fname = (
                f"{self.DEBUG_SAVE_DIR}/"
                f"{self._debug_call_count:04d}_{cam_name}.jpg"
            )
            cv2.imwrite(fname, debug_img)

        self._debug_call_count += 1

        # conf_thresh 필터 적용
        return [d for d in raw_dets if d["conf"] >= self._conf_thresh]

    @staticmethod
    def _triangulate(u_a, v_a, K_a, T_base_to_a,
                     u_b, v_b, K_b, T_base_to_b) -> np.ndarray:
        import cv2
        P_a = K_a @ T_base_to_a[:3, :]
        P_b = K_b @ T_base_to_b[:3, :]
        pts_4d = cv2.triangulatePoints(
            P_a, P_b,
            np.array([[u_a], [v_a]], dtype=np.float64),
            np.array([[u_b], [v_b]], dtype=np.float64),
        )
        return (pts_4d[:3] / pts_4d[3]).flatten()

    def estimate(self, obs, tf_buffer, target_class_id: int,
                 port_hint: Optional[str] = None) -> Optional[np.ndarray]:
        """포트 3D 좌표 추정 (호환성 유지).

        단일 best 결과를 반환. 세밀한 선택이 필요하면 estimate_all() 사용.
        """
        candidates = self.estimate_all(obs, tf_buffer, target_class_id)
        if not candidates:
            return None

        # port_hint가 주어지면 그에 맞는 걸 선택
        if port_hint is not None:
            chosen = self.select_by_port_name(candidates, port_hint)
            if chosen is not None:
                return chosen

        # 기본: 보드 중심에 가장 가까운 것 (첫 번째)
        return candidates[0]["pos"]

    def estimate_all(
        self, obs, tf_buffer, target_class_id: int
    ) -> list:
        """가능한 모든 유효 후보 반환. score 낮은 순(= 보드 중심 가까운 순) 정렬.

        Returns:
            [{"pos": np.ndarray(3), "score": float, "conf_sum": float}, ...]
        """
        self._ensure_loaded()
        if not self._loaded:
            return []

        # 1. 카메라별 이미지 + 내부 파라미터 + 외부 TF
        images = {
            "left": self._image_from_msg(obs.left_image),
            "center": self._image_from_msg(obs.center_image),
            "right": self._image_from_msg(obs.right_image),
        }
        cam_infos = {
            "left": obs.left_camera_info,
            "center": obs.center_camera_info,
            "right": obs.right_camera_info,
        }
        cam_T_in_base = {}
        for name, frame in self.CAMERAS:
            try:
                tf = tf_buffer.lookup_transform("base_link", frame, Time())
                cam_T_in_base[name] = transform_to_matrix(tf.transform)
            except TransformException:
                if self._logger:
                    self._logger.warn(f"Vision: 카메라 {name} TF 없음")
                return []

        # 2. 각 카메라에서 검출 (타겟 클래스만)
        if self._logger:
            self._logger.info(f"Vision: 검출 시작 (target_class_id={target_class_id})")
        detections = {}
        for name in ["left", "center", "right"]:
            dets = self._detect(images[name], cam_name=name)
            dets = [d for d in dets if d["class_id"] == target_class_id]
            detections[name] = dets

        cams_with_dets = [n for n, d in detections.items() if d]
        if len(cams_with_dets) < 2:
            if self._logger:
                self._logger.warn(
                    f"Vision: 2대 이상의 카메라에서 검출 실패 "
                    f"({cams_with_dets})"
                )
            return []

        # 3. 가능한 카메라 쌍 모두 사용 (3-view consistency 검증용)
        available_pairs = []
        for a, b in [("left", "center"), ("center", "right"),
                     ("left", "right")]:
            if a in cams_with_dets and b in cams_with_dets:
                available_pairs.append((a, b))

        if not available_pairs:
            return []

        # 카메라 정보 사전 캐시
        K = {n: np.array(cam_infos[n].k).reshape(3, 3) for n, _ in self.CAMERAS
             if n in cams_with_dets}
        T_base_to = {n: np.linalg.inv(cam_T_in_base[n]) for n, _ in self.CAMERAS
                     if n in cams_with_dets}

        # 4. 각 카메라 쌍으로 후보 삼각측량
        board_center = np.array(Stage1Config.BOARD_CENTER)

        # 각 detection (왼쪽 카메라 기준)마다 3D 추정
        #   → 다른 카메라들에서 "재투영" 해서 실제 검출된 bbox와 매칭되는지 확인
        # 이게 3-view consistency: 같은 포트의 진짜 3D라면 모든 카메라에서 일치
        PIXEL_MATCH_THRESH = 30.0  # 재투영된 점이 실제 검출과 이만큼 이내여야 OK

        candidates = []
        # 주 카메라 쌍은 베이스라인 큰 것 (left-right 우선)
        main_pair = available_pairs[-1]
        cam_a, cam_b = main_pair

        for da in detections[cam_a]:
            for db in detections[cam_b]:
                port_3d = self._triangulate(
                    da["u"], da["v"], K[cam_a], T_base_to[cam_a],
                    db["u"], db["v"], K[cam_b], T_base_to[cam_b],
                )
                dist = float(np.linalg.norm(port_3d - board_center))
                if dist > Stage1Config.BOARD_RADIUS:
                    continue
                if not (Stage1Config.Z_RANGE[0] <= port_3d[2]
                        <= Stage1Config.Z_RANGE[1]):
                    continue

                # 3-view consistency: 제3의 카메라(center)에 재투영했을 때
                # 그 카메라 검출과 실제로 일치하는가?
                third_cam = None
                for name in ["center", "left", "right"]:
                    if name not in (cam_a, cam_b) and name in detections:
                        third_cam = name
                        break

                consistent = True
                if third_cam and detections[third_cam]:
                    u_proj, v_proj = _project_3d_to_pixel(
                        port_3d, K[third_cam], T_base_to[third_cam]
                    )
                    # 제3카메라의 검출들 중 가장 가까운 것과의 거리
                    min_px_dist = min(
                        np.hypot(d["u"] - u_proj, d["v"] - v_proj)
                        for d in detections[third_cam]
                    )
                    if min_px_dist > PIXEL_MATCH_THRESH:
                        consistent = False

                if not consistent:
                    continue

                conf_sum = da["conf"] + db["conf"]
                score = dist - 0.1 * conf_sum
                candidates.append({
                    "pos": port_3d,
                    "score": score,
                    "conf_sum": conf_sum,
                })

        # score 낮은 순 정렬 (보드 중심 가깝고 conf 높을수록 앞)
        candidates.sort(key=lambda c: c["score"])

        # 중복 제거 (같은 3D 좌표로 수렴한 매칭들 — 1cm 이내면 같은 포트로 간주)
        unique = []
        for c in candidates:
            is_dup = False
            for u in unique:
                if np.linalg.norm(c["pos"] - u["pos"]) < 0.01:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(c)

        if self._logger and unique:
            self._logger.info(
                f"Vision: {len(unique)}개 후보 포트 3D 추정 "
                f"(best=({unique[0]['pos'][0]:+.3f}, "
                f"{unique[0]['pos'][1]:+.3f}, {unique[0]['pos'][2]:+.3f}))"
            )

        return unique

    @staticmethod
    def select_by_port_name(
        candidates: list, port_name: str
    ) -> Optional[np.ndarray]:
        """Task의 port_name으로 올바른 후보 선택.

        NIC 카드 SFP 포트의 경우:
          sfp_port_0 → x 좌표가 더 큰 쪽 (NIC 카드의 오른쪽)
          sfp_port_1 → x 좌표가 더 작은 쪽 (NIC 카드의 왼쪽)

        하나만 후보면 그걸 반환.
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]["pos"]

        # SFP 포트 인덱스 기반 구분
        if "sfp_port" in port_name:
            # 후보를 x 좌표 기준으로 정렬
            sorted_by_x = sorted(candidates, key=lambda c: c["pos"][0])
            # port_0: x가 큰 쪽 (리스트 끝)
            # port_1: x가 작은 쪽 (리스트 시작)
            if port_name.endswith("_0") or port_name.endswith("0_link"):
                return sorted_by_x[-1]["pos"]
            elif port_name.endswith("_1") or port_name.endswith("1_link"):
                return sorted_by_x[0]["pos"]

        # SC 포트나 기타는 보드 중심에 가장 가까운 것 (score 최소)
        return candidates[0]["pos"]
