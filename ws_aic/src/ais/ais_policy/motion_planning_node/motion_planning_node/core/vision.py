"""YOLO and stereo-based port localization."""

import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from motion_planning_node.core.config import Stage1Config
from motion_planning_node.core.geometry import (
    _project_3d_to_pixel,
)


# ═══════════════════════════════════════════════════════════
#  Vision 모듈 (`YOLO + Stereo)
# ═══════════════════════════════════════════════════════════

class VisionPortEstimator:
    """카메라 영상에서 포트 3D 좌표를 추정.

    YOLO 모델 로드와 추론은 백그라운드 스레드에서 수행한다.
    """

    CAMERAS = [("left", "left_camera/optical"),
               ("center", "center_camera/optical"),
               ("right", "right_camera/optical")]

    CLASS_NAMES = {0: "port_pair", 1: "sc_port", 2: "sfp_tip", 3: "sc_tip"}
    NIC_MOUNT_LOCAL_Y_M = {
        0: -0.1745,
        1: -0.1345,
        2: -0.0945,
        3: -0.0545,
        4: -0.0145,
    }
    SC_PORT_LOCAL_Y_M = {
        0: 0.0295,
        1: 0.0705,
    }
    NIC_MOUNT_CENTER_INDEX = 2
    SC_PORT_CENTER_INDEX = 0
    ROI_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
    TOOL0_TO_TCP_Z = 0.1965
    TOOL0_TO_OPTICAL = {
        "left": (
            [-0.100516584, -0.058032593, -0.008935891],
            [-0.113039947, 0.065265728, -0.495722390, 0.858616135],
        ),
        "center": (
            [-0.000000001, -0.116079183, -0.008937891],
            [-0.130528330, 0.000001827, -0.000000288, 0.991444580],
        ),
        "right": (
            [0.100516583, -0.058032595, -0.008935891],
            [-0.113041775, -0.065262563, 0.495721890, 0.858616424],
        ),
    }

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
        self._load_lock = threading.Lock()
        self._debug_call_count = 0  # 호출 횟수 (파일명 중복 방지)
        self._cache_max_age_sec = float(os.environ.get("AIC_YOLO_CACHE_MAX_AGE_SEC", "0.75"))
        self._async_wait_sec = float(os.environ.get("AIC_YOLO_ASYNC_WAIT_SEC", "2.0"))
        self._async_poll_sec = float(os.environ.get("AIC_YOLO_ASYNC_POLL_SEC", "0.02"))
        self._request_lock = threading.Lock()
        self._request_event = threading.Event()
        self._request: Optional[dict[str, Any]] = None
        self._cache_lock = threading.Lock()
        self._cache: dict[str, Any] = {
            "target_class_id": None,
            "candidates": [],
            "updated_at": 0.0,
            "request_id": 0,
        }
        self._t_tool0_tcp = np.eye(4, dtype=float)
        self._t_tool0_tcp[2, 3] = float(os.environ.get("AIC_TOOL0_TO_TCP_Z", str(self.TOOL0_TO_TCP_Z)))
        self._t_tool0_to_optical = {
            name: self._matrix_from_translation_quat(translation, quat)
            for name, (translation, quat) in self.TOOL0_TO_OPTICAL.items()
        }
        self._request_seq = 0
        threading.Thread(target=self._ensure_loaded, daemon=True).start()
        threading.Thread(target=self._estimate_worker, daemon=True).start()

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
        with self._load_lock:
            if self._loaded:
                return
            if not os.path.isfile(self._model_path):
                if self._logger:
                    self._logger.error(
                        f"YOLO 모델 파일 없음: {self._model_path}\n"
                        "  해결 방법:\n"
                        "  1) AIC_SFP_YOLO_MODEL_PATH 또는 AIC_SC_YOLO_MODEL_PATH 환경 변수로 경로 지정\n"
                        "  2) ws_aic/model/ais_yolo/approach/SFP/weights/best.pt 에 배치\n"
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

    def _estimate_worker(self) -> None:
        while True:
            self._request_event.wait()
            with self._request_lock:
                request = self._request
                self._request = None
                self._request_event.clear()
            if request is None:
                continue

            try:
                candidates = self._estimate_all_sync(
                    request["obs"],
                    request["target_class_id"],
                )
            except Exception as ex:
                candidates = []
                if self._logger:
                    self._logger.warn(f"Vision: 비동기 YOLO 추론 실패: {ex}")
            with self._cache_lock:
                self._cache = {
                    "target_class_id": request["target_class_id"],
                    "candidates": candidates,
                    "updated_at": time.time(),
                    "request_id": request["request_id"],
                }

    def _submit_estimate(self, obs, target_class_id: int) -> int:
        with self._request_lock:
            self._request_seq += 1
            request_id = self._request_seq
            self._request = {
                "obs": obs,
                "target_class_id": target_class_id,
                "request_id": request_id,
            }
            self._request_event.set()
            return request_id

    def _cached_candidates(self, target_class_id: int, min_request_id: int = 0) -> Optional[list]:
        with self._cache_lock:
            cache = dict(self._cache)
        if cache.get("target_class_id") != target_class_id:
            return None
        if int(cache.get("request_id", 0)) < min_request_id:
            return None
        if (time.time() - float(cache.get("updated_at", 0.0))) > self._cache_max_age_sec:
            return None
        return list(cache.get("candidates") or [])

    @staticmethod
    def _image_from_msg(img_msg):
        import cv2
        img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, 3)
        if img_msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    @staticmethod
    def _quat_to_matrix_xyzw(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        norm = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
        if norm < 1e-12:
            return np.eye(3, dtype=float)
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz
        return np.array([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ], dtype=float)

    @classmethod
    def _matrix_from_translation_quat(cls, translation, quat_xyzw) -> np.ndarray:
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = cls._quat_to_matrix_xyzw(*quat_xyzw)
        matrix[:3, 3] = np.asarray(translation, dtype=float)
        return matrix

    @classmethod
    def _matrix_from_pose(cls, pose) -> np.ndarray:
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = cls._quat_to_matrix_xyzw(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return matrix

    def _base_to_camera_optical_matrix(self, obs, camera_name: str) -> Optional[np.ndarray]:
        try:
            t_base_tcp = self._matrix_from_pose(obs.controller_state.tcp_pose)
            t_base_tool0 = t_base_tcp @ np.linalg.inv(self._t_tool0_tcp)
            t_base_optical = t_base_tool0 @ self._t_tool0_to_optical[camera_name]
            return np.linalg.inv(t_base_optical)
        except Exception as ex:
            if self._logger:
                self._logger.warn(
                    f"Vision: {camera_name} camera extrinsic 계산 실패: {ex}"
                )
            return None

    def _detect(self, image: np.ndarray, cam_name: str = "") -> list:
        import cv2

        # conf=0.01로 raw 결과 전부 받아서 로깅 후, conf_thresh 적용
        results = self._model(image, verbose=False, conf=0.01)
        raw_dets = []
        for r in results:
            keypoints_xy = None
            if r.keypoints is not None and r.keypoints.xy is not None:
                keypoints_xy = r.keypoints.xy.detach().cpu().numpy()

            for box_idx, box in enumerate(r.boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                xyxy = (int(x1), int(y1), int(x2), int(y2))

                if keypoints_xy is not None and box_idx < len(keypoints_xy):
                    kpts = np.asarray(keypoints_xy[box_idx], dtype=np.float64)
                    if len(kpts) >= 8:
                        for port_index, start in enumerate((0, 4)):
                            group = kpts[start:start + 4]
                            if not np.all(np.isfinite(group)):
                                continue
                            center = np.mean(group, axis=0)
                            raw_dets.append({
                                "class_id": cls,
                                "class_name": f"sfp_port_{port_index}",
                                "conf": conf,
                                "u": float(center[0]),
                                "v": float(center[1]),
                                "xyxy": xyxy,
                                "point_name": f"sfp_port_{port_index}",
                                "port_index": port_index,
                                "keypoints": group,
                            })
                        continue

                    if len(kpts) >= 4:
                        group = kpts[:4]
                        if np.all(np.isfinite(group)):
                            center = np.mean(group, axis=0)
                            raw_dets.append({
                                "class_id": cls,
                                "class_name": "sc_port",
                                "conf": conf,
                                "u": float(center[0]),
                                "v": float(center[1]),
                                "xyxy": xyxy,
                                "point_name": "sc_port",
                                "port_index": None,
                                "keypoints": group,
                            })
                            continue

                raw_dets.append({
                    "class_id": cls,
                    "class_name": self.CLASS_NAMES.get(cls, "unknown"),
                    "conf": conf,
                    "u": float((x1 + x2) / 2),
                    "v": float((y1 + y2) / 2),
                    "xyxy": xyxy,
                    "point_name": "bbox_center",
                    "port_index": None,
                })

        # 진단 로그
        if self._logger:
            if raw_dets:
                for d in raw_dets:
                    flag = "✓" if d["conf"] >= self._conf_thresh else "✗(low conf)"
                    self._logger.info(
                        f"  [{cam_name}] {flag} {d['class_name']} "
                        f"point={d.get('point_name', 'bbox_center')} "
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
                if d.get("port_index") == 1:
                    color = (255, 0, 255)
                passed = d["conf"] >= self._conf_thresh

                # 통과한 것: 실선 두껍게, 실패한 것: 점선 얇게
                thickness = 3 if passed else 1
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, thickness)
                cv2.circle(debug_img, (int(d["u"]), int(d["v"])), 5, color, -1)

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

    def estimate(self, obs, target_class_id: int,
                 port_hint: Optional[str] = None,
                 target_module_name: Optional[str] = None) -> Optional[np.ndarray]:
        """포트 3D 좌표 추정 (호환성 유지).

        단일 best 결과를 반환. 세밀한 선택이 필요하면 estimate_all() 사용.
        """
        candidates = self.estimate_all(obs, target_class_id)
        if not candidates:
            return None

        # Task hint가 주어지면 여러 카드/포트 후보 중 목표 rail ROI를 선택한다.
        if port_hint is not None or target_module_name is not None:
            chosen = self.select_by_task_hint(
                candidates,
                port_name=port_hint or "",
                target_module_name=target_module_name or "",
            )
            if chosen is not None:
                return chosen

        # 기본: 보드 중심에 가장 가까운 것 (첫 번째)
        return candidates[0]["pos"]

    def estimate_all(
        self, obs, target_class_id: int
    ) -> list:
        """가능한 모든 유효 후보 반환.

        무거운 YOLO 추론은 워커 스레드에 제출하고, 최신 캐시가 갱신될 때까지
        짧게 폴링한다. 캐시가 아직 없으면 빈 리스트를 반환한다.
        """
        if obs is None:
            return []

        request_id = self._submit_estimate(obs, target_class_id)
        deadline = time.time() + max(0.0, self._async_wait_sec)
        while time.time() < deadline:
            cached = self._cached_candidates(target_class_id, min_request_id=request_id)
            if cached is not None:
                return cached
            time.sleep(max(0.001, self._async_poll_sec))

        cached = self._cached_candidates(target_class_id)
        if cached is not None:
            return cached
        if self._logger:
            self._logger.warn(
                f"Vision: 비동기 YOLO 결과 대기 시간 초과 "
                f"(target_class_id={target_class_id})"
            )
        return []

    def _estimate_all_sync(
        self, obs, target_class_id: int
    ) -> list:
        """가능한 모든 유효 후보 반환. score 낮은 순(= 보드 중심 가까운 순) 정렬.

        Returns:
            [{"pos": np.ndarray(3), "score": float, "conf_sum": float}, ...]
        """
        self._ensure_loaded()
        if not self._loaded:
            return []

        # 1. 카메라별 이미지 + 내부 파라미터 + 고정 extrinsic
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
        T_base_to = {}
        for name, _ in self.CAMERAS:
            t_base_to_camera = self._base_to_camera_optical_matrix(obs, name) # t_base_optical
            if t_base_to_camera is None:
                return []
            T_base_to[name] = t_base_to_camera

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
        T_base_to = {n: T_base_to[n] for n, _ in self.CAMERAS
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
                if da.get("point_name") != db.get("point_name"):
                    continue
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
                    same_point_dets = [
                        d for d in detections[third_cam]
                        if d.get("point_name") == da.get("point_name")
                    ]
                    if not same_point_dets:
                        continue
                    # 제3카메라의 검출들 중 가장 가까운 것과의 거리
                    min_px_dist = min(
                        np.hypot(d["u"] - u_proj, d["v"] - v_proj)
                        for d in same_point_dets
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
                    "point_name": da.get("point_name"),
                    "port_index": da.get("port_index"),
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
                f"{unique[0]['pos'][1]:+.3f}, {unique[0]['pos'][2]:+.3f}), "
                f"point={unique[0].get('point_name')})"
            )

        return unique

    @staticmethod
    def _extract_named_index(text: str, prefix: str) -> Optional[int]:
        match = re.search(rf"{re.escape(prefix)}_(\d+)", text or "")
        if match is None:
            return None
        return int(match.group(1))

    def _roi_axis(self) -> tuple[str, int, bool]:
        axis_name = os.environ.get("AIC_VISION_ROI_AXIS", "y").strip().lower()
        if axis_name not in self.ROI_AXIS_TO_INDEX:
            if self._logger:
                self._logger.warn(
                    f"Vision ROI: invalid AIC_VISION_ROI_AXIS={axis_name!r}, using 'y'"
                )
            axis_name = "y"
        reverse = os.environ.get("AIC_VISION_ROI_REVERSE", "false").strip().lower()
        return axis_name, self.ROI_AXIS_TO_INDEX[axis_name], reverse in {
            "1", "true", "yes", "on",
        }

    def _roi_y_threshold_m(self) -> float:
        return float(os.environ.get("AIC_VISION_ROI_Y_THRESHOLD_M", "0.018"))

    def _rail_base_y_centers(
        self,
        local_y_by_index: dict[int, float],
        env_key: str,
        anchor_index: int,
    ) -> dict[int, float]:
        raw = os.environ.get(env_key, "").strip()
        if raw:
            values = [float(v.strip()) for v in raw.split(",") if v.strip()]
            ordered_indices = sorted(local_y_by_index)
            if len(values) == len(ordered_indices):
                return dict(zip(ordered_indices, values))
            if self._logger:
                self._logger.warn(
                    f"Vision ROI: {env_key} needs {len(ordered_indices)} comma-separated "
                    f"values, got {len(values)}; using derived defaults"
                )

        board_center_y = float(os.environ.get(
            "AIC_VISION_BOARD_CENTER_BASE_Y",
            str(Stage1Config.BOARD_CENTER[1]),
        ))
        _, _, reverse = self._roi_axis()
        sign = -1.0 if reverse else 1.0
        anchor_local_y = local_y_by_index[anchor_index]
        return {
            idx: board_center_y + sign * (local_y - anchor_local_y)
            for idx, local_y in local_y_by_index.items()
        }

    def _select_by_base_y_threshold(
        self,
        candidates: list,
        target_index: Optional[int],
        local_y_by_index: dict[int, float],
        label: str,
        env_key: str,
        anchor_index: int,
    ) -> Optional[dict]:
        if not candidates or target_index is None or target_index not in local_y_by_index:
            return None

        centers = self._rail_base_y_centers(local_y_by_index, env_key, anchor_index)
        target_y = centers[target_index]
        threshold = self._roi_y_threshold_m()
        matches = [
            (abs(float(c["pos"][1]) - target_y), c)
            for c in candidates
            if abs(float(c["pos"][1]) - target_y) <= threshold
        ]
        if not matches:
            if self._logger:
                candidate_y = ", ".join(f"{float(c['pos'][1]):+.4f}" for c in candidates)
                self._logger.warn(
                    "Vision ROI: no base_y candidate inside target rail threshold "
                    f"(target={label}_{target_index}, target_y={target_y:+.4f}, "
                    f"threshold={threshold:.4f}, candidate_y=[{candidate_y}])"
                )
            return None

        matches.sort(key=lambda item: (item[0], item[1].get("score", 0.0)))
        dy, chosen = matches[0]
        if self._logger:
            self._logger.info(
                "Vision ROI select: "
                f"target={label}_{target_index}, "
                f"target_y={target_y:+.4f}, dy={dy:.4f}, "
                f"threshold={threshold:.4f}, "
                f"chosen=({chosen['pos'][0]:+.4f}, "
                f"{chosen['pos'][1]:+.4f}, {chosen['pos'][2]:+.4f}), "
                f"point={chosen.get('point_name')}"
            )
        return chosen

    def select_by_task_hint(
        self,
        candidates: list,
        port_name: str = "",
        target_module_name: str = "",
    ) -> Optional[np.ndarray]:
        """Task hint로 multi-card/multi-port 상황의 올바른 후보를 선택.

        triangulation 결과는 base_link 좌표다. task board local 좌표를 직접 비교하지
        않고, local y 배치 순서와 base_link 후보들의 ROI axis 순서를 맞춰 선택한다.
        """
        if not candidates:
            return None
        if len(candidates) == 1 and not target_module_name:
            return candidates[0]["pos"]

        sfp_port_index = self._extract_named_index(port_name, "sfp_port")
        nic_mount_index = self._extract_named_index(target_module_name, "nic_card_mount")

        port_candidates = candidates
        if sfp_port_index is not None:
            same_port = [
                c for c in candidates
                if c.get("port_index") == sfp_port_index
            ]
            if same_port:
                port_candidates = same_port

        if nic_mount_index is not None:
            chosen = self._select_by_base_y_threshold(
                port_candidates,
                nic_mount_index,
                self.NIC_MOUNT_LOCAL_Y_M,
                "nic_card_mount",
                "AIC_VISION_NIC_RAIL_BASE_Y",
                self.NIC_MOUNT_CENTER_INDEX,
            )
            if chosen is not None:
                return chosen["pos"]
            return None

        sc_port_index = self._extract_named_index(port_name, "sc_port")
        if sc_port_index is None:
            sc_port_index = self._extract_named_index(target_module_name, "sc_port")
        if sc_port_index is not None:
            chosen = self._select_by_base_y_threshold(
                port_candidates,
                sc_port_index,
                self.SC_PORT_LOCAL_Y_M,
                "sc_port",
                "AIC_VISION_SC_PORT_BASE_Y",
                self.SC_PORT_CENTER_INDEX,
            )
            if chosen is not None:
                return chosen["pos"]
            return None

        return self.select_by_port_name(candidates, port_name)

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
            wanted = None
            if port_name.endswith("_0") or port_name.endswith("0_link"):
                wanted = 0
            elif port_name.endswith("_1") or port_name.endswith("1_link"):
                wanted = 1
            if wanted is not None:
                for candidate in candidates:
                    if candidate.get("port_index") == wanted:
                        return candidate["pos"]

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
