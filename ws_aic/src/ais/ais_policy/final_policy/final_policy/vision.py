"""YOLO 검출 결과를 여러 카메라 기하로 삼각측량해 포트 위치를 추정하는 모듈."""

from __future__ import annotations

import os
import re
import threading
import time
import numpy as np

from final_policy.config import FinalPolicyConfig
from final_policy.geometry import project_3d_to_pixel
from final_policy.model_store import format_model_log
from pathlib import Path
from typing import Any, Optional

def _resolve_project_root() -> Path:
    """현재 파일 위치에서 AIC_Sejong 프로젝트 루트를 역추적한다."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "ws_aic" / "src").is_dir():
            return parent
    return Path(__file__).resolve().parents[6]

DEFAULT_DEBUG_ROOT_DIR = _resolve_project_root() / "debug"

class VisionPortEstimator:
    """
    비동기 YOLO 추론 및 다중 카메라 삼각측량으로 base_link 기준 포트 좌표를 추정하는 함수
    """
    CAMERAS = [
        ("left", "left_camera/optical"),
        ("center", "center_camera/optical"),
        ("right", "right_camera/optical"),
    ]
    CLASS_NAMES = {0: "port_pair", 1: "sc_port", 2: "sfp_tip", 3: "sc_tip"}
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
    DEBUG_SAVE_DIR: str = os.environ.get(
        "AIC_DEBUG_SAVE_DIR",
        str(DEFAULT_DEBUG_ROOT_DIR),
    )

    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.8,
        logger=None,
        debug_save_enabled: bool = True,
        auto_start: bool = True,
    ):
        """모델 경로, 디버그 저장 경로, 카메라 외부 파라미터와 워커 상태를 초기화한다."""
        self._model_path = model_path
        self._conf_thresh = float(conf_thresh)
        self._logger = logger
        self._model = None
        self._loaded = False
        self._load_lock = threading.Lock()
        self._debug_save_enabled = bool(debug_save_enabled)
        self._debug_call_count = 0
        self._debug_task_label = "task_unknown"
        self._debug_root_dir = Path(self.DEBUG_SAVE_DIR) if self.DEBUG_SAVE_DIR else None
        self._debug_save_dir = (
            self._debug_root_dir / "detection"
            if self._debug_root_dir is not None
            else None
        )
        self._cache_max_age_sec = float(os.environ.get("AIC_YOLO_CACHE_MAX_AGE_SEC", "0.75"))
        self._async_wait_sec = float(os.environ.get("AIC_YOLO_ASYNC_WAIT_SEC", "10.0"))
        self._async_poll_sec = float(os.environ.get("AIC_YOLO_ASYNC_POLL_SEC", "0.02"))
        self._yolo_device = os.environ.get("AIC_YOLO_DEVICE", "").strip() or None
        self._request_lock = threading.Lock()
        self._request_event = threading.Event()
        self._request: Optional[dict[str, Any]] = None
        self._worker_stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._cache_lock = threading.Lock()
        self._cache: dict[str, Any] = {
            "target_class_id": None,
            "candidates": [],
            "updated_at": 0.0,
            "request_id": 0,
        }
        self._t_tool0_tcp = np.eye(4, dtype=float)
        self._t_tool0_tcp[2, 3] = float(
            os.environ.get("AIC_TOOL0_TO_TCP_Z", str(self.TOOL0_TO_TCP_Z))
        )
        self._t_tool0_to_optical = {
            name: self._matrix_from_translation_quat(translation, quat)
            for name, (translation, quat) in self.TOOL0_TO_OPTICAL.items()
        }
        self._request_seq = 0

        if self._debug_root_dir is not None:
            try:
                os.makedirs(self._debug_root_dir, exist_ok=True)
                os.makedirs(self._debug_save_dir, exist_ok=True)
                for cam_name, _ in self.CAMERAS:
                    os.makedirs(self._debug_save_dir / cam_name, exist_ok=True)
                if logger:
                    logger.info(f"[Vision Debug] root: {self._debug_root_dir}")
                    logger.info(f"[Vision Debug] detection dir: {self._debug_save_dir}")
            except Exception as exc:
                if logger:
                    logger.error(f"[Vision Debug] failed to create directory: {exc}")
                self.__class__.DEBUG_SAVE_DIR = None
                self._debug_root_dir = None
                self._debug_save_dir = None

        if auto_start:
            self.start_detection()

    def set_debug_save_enabled(self, enabled: bool, reset_counts: bool = False) -> None:
        """YOLO 검출 디버그 이미지 저장 여부를 런타임에 켜고 끈다."""
        self._debug_save_enabled = bool(enabled)
        if reset_counts:
            self._debug_call_count = 0
        if self._logger:
            state = "enabled" if self._debug_save_enabled else "disabled"
            self._logger.info(f"[Vision Debug] image saving {state}")

    @staticmethod
    def _sanitize_debug_token(value: Any) -> str:
        """파일명에 안전하게 쓸 수 있도록 task/debug 문자열을 정리한다."""
        token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
        token = token.strip("._-")
        return token[:80] if token else ""

    def set_debug_task_context(
        self,
        *,
        target_module_name: str = "",
        port_name: str = "",
        plug_name: str = "",
        cable_name: str = "",
        port_type: str = "",
    ) -> None:
        """디버그 이미지 파일명에 들어갈 task 라벨을 task 정보에서 만든다."""
        parts = []
        module = str(target_module_name or "")
        port = str(port_name or "")

        nic_match = re.search(r"nic_card_mount_(\d+)", module)
        sc_module_match = re.search(r"sc_port_(\d+)", module)
        sfp_port_match = re.search(r"sfp_port_(\d+)", port)
        sc_port_match = re.search(r"sc_port_(\d+)", port)

        if nic_match is not None:
            parts.append(f"m{nic_match.group(1)}")
        elif sc_module_match is not None:
            parts.append(f"sc{sc_module_match.group(1)}")

        if sfp_port_match is not None:
            parts.append(f"sfp{sfp_port_match.group(1)}")
        elif sc_port_match is not None and sc_module_match is None:
            parts.append(f"sc{sc_port_match.group(1)}")
        elif "sc_port_base" in port and not parts:
            parts.append("sc")

        if not parts:
            fallback_parts = [
                self._sanitize_debug_token(value)
                for value in (target_module_name, port_name, plug_name, cable_name, port_type)
            ]
            parts = [part for part in fallback_parts if part]
        self._debug_task_label = "_".join(parts)[:80] if parts else "task_unknown"

    @property
    def debug_task_label(self) -> str:
        """현재 디버그 이미지 파일명에 쓰는 task 라벨을 반환한다."""
        return self._debug_task_label

    def _worker_is_alive(self) -> bool:
        """YOLO 추론 워커 스레드가 살아있는지 확인한다."""
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def start_detection(
        self,
        *,
        enable_debug_save: Optional[bool] = None,
        reset_counts: bool = False,
        reset_cache: bool = False,
    ) -> None:
        """비동기 YOLO 추론 워커를 시작하고 필요하면 디버그 저장 상태를 갱신한다."""
        if enable_debug_save is not None:
            self.set_debug_save_enabled(enable_debug_save, reset_counts=reset_counts)
        if reset_cache:
            self.clear_cache()

        if self._worker_is_alive():
            if self._worker_stop_event.is_set():
                self._worker_thread.join(timeout=0.1)
            if self._worker_is_alive():
                return

        if self._worker_stop_event.is_set():
            self._worker_stop_event.clear()

        if self._worker_is_alive():
            return

        self._worker_thread = threading.Thread(target=self._estimate_worker, daemon=True)
        self._worker_thread.start()
        if self._logger:
            self._logger.info(
                f"Vision: YOLO detection worker started (conf>={self._conf_thresh:.2f})"
            )

    def load_model(self) -> bool:
        """YOLO 모델을 동기적으로 미리 로드하고 성공 여부를 반환한다."""
        self._ensure_loaded()
        return self._loaded

    def stop_detection(self) -> None:
        """비동기 YOLO 추론 워커를 멈추고 대기 중인 요청을 비운다."""
        if not self._worker_is_alive():
            return

        self._worker_stop_event.set()
        self._request_event.set()
        if threading.current_thread() is not self._worker_thread:
            self._worker_thread.join()

        with self._request_lock:
            self._request = None
            self._request_event.clear()
        if self._logger:
            self._logger.info("Vision: YOLO detection worker stopped")

    def clear_cache(self) -> None:
        """이전 task의 비동기 추정 결과가 섞이지 않도록 후보 캐시를 비운다."""
        with self._cache_lock:
            self._cache = {
                "target_class_id": None,
                "candidates": [],
                "updated_at": 0.0,
                "request_id": 0,
            }

    def _ensure_loaded(self):
        """YOLO 모델을 최초 사용 시점에 로드한다."""
        with self._load_lock:
            if self._loaded:
                return
            if not os.path.isfile(self._model_path):
                if self._logger:
                    self._logger.error(
                        format_model_log(
                            f"YOLO model file not found: {self._model_path}\n"
                            "Set AIC_SFP_YOLO_MODEL_PATH or AIC_SC_YOLO_MODEL_PATH."
                        )
                    )
                return
            try:
                from ultralytics import YOLO
                import cv2  # noqa: F401
            except ImportError as exc:
                if self._logger:
                    self._logger.error(f"Vision dependency missing: {exc}")
                return
            try:
                self._model = YOLO(self._model_path)
                self._loaded = True
                if self._logger:
                    self._logger.info(
                        format_model_log(f"YOLO model loaded: {self._model_path}")
                    )
                    if self._yolo_device:
                        self._logger.info(
                            format_model_log(
                                f"YOLO device override: {self._yolo_device}"
                            )
                        )
            except Exception as exc:
                if self._logger:
                    self._logger.error(format_model_log(f"YOLO load failed: {exc}"))

    def _estimate_worker(self) -> None:
        """백그라운드에서 최신 요청 하나를 처리하고 결과 후보를 캐시에 저장한다."""
        while not self._worker_stop_event.is_set():
            if not self._request_event.wait(0.1):
                continue
            if self._worker_stop_event.is_set():
                break
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
                    target_port_index=request.get("target_port_index"),
                )
            except Exception as exc:
                candidates = []
                if self._logger:
                    self._logger.warn(f"Vision: async YOLO inference failed: {exc}")
            with self._cache_lock:
                self._cache = {
                    "target_class_id": request["target_class_id"],
                    "candidates": candidates,
                    "updated_at": time.time(),
                    "request_id": request["request_id"],
                }

    def _submit_estimate(
        self,
        obs,
        target_class_id: int,
        target_port_index: Optional[int] = None,
    ) -> int:
        """워커에 새 추정 요청을 넣고 요청 id를 반환한다."""
        if not self._worker_is_alive():
            self.start_detection()
        with self._request_lock:
            self._request_seq += 1
            request_id = self._request_seq
            self._request = {
                "obs": obs,
                "target_class_id": target_class_id,
                "target_port_index": target_port_index,
                "request_id": request_id,
            }
            self._request_event.set()
            return request_id

    def request_estimate(
        self,
        obs,
        target_class_id: int,
        port_hint: Optional[str] = None,
    ) -> int:
        """비동기 포트 추정 요청을 워커에 넘기고 즉시 반환한다."""
        target_port_index = self._extract_named_index(port_hint or "", "sfp_port")
        return self._submit_estimate(
            obs,
            target_class_id,
            target_port_index=target_port_index,
        )

    def _cached_candidates(
        self,
        target_class_id: int,
        min_request_id: int = 0,
    ) -> Optional[list]:
        """target class와 request id, age 조건을 만족하는 캐시 후보를 반환한다."""
        with self._cache_lock:
            cache = dict(self._cache)
        if cache.get("target_class_id") != target_class_id:
            return None
        if int(cache.get("request_id", 0)) < min_request_id:
            return None
        if (time.time() - float(cache.get("updated_at", 0.0))) > self._cache_max_age_sec:
            return None
        return list(cache.get("candidates") or [])

    def cached_estimate(
        self,
        target_class_id: int,
        port_hint: Optional[str] = None,
        target_module_name: Optional[str] = None,
        min_request_id: int = 0,
    ) -> Optional[np.ndarray]:
        """현재 캐시에 있는 최신 후보 중 task hint와 맞는 포트 base 좌표를 즉시 반환한다."""
        candidates = self._cached_candidates(
            target_class_id,
            min_request_id=min_request_id,
        )
        if not candidates:
            return None

        if port_hint is not None or target_module_name is not None:
            return self.select_by_task_hint(
                candidates,
                port_name=port_hint or "",
                target_module_name=target_module_name or "",
            )
        return candidates[0]["pos"]

    @staticmethod
    def _image_from_msg(img_msg):
        """ROS Image 메시지를 OpenCV가 사용하는 BGR numpy 이미지로 변환한다."""
        import cv2

        img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height,
            img_msg.width,
            3,
        )
        if img_msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    @staticmethod
    def _quat_to_matrix_xyzw(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """xyzw 순서의 쿼터니언을 3x3 회전 행렬로 변환한다."""
        norm = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
        if norm < 1e-12:
            return np.eye(3, dtype=float)
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=float,
        )

    @classmethod
    def _matrix_from_translation_quat(cls, translation, quat_xyzw) -> np.ndarray:
        """translation과 xyzw 쿼터니언으로 4x4 동차 변환 행렬을 만든다."""
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = cls._quat_to_matrix_xyzw(*quat_xyzw)
        matrix[:3, 3] = np.asarray(translation, dtype=float)
        return matrix

    @classmethod
    def _matrix_from_pose(cls, pose) -> np.ndarray:
        """ROS Pose를 base/TCP 계산에 쓰는 4x4 동차 변환 행렬로 바꾼다."""
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
        """현재 TCP pose와 고정 카메라 extrinsic으로 base->camera optical 행렬을 계산한다."""
        try:
            t_base_tcp = self._matrix_from_pose(obs.controller_state.tcp_pose)
            t_base_tool0 = t_base_tcp @ np.linalg.inv(self._t_tool0_tcp)
            t_base_optical = t_base_tool0 @ self._t_tool0_to_optical[camera_name]
            return np.linalg.inv(t_base_optical)
        except Exception as exc:
            if self._logger:
                self._logger.warn(
                    f"Vision: {camera_name} camera extrinsic calculation failed: {exc}"
                )
            return None

    def _detect(
        self,
        image: np.ndarray,
        cam_name: str = "",
        target_class_id: Optional[int] = None,
        target_port_index: Optional[int] = None,
    ) -> list:
        """단일 카메라 이미지에서 YOLO 검출을 수행하고 포트 후보 2D 점을 추출한다."""
        import cv2

        predict_kwargs = {"verbose": False, "conf": self._conf_thresh}
        if self._yolo_device:
            predict_kwargs["device"] = self._yolo_device
        results = self._model(image, **predict_kwargs)
        raw_dets = []
        for result in results:
            keypoints_xy = None
            if result.keypoints is not None and result.keypoints.xy is not None:
                keypoints_xy = result.keypoints.xy.detach().cpu().numpy()

            for box_idx, box in enumerate(result.boxes):
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
                            raw_dets.append(
                                {
                                    "class_id": cls,
                                    "class_name": f"sfp_port_{port_index}",
                                    "conf": conf,
                                    "u": float(center[0]),
                                    "v": float(center[1]),
                                    "xyxy": xyxy,
                                    "point_name": f"sfp_port_{port_index}",
                                    "port_index": port_index,
                                    "keypoints": group,
                                }
                            )
                        continue

                    if len(kpts) >= 4:
                        group = kpts[:4]
                        if np.all(np.isfinite(group)):
                            center = np.mean(group, axis=0)
                            raw_dets.append(
                                {
                                    "class_id": cls,
                                    "class_name": "sc_port",
                                    "conf": conf,
                                    "u": float(center[0]),
                                    "v": float(center[1]),
                                    "xyxy": xyxy,
                                    "point_name": "sc_port",
                                    "port_index": None,
                                    "keypoints": group,
                                }
                            )
                            continue

                raw_dets.append(
                    {
                        "class_id": cls,
                        "class_name": self.CLASS_NAMES.get(cls, "unknown"),
                        "conf": conf,
                        "u": float((x1 + x2) / 2),
                        "v": float((y1 + y2) / 2),
                        "xyxy": xyxy,
                        "point_name": "bbox_center",
                        "port_index": None,
                    }
                )

        if self._logger:
            if raw_dets:
                for det in raw_dets:
                    flag = "ok" if det["conf"] >= self._conf_thresh else "low"
                    self._logger.info(
                        f"  [{cam_name}] {flag} {det['class_name']} "
                        f"point={det.get('point_name', 'bbox_center')} "
                        f"conf={det['conf']:.3f} (thresh={self._conf_thresh}) "
                        f"uv=({det['u']:.0f},{det['v']:.0f})"
                    )
            else:
                self._logger.warn(f"  [{cam_name}] no detections")

        passed_dets = [det for det in raw_dets if det["conf"] >= self._conf_thresh]
        if self._debug_save_enabled and self._debug_save_dir is not None and passed_dets:
            debug_img = image.copy()
            colors = {0: (0, 255, 0), 1: (255, 100, 0), -1: (0, 0, 255)}

            for det in passed_dets:
                x1, y1, x2, y2 = det["xyxy"]
                color = colors.get(det["class_id"], colors[-1])
                if det.get("port_index") == 1:
                    color = (255, 0, 255)
                is_target_class = (
                    target_class_id is not None
                    and int(det["class_id"]) == int(target_class_id)
                )
                det_port_index = det.get("port_index")
                is_target_port = (
                    target_port_index is None
                    or det_port_index is None
                    or int(det_port_index) == int(target_port_index)
                )
                is_target = is_target_class and is_target_port
                center = (int(round(det["u"])), int(round(det["v"])))

                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 3)
                cv2.circle(debug_img, center, 5, color, -1)
                if is_target:
                    cross_size = 18
                    cross_color = (0, 255, 255)
                    for thickness, line_color in ((8, (0, 0, 0)), (4, cross_color)):
                        cv2.line(
                            debug_img,
                            (center[0] - cross_size, center[1] - cross_size),
                            (center[0] + cross_size, center[1] + cross_size),
                            line_color,
                            thickness,
                            cv2.LINE_AA,
                        )
                        cv2.line(
                            debug_img,
                            (center[0] - cross_size, center[1] + cross_size),
                            (center[0] + cross_size, center[1] - cross_size),
                            line_color,
                            thickness,
                            cv2.LINE_AA,
                        )
                    cv2.circle(debug_img, center, 10, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.circle(debug_img, center, 10, cross_color, 2, cv2.LINE_AA)
                label = f"{det['class_name']} {det['conf']:.2f}"
                if is_target:
                    label = f"TARGET {label}"
                cv2.putText(
                    debug_img,
                    label,
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            self._put_text_lines(
                debug_img,
                [
                    f"task={self._debug_task_label}",
                    f"thresh={self._conf_thresh:.2f} cam={cam_name} dets={len(passed_dets)}",
                ],
                10,
                24,
            )

            fname = (
                self._debug_save_dir
                / self._sanitize_debug_token(cam_name or "camera")
                / f"{self._debug_task_label}__detect_{self._debug_call_count:04d}.jpg"
            )
            os.makedirs(fname.parent, exist_ok=True)
            saved = cv2.imwrite(str(fname), debug_img)
            if self._logger:
                if saved:
                    self._logger.info(f"[Vision Debug] saved: {fname}")
                else:
                    self._logger.warn(f"[Vision Debug] save failed: {fname}")
            self._debug_call_count += 1

        return passed_dets

    @staticmethod
    def _triangulate(u_a, v_a, k_a, t_base_to_optA, u_b, v_b, k_b, t_base_to_optB) -> np.ndarray:
        """두 카메라의 2D 픽셀 대응점으로 base 좌표계의 3D 점을 삼각측량한다."""
        import cv2

        p_a = k_a @ t_base_to_optA[:3, :]
        p_b = k_b @ t_base_to_optB[:3, :]
        pts_4d = cv2.triangulatePoints(
            p_a,
            p_b,
            np.array([[u_a], [v_a]], dtype=np.float64),
            np.array([[u_b], [v_b]], dtype=np.float64),
        )
        return (pts_4d[:3] / pts_4d[3]).flatten()

    def estimate(
        self,
        obs,
        target_class_id: int,
        port_hint: Optional[str] = None,
        target_module_name: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """포트 후보들 중 task hint와 가장 맞는 하나의 base 좌표를 반환한다."""
        candidates = self.estimate_all(
            obs,
            target_class_id,
            port_hint=port_hint,
        )
        if not candidates:
            return None

        if port_hint is not None or target_module_name is not None:
            chosen = self.select_by_task_hint(
                candidates,
                port_name=port_hint or "",
                target_module_name=target_module_name or "",
            )
            if chosen is not None:
                return chosen
            return None

        return candidates[0]["pos"]

    def estimate_all(
        self,
        obs,
        target_class_id: int,
        port_hint: Optional[str] = None,
    ) -> list:
        """비동기 추론 요청 후 timeout 안에 들어온 모든 3D 포트 후보를 반환한다."""
        if obs is None:
            return []

        target_port_index = self._extract_named_index(port_hint or "", "sfp_port")
        request_id = self._submit_estimate(
            obs,
            target_class_id,
            target_port_index=target_port_index,
        )
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
                f"Vision: async YOLO result timed out "
                f"(target_class_id={target_class_id})"
            )
        return []

    def _estimate_all_sync(
        self,
        obs,
        target_class_id: int,
        target_port_index: Optional[int] = None,
    ) -> list:
        """세 카메라의 YOLO 결과를 동기적으로 모아 3D 포트 후보 리스트를 만든다."""
        self._ensure_loaded()
        if not self._loaded:
            return []

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
        t_base_to = {}
        for name, _ in self.CAMERAS:
            t_base_to_camera = self._base_to_camera_optical_matrix(obs, name)
            if t_base_to_camera is None:
                return []
            t_base_to[name] = t_base_to_camera

        if self._logger:
            self._logger.info(f"Vision: detecting target_class_id={target_class_id}")
        detections = {}
        for name in ["left", "center", "right"]:
            dets = self._detect(
                images[name],
                cam_name=name,
                target_class_id=target_class_id,
                target_port_index=target_port_index,
            )
            detections[name] = [det for det in dets if det["class_id"] == target_class_id]

        cams_with_dets = [name for name, dets in detections.items() if dets]
        if len(cams_with_dets) < 2:
            if self._logger:
                self._logger.warn(
                    f"Vision: fewer than 2 cameras detected target ({cams_with_dets})"
                )
            return []

        available_pairs = []
        for cam_a, cam_b in [("left", "center"), ("center", "right"), ("left", "right")]:
            if cam_a in cams_with_dets and cam_b in cams_with_dets:
                available_pairs.append((cam_a, cam_b))
        if not available_pairs:
            return []

        k_all = {name: np.array(cam_infos[name].k).reshape(3, 3) for name, _ in self.CAMERAS}
        k = {name: k_all[name] for name in cams_with_dets}
        t_base_to = {name: t_base_to[name] for name in cams_with_dets}
        board_center = np.array(FinalPolicyConfig.BOARD_CENTER)
        pixel_match_thresh = 30.0
        candidates = []

        cam_a, cam_b = available_pairs[-1]
        for det_a in detections[cam_a]:
            for det_b in detections[cam_b]:
                if det_a.get("point_name") != det_b.get("point_name"):
                    continue
                port_3d = self._triangulate(
                    det_a["u"],
                    det_a["v"],
                    k[cam_a],
                    t_base_to[cam_a],
                    det_b["u"],
                    det_b["v"],
                    k[cam_b],
                    t_base_to[cam_b],
                )
                dist = float(np.linalg.norm(port_3d - board_center))
                if dist > FinalPolicyConfig.BOARD_RADIUS:
                    continue
                if not (FinalPolicyConfig.Z_RANGE[0] <= port_3d[2] <= FinalPolicyConfig.Z_RANGE[1]):
                    continue

                camera_points = {
                    cam_a: (float(det_a["u"]), float(det_a["v"])),
                    cam_b: (float(det_b["u"]), float(det_b["v"])),
                }

                third_cam = None
                for name in ["center", "left", "right"]:
                    if name not in (cam_a, cam_b) and name in detections:
                        third_cam = name
                        break

                consistent = True
                if third_cam and detections[third_cam]:
                    u_proj, v_proj = project_3d_to_pixel(
                        port_3d,
                        k[third_cam],
                        t_base_to[third_cam],
                    )
                    same_point_dets = [
                        det
                        for det in detections[third_cam]
                        if det.get("point_name") == det_a.get("point_name")
                    ]
                    if not same_point_dets:
                        continue
                    min_px_dist, nearest_det = min(
                        (
                            float(np.hypot(det["u"] - u_proj, det["v"] - v_proj)),
                            det,
                        )
                        for det in same_point_dets
                    )
                    if min_px_dist > pixel_match_thresh:
                        consistent = False
                    else:
                        camera_points[third_cam] = (
                            float(nearest_det["u"]),
                            float(nearest_det["v"]),
                        )

                if not consistent:
                    continue

                conf_sum = det_a["conf"] + det_b["conf"]
                score = dist - 0.1 * conf_sum
                candidates.append(
                    {
                        "pos": port_3d,
                        "score": score,
                        "conf_sum": conf_sum,
                        "point_name": det_a.get("point_name"),
                        "port_index": det_a.get("port_index"),
                        "camera_points": camera_points,
                    }
                )

        candidates.sort(key=lambda candidate: candidate["score"])

        unique = []
        for candidate in candidates:
            is_duplicate = False
            for existing in unique:
                if np.linalg.norm(candidate["pos"] - existing["pos"]) < 0.01:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(candidate)

        if self._logger and unique:
            self._logger.info(
                f"Vision: {len(unique)} candidate ports "
                f"(best=({unique[0]['pos'][0]:+.3f}, "
                f"{unique[0]['pos'][1]:+.3f}, {unique[0]['pos'][2]:+.3f}), "
                f"point={unique[0].get('point_name')})"
            )

        return unique

    @staticmethod
    def _extract_named_index(text: str, prefix: str) -> Optional[int]:
        """sfp_port_0 같은 이름에서 prefix 뒤 숫자 인덱스를 추출한다."""
        match = re.search(rf"{re.escape(prefix)}_(\d+)", text or "")
        if match is None:
            return None
        return int(match.group(1))

    @staticmethod
    def _put_text_lines(image: np.ndarray, lines: list[str], x: int, y: int) -> None:
        """디버그 이미지 위에 가독성 있는 상태 텍스트를 그린다."""
        import cv2

        colors = ((0, 255, 255), (0, 200, 255), (255, 255, 0))
        for index, line in enumerate(lines):
            origin = (x, y + index * 22)
            cv2.putText(
                image,
                line,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                4,
            )
            cv2.putText(
                image,
                line,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                colors[index % len(colors)],
                2,
            )

    def select_by_task_hint(
        self,
        candidates: list,
        port_name: str = "",
        target_module_name: str = "",
    ) -> Optional[np.ndarray]:
        """task의 port/module 이름을 이용해 여러 3D 후보 중 목표 포트를 선택한다."""
        if not candidates:
            return None
        if len(candidates) == 1 and not target_module_name:
            return candidates[0]["pos"]

        sfp_port_index = self._extract_named_index(port_name, "sfp_port")
        port_candidates = candidates
        if sfp_port_index is not None:
            same_port = [
                candidate
                for candidate in candidates
                if candidate.get("port_index") == sfp_port_index
            ]
            if same_port:
                port_candidates = same_port

        sc_port_index = self._extract_named_index(port_name, "sc_port")
        if sc_port_index is None:
            sc_port_index = self._extract_named_index(target_module_name, "sc_port")
        if sc_port_index is not None and port_candidates:
            return port_candidates[0]["pos"]

        return self.select_by_port_name(port_candidates, port_name)

    @staticmethod
    def select_by_port_name(candidates: list, port_name: str) -> Optional[np.ndarray]:
        """포트 이름 규칙만으로 후보를 선택한다. 실패 시 가장 점수가 좋은 후보를 쓴다."""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]["pos"]

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

            sorted_by_x = sorted(candidates, key=lambda candidate: candidate["pos"][0])
            if port_name.endswith("_0") or port_name.endswith("0_link"):
                return sorted_by_x[-1]["pos"]
            if port_name.endswith("_1") or port_name.endswith("1_link"):
                return sorted_by_x[0]["pos"]

        return candidates[0]["pos"]
