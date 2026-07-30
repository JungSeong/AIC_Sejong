from __future__ import annotations

"""FinalPolicy의 triangulation/align 디버그 이미지와 시각화 도우미."""

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
from geometry_msgs.msg import Pose

from final_policy.config import FinalPolicyConfig
from final_policy.geometry import project_3d_to_pixel
from final_policy.vision import VisionPortEstimator, observation_camera_timing


class FinalPolicyDebugMixin:
    """정책 제어와 분리된 TF 조회 및 디버그 이미지 저장 기능."""

    def _create_detection_debug_image_publishers(self) -> dict:
        """RViz Image display용 camera별 YOLO detection publisher를 만든다."""
        if not FinalPolicyConfig.DETECTION_DEBUG_RVIZ_ENABLED:
            return {}

        from sensor_msgs.msg import Image

        prefix = str(
            FinalPolicyConfig.DETECTION_DEBUG_IMAGE_TOPIC_PREFIX or ""
        ).strip().rstrip("/")
        if not prefix:
            prefix = "/final_policy/detection_debug"
        publishers = {}
        topics = []
        for camera, _frame in VisionPortEstimator.CAMERAS:
            topic = f"{prefix}/{camera}/image"
            publishers[camera] = self._parent_node.create_publisher(Image, topic, 10)
            topics.append(topic)
        self.get_logger().info("[Detection RViz] image topics: " + ", ".join(topics))
        return publishers

    def _publish_detection_debug_image(
        self,
        camera: str,
        debug_image: np.ndarray,
        source_image_msg,
    ) -> None:
        """YOLO overlay를 원본 camera header를 보존해 해당 camera topic에 발행한다."""
        publisher = getattr(self, "_detection_debug_image_pubs", {}).get(camera)
        if publisher is None:
            return
        publisher.publish(
            self._bgr_debug_image_message(debug_image, source_image_msg)
        )

    def _create_triangulation_debug_marker_publisher(self):
        """RViz 3D display용 prediction/GT/error MarkerArray publisher를 만든다."""
        if not FinalPolicyConfig.TRIANGULATION_DEBUG_RVIZ_ENABLED:
            return None

        from visualization_msgs.msg import MarkerArray

        topic = str(FinalPolicyConfig.TRIANGULATION_DEBUG_MARKER_TOPIC or "").strip()
        if not topic:
            topic = "/final_policy/triangulation_debug/markers"
        publisher = self._parent_node.create_publisher(MarkerArray, topic, 10)
        self.get_logger().info(f"[Triangulation RViz] marker topic: {topic}")
        return publisher

    def _create_triangulation_debug_image_publishers(self) -> dict:
        """RViz Image display용 camera별 triangulation overlay publisher를 만든다."""
        if not FinalPolicyConfig.TRIANGULATION_DEBUG_RVIZ_ENABLED:
            return {}

        from sensor_msgs.msg import Image

        prefix = str(
            FinalPolicyConfig.TRIANGULATION_DEBUG_IMAGE_TOPIC_PREFIX or ""
        ).strip().rstrip("/")
        if not prefix:
            prefix = "/final_policy/triangulation_debug"
        publishers = {}
        topics = []
        for camera, _frame in VisionPortEstimator.CAMERAS:
            topic = f"{prefix}/{camera}/image"
            publishers[camera] = self._parent_node.create_publisher(Image, topic, 10)
            topics.append(topic)
        self.get_logger().info(
            "[Triangulation RViz] image topics: " + ", ".join(topics)
        )
        return publishers

    @staticmethod
    def _bgr_debug_image_message(image: np.ndarray, source_image_msg):
        """OpenCV BGR 이미지를 원본 header를 보존한 sensor_msgs/Image로 바꾼다."""
        from sensor_msgs.msg import Image

        bgr = np.ascontiguousarray(image, dtype=np.uint8)
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR image, got shape={bgr.shape}")

        message = Image()
        message.header.stamp.sec = int(source_image_msg.header.stamp.sec)
        message.header.stamp.nanosec = int(source_image_msg.header.stamp.nanosec)
        message.header.frame_id = str(source_image_msg.header.frame_id)
        message.height = int(bgr.shape[0])
        message.width = int(bgr.shape[1])
        message.encoding = "bgr8"
        message.is_bigendian = 0
        message.step = int(bgr.shape[1] * 3)
        message.data = bgr.tobytes()
        return message

    @staticmethod
    def _triangulation_marker_array(
        predicted_port: np.ndarray,
        gt_port: Optional[np.ndarray],
        stamp,
        label: str,
    ):
        """base_link 기준 prediction, GT 및 오차선을 MarkerArray로 구성한다."""
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker, MarkerArray

        def point(values: np.ndarray) -> Point:
            return Point(
                x=float(values[0]),
                y=float(values[1]),
                z=float(values[2]),
            )

        def marker(marker_id: int, marker_type: int) -> Marker:
            item = Marker()
            item.header.frame_id = "base_link"
            item.header.stamp.sec = int(stamp.sec)
            item.header.stamp.nanosec = int(stamp.nanosec)
            item.ns = "final_policy_triangulation"
            item.id = marker_id
            item.type = marker_type
            item.action = Marker.ADD
            item.pose.orientation.w = 1.0
            return item

        predicted = np.asarray(predicted_port, dtype=np.float64).reshape(3)
        pred_marker = marker(0, Marker.SPHERE)
        pred_marker.pose.position = point(predicted)
        pred_marker.scale.x = pred_marker.scale.y = pred_marker.scale.z = 0.012
        pred_marker.color.r = 1.0
        pred_marker.color.g = 1.0
        pred_marker.color.a = 1.0
        pred_marker.text = f"PRED {label}"

        markers = [pred_marker]
        if gt_port is None:
            for marker_id in (1, 2):
                delete_marker = marker(marker_id, Marker.SPHERE)
                delete_marker.action = Marker.DELETE
                markers.append(delete_marker)
            return MarkerArray(markers=markers)

        gt = np.asarray(gt_port, dtype=np.float64).reshape(3)
        gt_marker = marker(1, Marker.SPHERE)
        gt_marker.pose.position = point(gt)
        gt_marker.scale.x = gt_marker.scale.y = gt_marker.scale.z = 0.012
        gt_marker.color.r = 1.0
        gt_marker.color.b = 1.0
        gt_marker.color.a = 1.0
        gt_marker.text = "GT port entrance"

        error_line = marker(2, Marker.LINE_LIST)
        error_line.scale.x = 0.003
        error_line.color.r = 1.0
        error_line.color.g = 1.0
        error_line.color.b = 1.0
        error_line.color.a = 1.0
        error_line.points = [point(predicted), point(gt)]
        markers.extend([gt_marker, error_line])
        return MarkerArray(markers=markers)

    def _publish_triangulation_debug_markers(
        self,
        predicted_port: np.ndarray,
        gt_port: Optional[np.ndarray],
        stamp,
        label: str,
    ) -> bool:
        """현재 triangulation의 3D prediction/GT/error MarkerArray를 발행한다."""
        publisher = getattr(self, "_triangulation_debug_marker_pub", None)
        if publisher is None:
            return False
        publisher.publish(
            self._triangulation_marker_array(predicted_port, gt_port, stamp, label)
        )
        return True

    def _align_debug_save_dir(self) -> Optional[Path]:
        """align 디버그 이미지를 저장할 디렉토리를 반환한다."""
        debug_root = getattr(VisionPortEstimator, "DEBUG_SAVE_DIR", None)
        if not debug_root:
            return None
        return Path(debug_root) / "align"

    def _triangulation_debug_save_dir(self) -> Optional[Path]:
        """triangulation GT/PRED 비교 디버그 이미지를 저장할 디렉토리를 반환한다."""
        if not FinalPolicyConfig.TRIANGULATION_DEBUG_SAVE_ENABLED:
            return None
        debug_root = getattr(VisionPortEstimator, "DEBUG_SAVE_DIR", None)
        if not debug_root:
            return None
        return Path(debug_root) / "triangulation"

    def _target_port_entrance_frame(self) -> str:
        """현재 task의 실제 포트 entrance GT frame 이름을 만든다."""
        module = str(getattr(self._task, "target_module_name", "") or "")
        port_name = str(getattr(self._task, "port_name", "") or "")
        if self._port_type() == "sc":
            return f"task_board/{module}/sc_port_base_link_entrance"
        if port_name.endswith("_link"):
            return f"task_board/{module}/{port_name}_entrance"
        return f"task_board/{module}/{port_name}_link_entrance"

    def _lookup_frame_base(
        self,
        target_frame: str,
        *,
        stamp=None,
        warn_on_failure: bool = True,
    ) -> Optional[np.ndarray]:
        """TF에서 camera stamp와 동기화된 target frame의 base 좌표를 조회한다."""
        buffer = getattr(self._parent_node, "_tf_buffer", None)
        if buffer is None:
            if warn_on_failure:
                self.get_logger().warn(
                    "[Triangulation Debug] parent node has no TF buffer"
                )
            return None

        from rclpy.duration import Duration
        from rclpy.time import Time

        def translation_array(transform_msg) -> np.ndarray:
            transform = transform_msg.transform
            return np.array(
                [
                    float(transform.translation.x),
                    float(transform.translation.y),
                    float(transform.translation.z),
                ],
                dtype=np.float64,
            )

        query_time = Time() if stamp is None else Time.from_msg(stamp)
        errors = []
        try:
            transform_msg = buffer.lookup_transform(
                "base_link",
                target_frame,
                query_time,
                timeout=Duration(seconds=0.2),
            )
            return translation_array(transform_msg)
        except Exception as direct_exc:
            errors.append(f"direct={direct_exc}")

        fixed_frame = str(
            FinalPolicyConfig.TRIANGULATION_DEBUG_FIXED_FRAME or ""
        ).strip()
        if fixed_frame and stamp is not None:
            try:
                transform_msg = buffer.lookup_transform_full(
                    "base_link",
                    query_time,
                    target_frame,
                    query_time,
                    fixed_frame,
                    timeout=Duration(seconds=0.2),
                )
                return translation_array(transform_msg)
            except Exception as full_exc:
                errors.append(f"fixed={full_exc}")

        if stamp is not None:
            try:
                latest_msg = buffer.lookup_transform(
                    "base_link",
                    target_frame,
                    Time(),
                    timeout=Duration(seconds=0.2),
                )
                requested_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
                latest_ns = (
                    int(latest_msg.header.stamp.sec) * 1_000_000_000
                    + int(latest_msg.header.stamp.nanosec)
                )
                delta_ms = abs(latest_ns - requested_ns) / 1_000_000.0
                threshold_ms = max(
                    0.0,
                    FinalPolicyConfig.TRIANGULATION_SYNC_THRESHOLD_MS,
                )
                if latest_ns > 0 and delta_ms <= threshold_ms:
                    self.get_logger().warn(
                        "[Triangulation Debug] exact GT TF unavailable; "
                        f"using nearby TF delta={delta_ms:.3f}ms"
                    )
                    return translation_array(latest_msg)
                errors.append(
                    f"latest_delta={delta_ms:.3f}ms threshold={threshold_ms:.3f}ms"
                )
            except Exception as latest_exc:
                errors.append(f"latest={latest_exc}")

        if warn_on_failure:
            self.get_logger().warn(
                f"[Triangulation Debug] synchronized GT lookup failed: "
                f"base_link <- {target_frame}: {'; '.join(errors)}"
            )
        return None

    def _lookup_gt_port_base(self, *, stamp=None) -> tuple[str, Optional[np.ndarray]]:
        """TF에서 base_link 기준 실제 포트 entrance 좌표를 조회한다."""
        target_frame = self._target_port_entrance_frame()
        return target_frame, self._lookup_frame_base(target_frame, stamp=stamp)

    @staticmethod
    def _projected_point_visible(point_px: np.ndarray, image_shape: tuple[int, ...]) -> bool:
        """투영된 픽셀이 이미지 내부에 있는지 확인한다."""
        if not np.isfinite(point_px).all():
            return False
        height, width = image_shape[:2]
        return 0.0 <= point_px[0] < float(width) and 0.0 <= point_px[1] < float(height)

    @staticmethod
    def _draw_triangulation_point(
        image: np.ndarray,
        point_px: np.ndarray,
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        """debug 이미지 위에 PRED/GT 포인트 마커와 라벨을 그린다."""
        import cv2

        x = int(round(float(point_px[0])))
        y = int(round(float(point_px[1])))
        cv2.drawMarker(
            image,
            (x, y),
            (0, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=26,
            thickness=7,
            line_type=cv2.LINE_AA,
        )
        cv2.drawMarker(
            image,
            (x, y),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=22,
            thickness=3,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(image, (x, y), 6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(image, (x, y), 6, color, 2, cv2.LINE_AA)
        text_xy = (x + 8, max(16, y - 8))
        cv2.putText(
            image,
            label,
            text_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            text_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    def _save_triangulation_debug_images(
        self,
        *,
        obs,
        vision: VisionPortEstimator,
        predicted_port: np.ndarray,
        label: str,
    ) -> None:
        """PRED/GT overlay를 파일로 저장하고 timestamp를 보존해 RViz에 발행한다."""
        save_dir = self._triangulation_debug_save_dir()
        image_publishers = getattr(self, "_triangulation_debug_image_pubs", {})
        marker_publisher = getattr(self, "_triangulation_debug_marker_pub", None)
        if (
            save_dir is None
            and not image_publishers
            and marker_publisher is None
        ) or obs is None or vision is None:
            return

        try:
            import cv2

            predicted_port = np.asarray(predicted_port, dtype=np.float64).reshape(3)
            camera_timing = observation_camera_timing(obs)
            if camera_timing is None:
                self.get_logger().warn(
                    "[Triangulation Debug] missing or zero camera timestamp"
                )
                return
            _reference_ns, camera_stamps_ns, sync_span_ms = camera_timing
            reference_stamp = obs.center_image.header.stamp
            target_frame, gt_port = self._lookup_gt_port_base(
                stamp=reference_stamp
            )
            frame_id = self._triangulation_debug_call_count
            self._triangulation_debug_call_count += 1
            if frame_id == 0 and save_dir is not None:
                self.get_logger().info(f"[Triangulation Debug] dir: {save_dir}")

            task_label = VisionPortEstimator._sanitize_debug_token(
                getattr(vision, "debug_task_label", "task_unknown")
            ) or "task_unknown"
            target_frame_short = target_frame.replace("task_board/", "", 1)
            error = None if gt_port is None else predicted_port - gt_port
            error_norm_mm = (
                None if error is None else float(np.linalg.norm(error) * 1000.0)
            )
            marker_published = self._publish_triangulation_debug_markers(
                predicted_port,
                gt_port,
                reference_stamp,
                label,
            )
            saved_count = 0
            published_count = 0

            for cam_name, _ in VisionPortEstimator.CAMERAS:
                img_msg = getattr(obs, f"{cam_name}_image", None)
                camera_info = getattr(obs, f"{cam_name}_camera_info", None)
                if img_msg is None or camera_info is None:
                    continue

                debug_img = VisionPortEstimator._image_from_msg(img_msg).copy()
                k_matrix = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
                t_camera_base = vision._base_to_camera_optical_matrix(obs, cam_name)
                pred_px = None
                gt_px = None
                if t_camera_base is not None:
                    pred_px = np.array(
                        project_3d_to_pixel(predicted_port, k_matrix, t_camera_base),
                        dtype=np.float64,
                    )
                    if self._projected_point_visible(pred_px, debug_img.shape):
                        self._draw_triangulation_point(
                            debug_img,
                            pred_px,
                            "PRED",
                            (0, 255, 255),
                        )

                    if gt_port is not None:
                        gt_px = np.array(
                            project_3d_to_pixel(gt_port, k_matrix, t_camera_base),
                            dtype=np.float64,
                        )
                        if self._projected_point_visible(gt_px, debug_img.shape):
                            self._draw_triangulation_point(
                                debug_img,
                                gt_px,
                                "GT",
                                (255, 0, 255),
                            )

                    if (
                        gt_px is not None
                        and pred_px is not None
                        and self._projected_point_visible(pred_px, debug_img.shape)
                        and self._projected_point_visible(gt_px, debug_img.shape)
                    ):
                        cv2.line(
                            debug_img,
                            tuple(np.round(pred_px).astype(int)),
                            tuple(np.round(gt_px).astype(int)),
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                text_lines = [
                    f"task={task_label} source={label} cam={cam_name}",
                    f"target_gt={target_frame_short}",
                    (
                        "stamp="
                        f"{img_msg.header.stamp.sec}."
                        f"{img_msg.header.stamp.nanosec:09d} "
                        f"sync_span={sync_span_ms:.3f}ms"
                    ),
                    (
                        "pred_xyz="
                        f"({predicted_port[0]:+.4f}, "
                        f"{predicted_port[1]:+.4f}, "
                        f"{predicted_port[2]:+.4f})m"
                    ),
                ]
                if gt_port is None:
                    text_lines.extend(
                        [
                            f"gt=unavailable frame={target_frame}",
                            "error=unavailable",
                        ]
                    )
                else:
                    text_lines.extend(
                        [
                            (
                                "gt_xyz="
                                f"({gt_port[0]:+.4f}, "
                                f"{gt_port[1]:+.4f}, "
                                f"{gt_port[2]:+.4f})m"
                            ),
                            (
                                "err_mm="
                                f"dx={error[0] * 1000.0:+.1f} "
                                f"dy={error[1] * 1000.0:+.1f} "
                                f"dz={error[2] * 1000.0:+.1f} "
                                f"norm={error_norm_mm:.1f}"
                            ),
                        ]
                    )

                VisionPortEstimator._put_text_lines(debug_img, text_lines, 10, 24)
                publisher = image_publishers.get(cam_name)
                if publisher is not None:
                    publisher.publish(
                        self._bgr_debug_image_message(debug_img, img_msg)
                    )
                    published_count += 1

                if save_dir is not None:
                    fname = (
                        save_dir
                        / VisionPortEstimator._sanitize_debug_token(
                            cam_name or "camera"
                        )
                        / f"{task_label}__triangulation_{frame_id:04d}.jpg"
                    )
                    os.makedirs(fname.parent, exist_ok=True)
                    if cv2.imwrite(str(fname), debug_img):
                        saved_count += 1
                        self.get_logger().info(
                            f"\033[1;92m[Triangulation Debug] saved: {fname}\033[0m"
                        )
                    else:
                        self.get_logger().warn(
                            f"[Triangulation Debug] save failed: {fname}"
                        )
            if save_dir is not None and saved_count == 0:
                self.get_logger().warn(
                    "[Triangulation Debug] no images saved "
                    f"(task={task_label}, source={label})"
                )
            if image_publishers:
                self.get_logger().info(
                    "\033[1;32m[Triangulation RViz] "
                    f"published={published_count}, "
                    f"marker={marker_published}, "
                    f"sync_span={sync_span_ms:.3f}ms\033[0m"
                )
        except Exception as exc:
            self.get_logger().warn(f"[Triangulation Debug] processing failed: {exc}")

    @staticmethod
    def _pose_position_array(pose: Pose) -> np.ndarray:
        """Pose의 position을 base_link 3D numpy 벡터로 변환한다."""
        return np.array(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=np.float64,
        )

    @staticmethod
    def _clip_pixel(pixel: np.ndarray, width: int, height: int) -> tuple[int, int]:
        """이미지 밖으로 나간 픽셀 좌표를 이미지 경계 안으로 제한한다."""
        return (
            int(np.clip(pixel[0], 0, max(0, width - 1))),
            int(np.clip(pixel[1], 0, max(0, height - 1))),
        )

    @staticmethod
    def _draw_align_arrow(
        image: np.ndarray,
        start_px: np.ndarray,
        end_px: np.ndarray,
        label: str,
    ) -> bool:
        """align 추론 이동 방향을 이미지 위 화살표와 X 목표점으로 그린다."""
        import cv2

        height, width = image.shape[:2]
        start_px = np.asarray(start_px, dtype=np.float64)
        end_px = np.asarray(end_px, dtype=np.float64)
        if not (np.isfinite(start_px).all() and np.isfinite(end_px).all()):
            return False

        delta = end_px - start_px
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm < 1e-6:
            delta = np.array([1.0, 0.0], dtype=np.float64)
            delta_norm = 1.0

        start_in_image = 0 <= start_px[0] < width and 0 <= start_px[1] < height
        end_in_image = 0 <= end_px[0] < width and 0 <= end_px[1] < height
        if start_in_image and end_in_image:
            arrow_start = start_px
            arrow_end = end_px
        else:
            arrow_start = np.array([width * 0.5, height * 0.5], dtype=np.float64)
            arrow_len = min(max(delta_norm, 36.0), 120.0)
            arrow_end = arrow_start + delta / delta_norm * arrow_len
            label = f"{label} screen-scaled"

        p0 = FinalPolicyDebugMixin._clip_pixel(arrow_start, width, height)
        p1 = FinalPolicyDebugMixin._clip_pixel(arrow_end, width, height)
        for thickness, color in ((9, (0, 0, 0)), (5, (0, 255, 255))):
            cv2.arrowedLine(
                image,
                p0,
                p1,
                color,
                thickness,
                cv2.LINE_AA,
                tipLength=0.25,
            )
        cross_size = 14
        for thickness, color in ((7, (0, 0, 0)), (3, (0, 255, 255))):
            cv2.line(
                image,
                (p1[0] - cross_size, p1[1] - cross_size),
                (p1[0] + cross_size, p1[1] + cross_size),
                color,
                thickness,
                cv2.LINE_AA,
            )
            cv2.line(
                image,
                (p1[0] - cross_size, p1[1] + cross_size),
                (p1[0] + cross_size, p1[1] - cross_size),
                color,
                thickness,
                cv2.LINE_AA,
            )
        cv2.circle(image, p0, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, p0, 6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (max(8, p1[0] + 8), max(18, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (max(8, p1[0] + 8), max(18, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return True

    @staticmethod
    def _draw_axis_arrow(
        image: np.ndarray,
        start_px: np.ndarray,
        end_px: np.ndarray,
        label: str,
        color: tuple[int, int, int],
    ) -> bool:
        """카메라 이미지 위에 base 축 방향 참조 화살표를 그린다."""
        import cv2

        height, width = image.shape[:2]
        start_px = np.asarray(start_px, dtype=np.float64)
        end_px = np.asarray(end_px, dtype=np.float64)
        if not (np.isfinite(start_px).all() and np.isfinite(end_px).all()):
            return False
        if np.any(start_px < 0.0) or np.any(end_px < 0.0):
            return False

        start_in_image = 0 <= start_px[0] < width and 0 <= start_px[1] < height
        end_in_image = 0 <= end_px[0] < width and 0 <= end_px[1] < height
        if not (start_in_image or end_in_image):
            return False

        p0 = FinalPolicyDebugMixin._clip_pixel(start_px, width, height)
        p1 = FinalPolicyDebugMixin._clip_pixel(end_px, width, height)
        for thickness, line_color in ((7, (0, 0, 0)), (3, color)):
            cv2.arrowedLine(
                image,
                p0,
                p1,
                line_color,
                thickness,
                cv2.LINE_AA,
                tipLength=0.25,
            )
        text_origin = (max(8, p1[0] + 6), max(18, p1[1] - 6))
        cv2.putText(
            image,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
        return True

    def _draw_align_axes(
        self,
        image: np.ndarray,
        anchor_base: np.ndarray,
        k_matrix: np.ndarray,
        t_camera_base: np.ndarray,
    ) -> None:
        """align debug 이미지에 base_link +X/+Y 방향을 카메라 투영으로 표시한다."""
        axis_length = float(os.environ.get("AIC_ALIGN_DEBUG_AXIS_LENGTH_M", "0.03"))
        anchor_base = np.asarray(anchor_base, dtype=np.float64)
        anchor_px = np.array(
            project_3d_to_pixel(anchor_base, k_matrix, t_camera_base),
            dtype=np.float64,
        )
        axes = (
            ("+X base", np.array([axis_length, 0.0, 0.0], dtype=np.float64), (0, 0, 255)),
            ("+Y base", np.array([0.0, axis_length, 0.0], dtype=np.float64), (0, 255, 0)),
        )
        for label, axis, color in axes:
            axis_px = np.array(
                project_3d_to_pixel(anchor_base + axis, k_matrix, t_camera_base),
                dtype=np.float64,
            )
            self._draw_axis_arrow(image, anchor_px, axis_px, label, color)

    def _save_align_debug_images(
        self,
        *,
        obs,
        vision: VisionPortEstimator,
        tcp_pose: Pose,
        target_pose: Pose,
        step_index: int,
        position_correction: np.ndarray,
        rpy_correction: np.ndarray,
        step_xyz: np.ndarray,
        step_rpy: np.ndarray,
        position_xy_error: float,
        position_z_error: float,
        rpy_error: float,
        stable_count: int,
    ) -> None:
        """align 추론 보정량과 실제 이동 목표를 카메라 이미지 위에 저장한다."""
        save_dir = self._align_debug_save_dir()
        if save_dir is None or obs is None:
            return

        try:
            import cv2

            frame_id = self._align_debug_call_count
            self._align_debug_call_count += 1
            if frame_id == 0:
                self.get_logger().info(f"[Align Debug] dir: {save_dir}")
            task_label = VisionPortEstimator._sanitize_debug_token(
                getattr(vision, "debug_task_label", "task_unknown")
            ) or "task_unknown"
            move_anchor_base = self._pose_position_array(tcp_pose)
            move_target_base = self._pose_position_array(target_pose)

            for cam_name, _ in VisionPortEstimator.CAMERAS:
                img_msg = getattr(obs, f"{cam_name}_image", None)
                camera_info = getattr(obs, f"{cam_name}_camera_info", None)
                if img_msg is None or camera_info is None:
                    continue

                debug_img = VisionPortEstimator._image_from_msg(img_msg).copy()
                k_matrix = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
                t_camera_base = vision._base_to_camera_optical_matrix(obs, cam_name)
                arrow_drawn = False
                if t_camera_base is not None:
                    self._draw_align_axes(
                        debug_img,
                        move_anchor_base,
                        k_matrix,
                        t_camera_base,
                    )
                    start_px = np.array(
                        project_3d_to_pixel(move_anchor_base, k_matrix, t_camera_base),
                        dtype=np.float64,
                    )
                    target_px = np.array(
                        project_3d_to_pixel(move_target_base, k_matrix, t_camera_base),
                        dtype=np.float64,
                    )
                    if np.all(start_px >= 0.0) and np.all(target_px >= 0.0):
                        arrow_drawn = self._draw_align_arrow(
                            debug_img,
                            start_px,
                            target_px,
                            "OFFSET base-vector",
                        )

                if not arrow_drawn:
                    height, width = debug_img.shape[:2]
                    center = np.array([width * 0.5, height * 0.5], dtype=np.float64)
                    base_xy = np.array(
                        [float(step_xyz[0]), -float(step_xyz[1])],
                        dtype=np.float64,
                    )
                    base_norm = float(np.linalg.norm(base_xy))
                    if base_norm >= 1e-9:
                        length = min(max(base_norm * 4000.0, 36.0), 120.0)
                        self._draw_align_arrow(
                            debug_img,
                            center,
                            center + base_xy / base_norm * length,
                            "OFFSET base-xy",
                        )

                VisionPortEstimator._put_text_lines(
                    debug_img,
                    [
                        f"task={task_label} align_step={step_index:03d} cam={cam_name}",
                        (
                            "pred_xyz="
                            f"({position_correction[0] * 1000.0:+.1f}, "
                            f"{position_correction[1] * 1000.0:+.1f}, "
                            f"{position_correction[2] * 1000.0:+.1f})mm"
                        ),
                        (
                            "move_xyz="
                            f"({step_xyz[0] * 1000.0:+.1f}, "
                            f"{step_xyz[1] * 1000.0:+.1f}, "
                            f"{step_xyz[2] * 1000.0:+.1f})mm"
                        ),
                        (
                            "pred_rpy="
                            f"({math.degrees(rpy_correction[0]):+.1f}, "
                            f"{math.degrees(rpy_correction[1]):+.1f}, "
                            f"{math.degrees(rpy_correction[2]):+.1f})deg"
                        ),
                        (
                            "move_rpy="
                            f"({math.degrees(step_rpy[0]):+.1f}, "
                            f"{math.degrees(step_rpy[1]):+.1f}, "
                            f"{math.degrees(step_rpy[2]):+.1f})deg"
                        ),
                        (
                            f"xy_norm={position_xy_error * 1000.0:.1f}mm "
                            f"z_offset={position_z_error * 1000.0:.1f}mm "
                            f"rpy_norm={math.degrees(rpy_error):.1f}deg "
                            f"stable={stable_count}/"
                            f"{max(FinalPolicyConfig.STABLE_STEPS, FinalPolicyConfig.VISION_OFFSET_VALIDATION_STEPS)}"
                        ),
                    ],
                    10,
                    24,
                )

                fname = (
                    save_dir
                    / VisionPortEstimator._sanitize_debug_token(cam_name or "camera")
                    / f"{task_label}__align_{frame_id:04d}.jpg"
                )
                os.makedirs(fname.parent, exist_ok=True)
                if not cv2.imwrite(str(fname), debug_img):
                    self.get_logger().warn(f"[Align Debug] save failed: {fname}")
        except Exception as exc:
            self.get_logger().warn(f"[Align Debug] save failed: {exc}")
