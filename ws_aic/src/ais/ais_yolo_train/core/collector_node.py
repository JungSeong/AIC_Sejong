import csv
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from aic_control_interfaces.msg import (
    ControllerState,
    MotionUpdate,
    TrajectoryGenerationMode,
)
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener

try:
    from ais_transform import Pose3D
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2] / "ais_transform"))
    from ais_transform import Pose3D

try:
    from .dataset_format import (
        CAMERAS,
        MOUNT_CANDIDATE_COUNT,
        SC_PORT_CANDIDATE_COUNT,
        SC_PORT_CORNER_POINTS_M,
        TASK_BOARD_KEYPOINT_POINTS_M,
        draw_label,
        get_target_config,
        normalize_target,
    )
    from .geometry import (
        euler_to_quat,
        make_bbox_from_points,
        order_image_corners,
        port_corners_in_frame,
        project_to_camera,
    )
except ImportError:
    from dataset_format import (
        CAMERAS,
        MOUNT_CANDIDATE_COUNT,
        SC_PORT_CANDIDATE_COUNT,
        SC_PORT_CORNER_POINTS_M,
        TASK_BOARD_KEYPOINT_POINTS_M,
        draw_label,
        get_target_config,
        normalize_target,
    )
    from geometry import (
        euler_to_quat,
        make_bbox_from_points,
        order_image_corners,
        port_corners_in_frame,
        project_to_camera,
    )


class NicPortPoseCollector(Node):
    def __init__(self, target: str = "SFP"):
        super().__init__("approach_yolo_collector")
        self.target = normalize_target(target)
        self.config = get_target_config(self.target)
        self._cam_info = {}
        self._latest_image = {}
        self._latest_controller_state = None

        for name, _ in CAMERAS:
            self.create_subscription(
                CameraInfo,
                f"/{name}_camera/camera_info",
                lambda msg, n=name: self._on_info(n, msg),
                10,
            )
            self.create_subscription(
                Image,
                f"/{name}_camera/image",
                lambda msg, n=name: self._on_image(n, msg),
                10,
            )
        self.create_subscription(
            ControllerState,
            "/aic_controller/controller_state",
            self._on_controller_state,
            10,
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._motion_pub = self.create_publisher(
            MotionUpdate, "/aic_controller/pose_commands", 10
        )
        self._logged_tf_diagnostics = False
        self.get_logger().info(f"{self.target} approach YOLO collector ready.")

    @staticmethod
    def _format_optional_float(value: float | None) -> str:
        return "" if value is None else f"{float(value):.6f}"

    @staticmethod
    def _append_metadata_csv(output_dir: Path, row: dict) -> None:
        metadata_path = output_dir / "metadata.csv"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not metadata_path.exists()
        with metadata_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _detect_magenta_feature(
        image: np.ndarray,
        hue_min: int,
        hue_max: int,
        sat_min: int,
        val_min: int,
        min_area: float,
    ) -> dict | None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if hue_min <= hue_max:
            mask = cv2.inRange(
                hsv,
                np.array([hue_min, sat_min, val_min], dtype=np.uint8),
                np.array([hue_max, 255, 255], dtype=np.uint8),
            )
        else:
            lower_mask = cv2.inRange(
                hsv,
                np.array([0, sat_min, val_min], dtype=np.uint8),
                np.array([hue_max, 255, 255], dtype=np.uint8),
            )
            upper_mask = cv2.inRange(
                hsv,
                np.array([hue_min, sat_min, val_min], dtype=np.uint8),
                np.array([179, 255, 255], dtype=np.uint8),
            )
            mask = cv2.bitwise_or(lower_mask, upper_mask)

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            candidates.append((area, x, y, w, h))

        if not candidates:
            return None

        area, x, y, w, h = max(candidates, key=lambda item: item[0])
        return {
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "center_x": x + w / 2.0,
            "center_y": y + h / 2.0,
            "area_px": area,
            "candidate_count": len(candidates),
        }

    def detect_magenta_visibility(
        self,
        hue_min: int,
        hue_max: int,
        sat_min: int,
        val_min: int,
        min_area: float,
        quality_min_area: float,
        edge_margin_px: int,
    ) -> dict:
        best = None
        detections = []
        margin = max(0, int(edge_margin_px))

        for name, _ in CAMERAS:
            image = self._latest_image.get(name)
            if image is None:
                continue

            detection = self._detect_magenta_feature(
                image,
                hue_min,
                hue_max,
                sat_min,
                val_min,
                min_area,
            )
            if detection is None:
                continue

            h, w = image.shape[:2]
            x = int(detection["bbox_x"])
            y = int(detection["bbox_y"])
            box_w = int(detection["bbox_w"])
            box_h = int(detection["bbox_h"])
            edge_ok = (
                x >= margin
                and y >= margin
                and x + box_w <= w - margin
                and y + box_h <= h - margin
            )
            item = {
                "camera": name,
                **detection,
                "edge_ok": edge_ok,
            }
            detections.append(item)
            if best is None or float(item["area_px"]) > float(best["area_px"]):
                best = item

        quality = any(
            bool(item["edge_ok"]) and float(item["area_px"]) >= float(quality_min_area)
            for item in detections
        )
        return {
            "detected": bool(detections),
            "quality": quality,
            "best": best,
            "detections": detections,
        }

    def _on_info(self, name: str, msg: CameraInfo) -> None:
        self._cam_info[name] = msg

    def _on_controller_state(self, msg: ControllerState) -> None:
        self._latest_controller_state = msg

    def _on_image(self, name: str, msg: Image) -> None:
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1
        )
        if msg.encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif msg.encoding == "rgba8":
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif msg.encoding == "bgra8":
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif msg.encoding == "mono8":
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self._latest_image[name] = image

    def ready(self) -> bool:
        return len(self._cam_info) == len(CAMERAS) and len(self._latest_image) == len(CAMERAS)

    def controller_state_ready(self) -> bool:
        return self._latest_controller_state is not None

    @staticmethod
    def copy_pose(pose: Pose) -> Pose:
        return Pose(
            position=Point(
                x=float(pose.position.x),
                y=float(pose.position.y),
                z=float(pose.position.z),
            ),
            orientation=Quaternion(
                x=float(pose.orientation.x),
                y=float(pose.orientation.y),
                z=float(pose.orientation.z),
                w=float(pose.orientation.w),
            ),
        )

    def copy_current_tcp_pose(self) -> Pose | None:
        if self._latest_controller_state is None:
            return None
        return self.copy_pose(self._latest_controller_state.tcp_pose)

    def offset_pose(self, base_pose: Pose, dx: float, dy: float, dz: float) -> Pose:
        pose = self.copy_pose(base_pose)
        pose.position.x = float(pose.position.x + dx)
        pose.position.y = float(pose.position.y + dy)
        pose.position.z = float(pose.position.z + dz)
        return pose

    def lookup_transform_matrix(self, target_frame: str, source_frame: str) -> np.ndarray:
        tf = self._tf_buffer.lookup_transform(target_frame, source_frame, Time())
        t = tf.transform.translation
        q = tf.transform.rotation
        return Pose3D.from_xyz_quat(
            [t.x, t.y, t.z],
            [q.x, q.y, q.z, q.w],
        ).transform_matrix

    def lookup_frame_with_entrance(self, base_frame: str) -> tuple[str, np.ndarray]:
        candidates = [f"{base_frame}_entrance", base_frame]
        last_error = None
        for frame in candidates:
            try:
                return frame, self.lookup_transform_matrix("base_link", frame)
            except TransformException as ex:
                last_error = ex
        raise TransformException(str(last_error))

    def lookup_sfp_port_frame(self, mount_idx: int, port_idx: int) -> tuple[str, np.ndarray]:
        base = f"task_board/nic_card_mount_{mount_idx}/sfp_port_{port_idx}_link"
        return self.lookup_frame_with_entrance(base)

    def lookup_sc_port_frame(self, port_idx: int) -> tuple[str, np.ndarray]:
        frame = f"task_board/sc_port_{port_idx}/sc_port_link"
        return frame, self.lookup_transform_matrix("base_link", frame)

    def lookup_task_board_base_frame(self) -> tuple[str, np.ndarray]:
        candidates = [
            "task_board_base_link",
            "task_board/task_board_base_link",
            "task_board/base_link",
            "task_board",
        ]
        last_error = None
        for frame in candidates:
            try:
                return frame, self.lookup_transform_matrix("base_link", frame)
            except TransformException as ex:
                last_error = ex
        raise TransformException(str(last_error))

    def log_port_tf_diagnostics(self) -> None:
        if self._logged_tf_diagnostics:
            return
        self._logged_tf_diagnostics = True

        frames_text = self._tf_buffer.all_frames_as_string()
        frame_names = sorted(set(re.findall(r"Frame ([^ ]+) exists", frames_text)))
        if self.target == "SFP":
            terms = ("task_board", "nic_card_mount", "sfp_port")
        elif self.target == "SC":
            terms = ("task_board", "sc_port")
        else:
            terms = ("task_board", "task_board_base_link")
        hints = [name for name in frame_names if any(term in name for term in terms)]

        if hints:
            self.get_logger().error(
                f"{self.target} TF lookup failed, but related TF frames exist: "
                + ", ".join(hints[:20])
            )
            return

        if self.target == "SFP":
            self.get_logger().error(
                "No task_board/NIC/SFP TF frames are visible in /tf. "
                "Launch with spawn_task_board:=true, "
                "nic_card_mount_N_present:=true, and ground_truth:=true."
            )
        elif self.target == "SC":
            self.get_logger().error(
                "No task_board/SC TF frames are visible in /tf. "
                "Launch with spawn_task_board:=true, "
                "sc_port_N_present:=true, and ground_truth:=true."
            )
        else:
            self.get_logger().error(
                "No task_board base TF frame is visible in /tf. "
                "Launch with spawn_task_board:=true and ground_truth:=true."
            )

    def discover_sfp_targets(self) -> list[dict]:
        for mount_idx in range(MOUNT_CANDIDATE_COUNT):
            try:
                port_to_base_list = []
                for source_port_idx in [0, 1]:
                    _, port_to_base = self.lookup_sfp_port_frame(
                        mount_idx, source_port_idx
                    )
                    port_to_base_list.append(port_to_base)
            except TransformException:
                continue

            return [{"port_to_base": port_to_base_list}]
        return []

    def discover_sc_targets(self) -> list[dict]:
        targets = []
        for port_idx in range(SC_PORT_CANDIDATE_COUNT):
            try:
                _, port_to_base = self.lookup_sc_port_frame(port_idx)
            except TransformException:
                continue
            targets.append(
                {
                    "port_to_base": [port_to_base],
                    "local_corners": np.array(
                        SC_PORT_CORNER_POINTS_M,
                        dtype=np.float64,
                    ),
                }
            )
        return targets

    def discover_task_board_targets(self) -> list[dict]:
        try:
            _, board_to_base = self.lookup_task_board_base_frame()
        except TransformException:
            return []
        return [
            {
                "port_to_base": [board_to_base],
                "local_points": np.array(
                    TASK_BOARD_KEYPOINT_POINTS_M,
                    dtype=np.float64,
                ),
                "preserve_keypoint_order": True,
            }
        ]

    def discover_targets(self) -> list[dict]:
        if self.target == "SFP":
            return self.discover_sfp_targets()
        if self.target == "SC":
            return self.discover_sc_targets()
        return self.discover_task_board_targets()

    def move_robot_to(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> None:
        w, qx, qy, qz = euler_to_quat(roll, pitch, yaw)
        self.move_robot_to_pose(
            Pose(
                position=Point(x=float(x), y=float(y), z=float(z)),
                orientation=Quaternion(w=w, x=qx, y=qy, z=qz),
            )
        )

    def move_robot_to_pose(self, pose: Pose) -> None:
        stiffness = [200.0, 200.0, 200.0, 50.0, 50.0, 50.0]
        damping = [80.0, 80.0, 80.0, 20.0, 20.0, 20.0]
        msg = MotionUpdate(
            header=Header(
                frame_id="base_link",
                stamp=self.get_clock().now().to_msg(),
            ),
            pose=self.copy_pose(pose),
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

    def project_target_points(
        self,
        target: dict,
        camera_k: np.ndarray,
        base_to_cam: np.ndarray,
        image_w: int,
        image_h: int,
        min_depth: float,
        max_depth: float,
    ) -> list[np.ndarray] | None:
        projected_ports = []
        local_points = target.get("local_points")
        if local_points is None:
            local_points = target.get("local_corners")
        if local_points is None:
            local_points = port_corners_in_frame(self.config["port_size_m"])
        preserve_keypoint_order = target.get(
            "preserve_keypoint_order",
            self.config.get("preserve_keypoint_order", False),
        )

        for port_to_base in target["port_to_base"]:
            projected = []
            for point in local_points:
                point_3d = (port_to_base @ np.append(point, 1.0))[:3]
                uvz = project_to_camera(point_3d, camera_k, base_to_cam)
                if uvz is None:
                    return None
                u, v, depth = uvz
                if depth < min_depth or depth > max_depth:
                    return None
                if u < 0 or u >= image_w or v < 0 or v >= image_h:
                    return None
                projected.append([u, v])
            projected_points = np.array(projected, dtype=np.float64)
            if not preserve_keypoint_order:
                projected_points = order_image_corners(projected_points)
            projected_ports.append(projected_points)
        return projected_ports

    def make_label_for_target(
        self,
        target: dict,
        camera_k: np.ndarray,
        base_to_cam: np.ndarray,
        image_w: int,
        image_h: int,
        min_depth: float,
        max_depth: float,
        bbox_margin: float,
    ) -> str | None:
        projected_ports = self.project_target_points(
            target,
            camera_k,
            base_to_cam,
            image_w,
            image_h,
            min_depth,
            max_depth,
        )
        if projected_ports is None:
            return None

        if self.config["task"] == "pose":
            preserve_keypoint_order = target.get(
                "preserve_keypoint_order",
                self.config.get("preserve_keypoint_order", False),
            )
            if preserve_keypoint_order:
                points = np.vstack(projected_ports)
            else:
                port_points = [
                    (float(np.mean(points[:, 0])), points)
                    for points in projected_ports
                ]
                # Port identity is assigned after projection: left port is port0, right port is port1.
                port_points.sort(key=lambda item: item[0])
                points = np.vstack([ordered_points for _, ordered_points in port_points])
        else:
            points = np.vstack(projected_ports)

        bbox = make_bbox_from_points(points, image_w, image_h, bbox_margin)
        if bbox is None:
            return None

        values = [self.config["class_id"], *bbox]
        if self.config["task"] == "pose":
            normalized_kpts = points.copy()
            normalized_kpts[:, 0] /= image_w
            normalized_kpts[:, 1] /= image_h
            values.extend(normalized_kpts.reshape(-1).tolist())

        return " ".join(
            f"{value:.6f}" if isinstance(value, float) else str(value)
            for value in values
        )

    def collect_one_frame(
        self,
        episode_id: int,
        output_dir: Path,
        split: str,
        stem_prefix: str,
        min_depth: float,
        max_depth: float,
        bbox_margin: float,
        debug_dir: Path | None,
        debug_every: int,
    ) -> bool:
        cam_to_base = {}
        for name, frame in CAMERAS:
            try:
                cam_to_base[name] = self.lookup_transform_matrix("base_link", frame)
            except TransformException as ex:
                self.get_logger().warn(f"Camera TF failed for {name}: {ex}")
                return False

        targets = self.discover_targets()
        if not targets:
            self.get_logger().warn(f"No {self.target} targets found.")
            return False

        saved = False
        for name, _ in CAMERAS:
            image = self._latest_image.get(name)
            if image is None:
                continue

            h, w = image.shape[:2]
            camera_k = np.array(self._cam_info[name].k, dtype=np.float64).reshape(3, 3)
            base_to_cam = np.linalg.inv(cam_to_base[name])
            labels = []

            for target in targets:
                label = self.make_label_for_target(
                    target,
                    camera_k,
                    base_to_cam,
                    w,
                    h,
                    min_depth,
                    max_depth,
                    bbox_margin,
                )
                if label is not None:
                    labels.append(label)

            if not labels:
                continue

            stem = f"{stem_prefix}ep{episode_id:05d}_{name}"
            image_path = output_dir / "images" / split / f"{stem}.jpg"
            label_path = output_dir / "labels" / split / f"{stem}.txt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(image_path), image)
            label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

            if debug_dir and (episode_id % max(1, debug_every) == 0):
                debug_image = image.copy()
                for label in labels:
                    debug_image = draw_label(debug_image, label, self.target)
                debug_path = debug_dir / split / f"{stem}.jpg"
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(debug_path), debug_image)

            saved = True
        return saved

    def collect_magenta_frame(
        self,
        episode_id: int,
        output_dir: Path,
        split: str,
        stem_prefix: str,
        board_pose_id: str,
        trial_group: str,
        board_x: float | None,
        board_y: float | None,
        board_yaw: float | None,
        viewpoint_idx: int,
        viewpoint,
        debug_dir: Path | None,
        debug_every: int,
        hue_min: int,
        hue_max: int,
        sat_min: int,
        val_min: int,
        min_area: float,
    ) -> int:
        saved_count = 0
        view_x = view_y = view_z = view_roll = view_pitch = view_yaw = None
        view_qx = view_qy = view_qz = view_qw = None
        view_dx = view_dy = view_dz = None
        if isinstance(viewpoint, dict):
            view_x = viewpoint.get("x")
            view_y = viewpoint.get("y")
            view_z = viewpoint.get("z")
            view_qx = viewpoint.get("qx")
            view_qy = viewpoint.get("qy")
            view_qz = viewpoint.get("qz")
            view_qw = viewpoint.get("qw")
            view_dx = viewpoint.get("dx")
            view_dy = viewpoint.get("dy")
            view_dz = viewpoint.get("dz")
        elif viewpoint is not None:
            view_x, view_y, view_z, view_roll, view_pitch, view_yaw = viewpoint

        for name, _ in CAMERAS:
            image = self._latest_image.get(name)
            if image is None:
                continue

            detection = self._detect_magenta_feature(
                image,
                hue_min,
                hue_max,
                sat_min,
                val_min,
                min_area,
            )
            if detection is None:
                continue

            stem = f"{stem_prefix}{board_pose_id}_vp{viewpoint_idx:03d}_ep{episode_id:05d}_{name}"
            image_path = output_dir / "images" / split / f"{stem}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), image)

            if debug_dir and (episode_id % max(1, debug_every) == 0):
                x = detection["bbox_x"]
                y = detection["bbox_y"]
                w = detection["bbox_w"]
                h = detection["bbox_h"]
                debug_image = image.copy()
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(
                    debug_image,
                    f"magenta area={detection['area_px']:.0f}",
                    (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 255),
                    2,
                )
                debug_path = debug_dir / split / f"{stem}.jpg"
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(debug_path), debug_image)

            row = {
                "target": self.target,
                "split": split,
                "episode_id": episode_id,
                "camera": name,
                "stem": stem,
                "image_path": str(image_path.relative_to(output_dir)),
                "board_pose_id": board_pose_id,
                "trial_group": trial_group,
                "board_x_m": self._format_optional_float(board_x),
                "board_y_m": self._format_optional_float(board_y),
                "board_yaw_rad": self._format_optional_float(board_yaw),
                "viewpoint_idx": viewpoint_idx,
                "view_x_m": self._format_optional_float(view_x),
                "view_y_m": self._format_optional_float(view_y),
                "view_z_m": self._format_optional_float(view_z),
                "view_roll_rad": self._format_optional_float(view_roll),
                "view_pitch_rad": self._format_optional_float(view_pitch),
                "view_yaw_rad": self._format_optional_float(view_yaw),
                "view_qx": self._format_optional_float(view_qx),
                "view_qy": self._format_optional_float(view_qy),
                "view_qz": self._format_optional_float(view_qz),
                "view_qw": self._format_optional_float(view_qw),
                "view_dx_m": self._format_optional_float(view_dx),
                "view_dy_m": self._format_optional_float(view_dy),
                "view_dz_m": self._format_optional_float(view_dz),
                "magenta_bbox_x": detection["bbox_x"],
                "magenta_bbox_y": detection["bbox_y"],
                "magenta_bbox_w": detection["bbox_w"],
                "magenta_bbox_h": detection["bbox_h"],
                "magenta_center_x": f"{detection['center_x']:.3f}",
                "magenta_center_y": f"{detection['center_y']:.3f}",
                "magenta_area_px": f"{detection['area_px']:.3f}",
                "magenta_candidate_count": detection["candidate_count"],
                "saved_unix_s": f"{time.time():.6f}",
            }
            self._append_metadata_csv(output_dir, row)
            saved_count += 1

        return saved_count
