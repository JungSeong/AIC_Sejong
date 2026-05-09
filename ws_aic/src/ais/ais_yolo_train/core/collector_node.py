import re
import sys
from pathlib import Path

import cv2
import numpy as np
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
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

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._motion_pub = self.create_publisher(
            MotionUpdate, "/aic_controller/pose_commands", 10
        )
        self._logged_tf_diagnostics = False
        self.get_logger().info(f"{self.target} approach YOLO collector ready.")

    def _on_info(self, name: str, msg: CameraInfo) -> None:
        self._cam_info[name] = msg

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

    def log_port_tf_diagnostics(self) -> None:
        if self._logged_tf_diagnostics:
            return
        self._logged_tf_diagnostics = True

        frames_text = self._tf_buffer.all_frames_as_string()
        frame_names = sorted(set(re.findall(r"Frame ([^ ]+) exists", frames_text)))
        if self.target == "SFP":
            terms = ("task_board", "nic_card_mount", "sfp_port")
        else:
            terms = ("task_board", "sc_port")
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
        else:
            self.get_logger().error(
                "No task_board/SC TF frames are visible in /tf. "
                "Launch with spawn_task_board:=true, "
                "sc_port_N_present:=true, and ground_truth:=true."
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

    def discover_targets(self) -> list[dict]:
        if self.target == "SFP":
            return self.discover_sfp_targets()
        return self.discover_sc_targets()

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
        stiffness = [200.0, 200.0, 200.0, 50.0, 50.0, 50.0]
        damping = [80.0, 80.0, 80.0, 20.0, 20.0, 20.0]
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
        local_corners = target.get("local_corners")
        if local_corners is None:
            local_corners = port_corners_in_frame(self.config["port_size_m"])

        for port_to_base in target["port_to_base"]:
            projected = []
            for corner in local_corners:
                point_3d = (port_to_base @ np.append(corner, 1.0))[:3]
                uvz = project_to_camera(point_3d, camera_k, base_to_cam)
                if uvz is None:
                    return None
                u, v, depth = uvz
                if depth < min_depth or depth > max_depth:
                    return None
                if u < 0 or u >= image_w or v < 0 or v >= image_h:
                    return None
                projected.append([u, v])
            projected_ports.append(order_image_corners(np.array(projected, dtype=np.float64)))
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
