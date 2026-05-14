"""Stage 0 pre-alignment controller for the staged policy."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import transforms3d.quaternions as qmath
from geometry_msgs.msg import Pose, Quaternion, Transform, Vector3
from motion_planning_node.core.geometry import quat_to_mat
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from motion_planning_node.core.config import VisionConfig, Stage0Config, Stage1Config
from motion_planning_node.core.geometry import pose_to_matrix

def _get_ws_src_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "aic").is_dir() and (parent / "ais").is_dir():
            return parent
    return Path(__file__).resolve().parents[4]


class Stage0PreAlign:
    """Uses YOLO pose to align the gripper's z-axis with the port's normal vector."""

    def __init__(self, policy, debug_video=None):
        """Initialize Stage 0 and load the YOLO pose model."""
        self._policy = policy
        self._debug_video = debug_video
        
        # --- File logger setup ---
        log_dir = _get_ws_src_dir() / "debug" / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"stage0_{timestamp}.log"
        self._file_logger = logging.getLogger(f"stage0_{timestamp}")
        self._file_logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self._file_logger.addHandler(fh)
        self._file_logger.info(f"Stage0 log started: {log_path}")
        
        # Load YOLO model from pixi environment
        _pixi_site = _get_ws_src_dir() / "aic" / ".pixi" / "envs" / "default" / "lib" / "python3.12" / "site-packages"
        if _pixi_site.exists() and str(_pixi_site) not in sys.path:
            sys.path.insert(0, str(_pixi_site))
            
        self.pose_model_path = Stage0Config.POSE_MODEL_PATH
        self.pose_model = None
        try:
            from ultralytics import YOLO
            self.pose_model = YOLO(self.pose_model_path)
            self._log(f"Loaded YOLO pose model from {self.pose_model_path}")
        except Exception as e:
            self._log(f"Failed to load pose model: {e}", level="warn")

    @property
    def _task(self):
        return self._policy._task

    def get_logger(self):
        return self._policy.get_logger()

    def _log(self, msg: str, level: str = "info"):
        """Log to both ROS logger and debug log file."""
        ros_logger = self.get_logger()
        if level == "warn":
            ros_logger.warn(msg)
            self._file_logger.warning(msg)
        elif level == "error":
            ros_logger.error(msg)
            self._file_logger.error(msg)
        else:
            ros_logger.info(msg)
            self._file_logger.info(msg)

    def sleep_for(self, duration_sec: float) -> None:
        self._policy.sleep_for(duration_sec)

    def time_now(self):
        return self._policy.time_now()

    def set_pose_target(self, *args, **kwargs):
        return self._policy.set_pose_target(*args, **kwargs)

    @staticmethod
    def _position_error(current_pose: Pose, target_pose: Pose) -> float:
        current_position = np.array([
            current_pose.position.x,
            current_pose.position.y,
            current_pose.position.z,
        ], dtype=float)
        target_position = np.array([
            target_pose.position.x,
            target_pose.position.y,
            target_pose.position.z,
        ], dtype=float)
        return float(np.linalg.norm(current_position - target_position))

    @staticmethod
    def _orientation_error(current_pose: Pose, target_pose: Pose) -> float:
        current_quat = np.array([
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w,
        ], dtype=float)
        target_quat = np.array([
            target_pose.orientation.x,
            target_pose.orientation.y,
            target_pose.orientation.z,
            target_pose.orientation.w,
        ], dtype=float)
        current_norm = np.linalg.norm(current_quat)
        target_norm = np.linalg.norm(target_quat)
        if current_norm < 1e-9 or target_norm < 1e-9:
            return float("inf")
        current_quat /= current_norm
        target_quat /= target_norm
        quat_dot = abs(float(np.dot(current_quat, target_quat)))
        quat_dot = min(1.0, max(-1.0, quat_dot))
        return float(2.0 * np.arccos(quat_dot))

    def _wait_until_target_pose(
        self,
        get_observation: GetObservationCallback,
        target_pose: Pose,
        check_position: bool = True,
    ) -> bool:
        start = self.time_now()
        last_position_error = float("inf")
        last_orientation_error = float("inf")
        while (self.time_now() - start).nanoseconds / 1e9 < Stage0Config.POSE_WAIT_TIMEOUT_SEC:
            obs = get_observation()
            current_pose = getattr(getattr(obs, "controller_state", None), "tcp_pose", None)
            if current_pose is not None:
                last_position_error = self._position_error(current_pose, target_pose)
                last_orientation_error = self._orientation_error(current_pose, target_pose)
                if (
                    (not check_position or last_position_error <= Stage0Config.POSITION_TOLERANCE_M)
                    and last_orientation_error <= Stage0Config.ORIENTATION_TOLERANCE_RAD
                ):
                    self.get_logger().info(
                        f"Stage 0 target reached: "
                        f"pos_err={last_position_error:.4f}m, "
                        f"rot_err={last_orientation_error:.4f}rad"
                    )
                    return True
            self.sleep_for(Stage0Config.POSE_WAIT_DT_SEC)
        self.get_logger().warn(
            f"Stage 0 target wait timeout: "
            f"pos_err={last_position_error:.4f}m, "
            f"rot_err={last_orientation_error:.4f}rad"
        )
        return False

    def _write_pose_debug_frame(
        self,
        bgr: np.ndarray,
        result,
        target_cls: int,
        best_box_idx: int = -1,
        keypoints: Optional[np.ndarray] = None,
        status: str = "",
        target_z: Optional[np.ndarray] = None,
    ) -> None:
        if self._debug_video is None or bgr is None:
            return
        annotated = bgr.copy()
        class_names = {0: "sfp_port", 1: "sc_port"}
        if result is not None and len(result) > 0:
            boxes = getattr(result[0], "boxes", None)
            if boxes is not None:
                for i, box in enumerate(boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    is_selected = i == best_box_idx
                    color = (0, 255, 0) if cls == target_cls else (0, 0, 255)
                    thickness = 3 if is_selected else 1
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                    label = f"{class_names.get(cls, 'unknown')} {conf:.2f}"
                    if is_selected:
                        label += " selected"
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )
        if keypoints is not None:
            # 클래스별 색상: 0:청록, 1:빨강, 2:파랑, 3:초록, 4:보라
            KPT_COLORS = [
                (0, 220, 255),  # 0: Center
                (0, 0, 255),    # 1: Top-Left
                (255, 0, 0),    # 2: Top-Right
                (0, 255, 0),    # 3: Bottom-Right
                (255, 0, 255),  # 4: Bottom-Left
            ]
            for idx, (u, v) in enumerate(keypoints[:5]):
                center = (int(round(u)), int(round(v)))
                color = KPT_COLORS[idx] if idx < len(KPT_COLORS) else (255, 255, 255)
                cv2.circle(annotated, center, 5, color, -1)
                cv2.putText(
                    annotated,
                    str(idx),
                    (center[0] + 6, center[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
        lines = [
            f"target={class_names.get(target_cls, target_cls)} status={status}",
        ]
        if target_z is not None:
            lines.append(
                f"normal_base=({target_z[0]:+.3f}, {target_z[1]:+.3f}, {target_z[2]:+.3f})"
            )
        self._debug_video.write("Stage 0 pose estimation", annotated, lines)

    @staticmethod
    def _image_msg_to_bgr(img_msg):
        if img_msg is None or img_msg.width == 0 or img_msg.height == 0:
            return None
        img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, 3
        )
        if img_msg.encoding == "rgb8":
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img.copy()

    def _load_port_rpy(self) -> tuple[float, float, float]:
        """Config YAML에서 task board와 port mount의 roll, pitch, yaw를 회전 행렬로 곱하여 최종 RPY를 반환한다."""
        import yaml
        import transforms3d.euler as euler
        config_candidates = [
            Path("/tmp/aic_custom_config.yaml"),
            _get_ws_src_dir() / "aic" / "aic_engine" / "config" / "sample_config.yaml",
        ]
        for cp in config_candidates:
            if cp.is_file():
                with open(cp) as f:
                    config_data = yaml.safe_load(f)
                self._log(f"  Config loaded from: {cp}")

                task_port = self._task.port_name if self._task else ""
                task_module = self._task.target_module_name if self._task else ""

                for trial_key, trial in config_data.get("trials", {}).items():
                    scene = trial.get("scene", {})
                    tasks = trial.get("tasks", {})
                    for tk, tv in tasks.items():
                        if tv.get("port_name") == task_port or tv.get("target_module_name") == task_module:
                            tb = scene.get("task_board", {})
                            board_roll = float(tb.get("pose", {}).get("roll", 0.0))
                            board_pitch = float(tb.get("pose", {}).get("pitch", 0.0))
                            board_yaw = float(tb.get("pose", {}).get("yaw", 0.0))
                            
                            mount_roll, mount_pitch, mount_yaw = 0.0, 0.0, 0.0
                            for mk, mv in tb.items():
                                if isinstance(mv, dict) and mv.get("entity_present"):
                                    if task_module and task_module in str(mv.get("entity_name", "")):
                                        ep = mv.get("entity_pose", {})
                                        mount_roll = float(ep.get("roll", 0.0))
                                        mount_pitch = float(ep.get("pitch", 0.0))
                                        mount_yaw = float(ep.get("yaw", 0.0))
                                        break
                                        
                            R_board = euler.euler2mat(board_roll, board_pitch, board_yaw, axes='sxyz')
                            R_mount = euler.euler2mat(mount_roll, mount_pitch, mount_yaw, axes='sxyz')
                            
                            R_total = R_board @ R_mount
                            total_roll, total_pitch, total_yaw = euler.mat2euler(R_total, axes='sxyz')
                            
                            self._log(f"  Port RPY: [{total_roll:.4f}, {total_pitch:.4f}, {total_yaw:.4f}] rad")
                            return total_roll, total_pitch, total_yaw
                break
        self._log("  Config not found, using default RPY=(0, 0, π)", level="warn")
        return 0.0, 0.0, np.pi

    def run(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        """Stage 0: pre-alignment.

        그리퍼를 아래로 향하게 유지하면서, 포트의 yaw 방향에 맞춰
        Z축 기준으로만 회전한다. (케이블은 중력으로 자연스럽게 아래로 늘어짐)
        """
        self._log("━━━ Stage 0: Pre-Align 시작 ━━━")
        send_feedback("Stage 0: Pre-alignment")

        obs = get_observation()
        if obs is None:
            return False
        
        tcp_pose = obs.controller_state.tcp_pose
        T_base_tcp = pose_to_matrix(tcp_pose)
        R_base_tcp = T_base_tcp[:3, :3]

        from transforms3d._gohlketransforms import quaternion_multiply
        import transforms3d.euler as t3d_euler

        # ── 1. base_link 기준 gripper orientation 추출──
        q_gripper = (
            tcp_pose.orientation.w,
            tcp_pose.orientation.x,
            tcp_pose.orientation.y,
            tcp_pose.orientation.z,
        ) # OK

        # ── 2. Config에서 port의 world RPY 계산 및 q_port 생성 ──
        yaml_roll, yaml_pitch, yaml_yaw = self._load_port_rpy() # port_mount의 RPY 값을 반환 # OK
        
        target_cls = 1 if "sc" in (self._task.plug_name or "").lower() else 0
        
        # 기본 YAML 자세 회전 행렬
        R_yaml = t3d_euler.euler2mat(yaml_roll, yaml_pitch, yaml_yaw, axes='sxyz')
        
        if target_cls == 1:
            R_yaw_offset = t3d_euler.euler2mat(0.0, 0.0, 1.57, axes='sxyz')
            R_roll_offset = t3d_euler.euler2mat(1.57, 0.0, 0.0, axes='sxyz')
            R_port = R_yaw_offset @ R_yaml @ R_roll_offset
        else:
            R_sfp_offset = t3d_euler.euler2mat(3.12895, 0.0, 0.0, axes='sxyz')
            R_port = R_yaml @ R_sfp_offset

        q_port_mat = qmath.mat2quat(R_port)
        q_port = (
            float(q_port_mat[0]),
            float(q_port_mat[1]),
            float(q_port_mat[2]),
            float(q_port_mat[3]),
        )

        # ── 3. 벡터 정렬 방식 (Vector Alignment) ──
        # Full 3D 회전(Roll, Pitch, Yaw 모두 일치)을 강제하면 로봇 손목이 180도 뒤틀리는 특이점(Singularity)에 빠짐.
        # 따라서, "케이블이 찌르는 방향(Z축)"이 "포트 구멍 방향(Z축)"과 평행해지도록 '최단 거리 회전(Minimal Rotation)'만 수행함.

        # a) 케이블이 그리퍼에 물려있는 로컬 방향(Z축) 계산
        offset_roll, offset_pitch, offset_yaw = Stage0Config.CABLE_GRIPPER_OFFSET_RPY
        R_offset = t3d_euler.euler2mat(offset_roll, offset_pitch, offset_yaw, axes='sxyz')
        cable_z_local = R_offset[:, 2] # 케이블의 Z축이 그리퍼 프레임 내에서 어디를 향하는가?
        
        # b) 현재 우주(World) 공간에서 케이블 Z축의 방향
        cable_z_world = R_base_tcp @ cable_z_local
        
        # c) 목표하는 포트 구멍의 방향 (Z축)
        port_z_world = R_port[:, 2]
        
        # d) 케이블 Z축을 포트 Z축으로 맞추기 위한 회전 축과 각도 계산
        axis = np.cross(cable_z_world, port_z_world)
        axis_len = np.linalg.norm(axis)
        
        if axis_len > 1e-6:
            axis /= axis_len
            # 두 벡터 사이의 각도
            angle = np.arccos(np.clip(np.dot(cable_z_world, port_z_world), -1.0, 1.0))
            
            # 회전 축과 각도를 쿼터니언으로 변환
            sin_a = np.sin(angle / 2.0)
            q_rot = (np.cos(angle / 2.0), axis[0] * sin_a, axis[1] * sin_a, axis[2] * sin_a)
            
            # 현재 그리퍼 자세에 회전 적용
            q_gripper_target = quaternion_multiply(q_rot, q_gripper)
        else:
            # 이미 정렬되어 있음
            q_gripper_target = q_gripper

        # ── 4. Quaternion 변환 & pose command ──
        target_pose = Pose()
        target_pose.position = tcp_pose.position  # 위치는 유지
        target_pose.orientation.w = float(q_gripper_target[0])
        target_pose.orientation.x = float(q_gripper_target[1])
        target_pose.orientation.y = float(q_gripper_target[2])
        target_pose.orientation.z = float(q_gripper_target[3])

        self._log(
            f"[Stage0 진단]\n"
            f"  목표 port yaw:    {np.degrees(yaml_yaw):+.2f}°\n"
            f"  q_gripper_target: {q_gripper_target}"
        )

        from transforms3d._gohlketransforms import quaternion_slerp
        from motion_planning_node.core.geometry import interp_profile

        self._log("Rotating gripper to match port smoothly...")
        
        N_STEPS = 20
        for i in range(N_STEPS):
            # 5차 Hermite 보간으로 부드러운 가속/감속 궤적 생성
            t_smooth = interp_profile((i + 1) / float(N_STEPS), quintic=True)
            q_interp = quaternion_slerp(q_gripper, q_gripper_target, t_smooth)
            
            interp_pose = Pose()
            interp_pose.position = tcp_pose.position  # 위치는 유지
            interp_pose.orientation.w = float(q_interp[0])
            interp_pose.orientation.x = float(q_interp[1])
            interp_pose.orientation.y = float(q_interp[2])
            interp_pose.orientation.z = float(q_interp[3])
            
            self.set_pose_target(
                move_robot=move_robot,
                pose=interp_pose,
                stiffness=list(Stage0Config.STIFFNESS),
                damping=list(Stage0Config.DAMPING),
            )
            self.sleep_for(Stage1Config.STEP_SLEEP_SEC)

        if not self._wait_until_target_pose(
            get_observation,
            target_pose,
            check_position=False,
        ):
            return False

        # ── 5. 진단 로그: 회전 후 상태 ──
        obs_after = get_observation()
        if obs_after is not None:
            tcp_after = obs_after.controller_state.tcp_pose
            T_after = pose_to_matrix(tcp_after)
            R_after = T_after[:3, :3]
            achieved_rpy = t3d_euler.mat2euler(R_after, axes='sxyz')
            achieved_z = R_after[:, 2]
            yaw_error = np.degrees(achieved_rpy[2] - yaml_yaw)
            self._log(
                f"[Stage0 진단] 회전 후:\n"
                f"  달성된 RPY (deg): R={np.degrees(achieved_rpy[0]):+.2f}, "
                f"P={np.degrees(achieved_rpy[1]):+.2f}, "
                f"Y={np.degrees(achieved_rpy[2]):+.2f}\n"
                f"  달성된 TCP Z축:   [{achieved_z[0]:+.4f}, {achieved_z[1]:+.4f}, {achieved_z[2]:+.4f}]\n"
                f"  yaw 잔여 오차:    {yaw_error:+.2f}°"
            )

        self._log("Stage 0 완료")
        return True
