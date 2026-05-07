import json
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Transform
from scipy.spatial.transform import Rotation

try:
    import importlib.util
    _LEROBOT_AVAILABLE = importlib.util.find_spec("lerobot") is not None
except Exception:
    _LEROBOT_AVAILABLE = False

# ──────────────────────────────────────────
# LeRobot dataset 스펙
# ──────────────────────────────────────────
_IMG_H, _IMG_W = 256, 288

LEROBOT_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (35,),
        "names": [
            "tcp_pose.position.x", "tcp_pose.position.y", "tcp_pose.position.z",
            "tcp_pose.orientation.x", "tcp_pose.orientation.y",
            "tcp_pose.orientation.z", "tcp_pose.orientation.w",
            "tcp_velocity.linear.x", "tcp_velocity.linear.y", "tcp_velocity.linear.z",
            "tcp_velocity.angular.x", "tcp_velocity.angular.y", "tcp_velocity.angular.z",
            "tcp_error.x", "tcp_error.y", "tcp_error.z",
            "tcp_error.rx", "tcp_error.ry", "tcp_error.rz",
            "joint_positions.0", "joint_positions.1", "joint_positions.2",
            "joint_positions.3", "joint_positions.4", "joint_positions.5",
            "joint_positions.6",
            "force.x", "force.y", "force.z",
            "torque.x", "torque.y", "torque.z",
            "gripper_offset.x", "gripper_offset.y", "gripper_offset.z",
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": [
            "position.x", "position.y", "position.z",
            "orientation.x", "orientation.y", "orientation.z", "orientation.w",
        ],
    },
    "observation.plug_to_port": {
        "dtype": "float32",
        "shape": (7,),
        "names": [
            "translation.x", "translation.y", "translation.z",
            "rotation.x", "rotation.y", "rotation.z", "rotation.w",
        ],
    },
    "observation.images.left_camera": {
        "dtype": "video",
        "shape": (_IMG_H, _IMG_W, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.center_camera": {
        "dtype": "video",
        "shape": (_IMG_H, _IMG_W, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.right_camera": {
        "dtype": "video",
        "shape": (_IMG_H, _IMG_W, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.scenario_params": {
        "dtype": "float32",
        "shape": (11,),
        "names": [
            "trial_type",       # 0=NIC, 1=SC
            "rail_idx",
            "board_x",
            "board_y",
            "board_yaw",
            "gripper_offset_x",
            "gripper_offset_y",
            "gripper_offset_z",
            "nic_translation",  # SC trial이면 0
            "nic_yaw",          # SC trial이면 0
            "sc_translation",
        ],
    },
    "observation.stiffness": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "rx", "ry", "rz"],
    },
    "observation.damping": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "rx", "ry", "rz"],
    },
    "insertion_success": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
    "phase": {
        "dtype": "string",
        "shape": (1,),
        "names": None,  # "approach" | "collect"
    },
}


# ──────────────────────────────────────────
# 공통 유틸리티
# ──────────────────────────────────────────
def compute_plug_to_port(port_tf: Transform, plug_tf: Transform) -> np.ndarray:
    """plug 포즈를 port 좌표계 기준으로 표현한 상대 포즈 (xyz + quat xyzw)."""
    t_port = np.array([port_tf.translation.x, port_tf.translation.y, port_tf.translation.z])
    q_port = np.array([port_tf.rotation.x, port_tf.rotation.y, port_tf.rotation.z, port_tf.rotation.w])
    t_plug = np.array([plug_tf.translation.x, plug_tf.translation.y, plug_tf.translation.z])
    q_plug = np.array([plug_tf.rotation.x, plug_tf.rotation.y, plug_tf.rotation.z, plug_tf.rotation.w])
    R_port = Rotation.from_quat(q_port)
    R_plug = Rotation.from_quat(q_plug)
    t_rel = R_port.inv().apply(t_plug - t_port)
    q_rel = (R_port.inv() * R_plug).as_quat()
    return np.concatenate([t_rel, q_rel]).astype(np.float32)


def decode_image(image_msg, h: int = _IMG_H, w: int = _IMG_W) -> np.ndarray:
    """ROS Image 메시지 → RGB numpy array (HWC), target 크기로 resize."""
    if image_msg.width == 0 or image_msg.height == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    img = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
        image_msg.height, image_msg.width, 3
    )
    if image_msg.encoding == "rgb8":
        pass  # already RGB
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


# ──────────────────────────────────────────
# LeRobot 직접 저장 recorder
# ──────────────────────────────────────────
class LeRobotRecorder:
    """에피소드 스텝을 LeRobotDataset에 직접 기록."""

    def __init__(self, dataset: "_LeRobotDataset", scenario_params_vec: np.ndarray):
        if not _LEROBOT_AVAILABLE:
            raise RuntimeError("lerobot package is not installed.")
        self.dataset = dataset
        self.scenario_params_vec = scenario_params_vec.astype(np.float32)
        # indices 5-7: gripper_offset x/y/z (observation.state 마지막 3차원과 동일)
        self._gripper_offset_xyz = self.scenario_params_vec[5:8]
        self._episode_frame_count = 0

    def _build_state(self, obs: Observation) -> np.ndarray:
        cs = obs.controller_state
        ww = obs.wrist_wrench.wrench
        state = [
            cs.tcp_pose.position.x, cs.tcp_pose.position.y, cs.tcp_pose.position.z,
            cs.tcp_pose.orientation.x, cs.tcp_pose.orientation.y,
            cs.tcp_pose.orientation.z, cs.tcp_pose.orientation.w,
            cs.tcp_velocity.linear.x, cs.tcp_velocity.linear.y, cs.tcp_velocity.linear.z,
            cs.tcp_velocity.angular.x, cs.tcp_velocity.angular.y, cs.tcp_velocity.angular.z,
            float(cs.tcp_error[0]), float(cs.tcp_error[1]), float(cs.tcp_error[2]),
            float(cs.tcp_error[3]), float(cs.tcp_error[4]), float(cs.tcp_error[5]),
        ]
        for pos in obs.joint_states.position:
            state.append(float(pos))
        state += [
            ww.force.x, ww.force.y, ww.force.z,
            ww.torque.x, ww.torque.y, ww.torque.z,
            float(self._gripper_offset_xyz[0]),
            float(self._gripper_offset_xyz[1]),
            float(self._gripper_offset_xyz[2]),
        ]
        return np.array(state, dtype=np.float32)

    def record_step(
        self,
        phase: str,
        task: Task,
        obs: Observation,
        action: MotionUpdate,
        port_tf: Transform,
        plug_tf: Transform,
        gripper_tf: Transform,   # not used in LeRobot format, kept for signature compat
        extras: dict[str, Any],  # not used in LeRobot format, kept for signature compat
        stiffness: Optional[list[float]] = None,
        damping: Optional[list[float]] = None,
    ) -> None:
        task_name = "sfp_insertion" if "sfp" in task.port_type.lower() else "sc_insertion"

        # Default parameters if not provided (for non-insertion phases)
        _stiffness = np.array(stiffness if stiffness is not None else [0.0]*6, dtype=np.float32)
        _damping = np.array(damping if damping is not None else [0.0]*6, dtype=np.float32)

        self.dataset.add_frame({
            "observation.state": self._build_state(obs),
            "action": np.array([
                action.pose.position.x, action.pose.position.y, action.pose.position.z,
                action.pose.orientation.x, action.pose.orientation.y,
                action.pose.orientation.z, action.pose.orientation.w,
            ], dtype=np.float32),
            "observation.plug_to_port":    compute_plug_to_port(port_tf, plug_tf),
            "observation.scenario_params": self.scenario_params_vec,
            "observation.stiffness": _stiffness,
            "observation.damping": _damping,
            "observation.images.left_camera":   decode_image(obs.left_image),
            "observation.images.center_camera": decode_image(obs.center_image),
            "observation.images.right_camera":  decode_image(obs.right_image),
            "insertion_success": np.array([0], dtype=np.int64),
            "phase": phase,
            "task": task_name,
        })
        self._episode_frame_count += 1

    def record_terminal_step(
        self,
        phase: str,
        task: Task,
        obs: Observation,
        port_tf: Transform,
        plug_tf: Transform,
        gripper_tf: Transform,
        extras: dict[str, Any],
        stiffness: Optional[list[float]] = None,
        damping: Optional[list[float]] = None,
    ) -> None:
        action = MotionUpdate()
        action.header.frame_id = "base_link"
        action.trajectory_generation_mode = TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_UNSPECIFIED
        )
        self.record_step(
            phase=phase, task=task, obs=obs, action=action,
            port_tf=port_tf, plug_tf=plug_tf, gripper_tf=gripper_tf, extras=extras,
            stiffness=stiffness, damping=damping,
        )

    def save_episode(self, insertion_success: bool = False) -> None:
        # Backfill the last frame's insertion_success with the actual outcome.
        # LeRobotDataset buffers frames in memory until save_episode() is called,
        # so we can patch the field directly before flushing.
        if self._episode_frame_count > 0 and insertion_success:
            self.dataset.writer.episode_buffer["insertion_success"][-1] = np.array([1], dtype=np.int64)
        self._episode_frame_count = 0
        self.dataset.save_episode()
