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
    from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset
    _LEROBOT_AVAILABLE = True
except ImportError:
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
# Raw 포맷 recorder (기존 유지)
# ──────────────────────────────────────────
class AutoCaptureRecorder:
    def __init__(self, episode_dir: Path):
        self.episode_dir = episode_dir
        self.steps_path = episode_dir / "steps.jsonl"
        self.step_idx = 0

    def task_to_dict(self, task: Task) -> dict[str, Any]:
        return {
            "id": task.id,
            "cable_type": task.cable_type,
            "cable_name": task.cable_name,
            "plug_type": task.plug_type,
            "plug_name": task.plug_name,
            "port_type": task.port_type,
            "port_name": task.port_name,
            "target_module_name": task.target_module_name,
            "time_limit": int(task.time_limit),
        }

    def write_meta(self, meta: dict[str, Any]) -> None:
        (self.episode_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    def write_summary(self, summary: dict[str, Any]) -> None:
        (self.episode_dir / "episode_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    def transform_to_dict(self, tfm: Transform) -> dict[str, Any]:
        return {
            "translation": {
                "x": tfm.translation.x, "y": tfm.translation.y, "z": tfm.translation.z,
            },
            "rotation": {
                "w": tfm.rotation.w, "x": tfm.rotation.x,
                "y": tfm.rotation.y, "z": tfm.rotation.z,
            },
        }

    def motion_update_to_dict(self, mu: MotionUpdate) -> dict[str, Any]:
        return {
            "header": {
                "frame_id": mu.header.frame_id,
                "stamp": {"sec": mu.header.stamp.sec, "nanosec": mu.header.stamp.nanosec},
            },
            "pose": {
                "position": {
                    "x": mu.pose.position.x, "y": mu.pose.position.y, "z": mu.pose.position.z,
                },
                "orientation": {
                    "w": mu.pose.orientation.w, "x": mu.pose.orientation.x,
                    "y": mu.pose.orientation.y, "z": mu.pose.orientation.z,
                },
            },
            "trajectory_generation_mode": int(mu.trajectory_generation_mode.mode),
        }

    def _save_image(self, image_msg, path: Path) -> dict[str, Any]:
        info = {
            "path": str(path),
            "width": int(image_msg.width),
            "height": int(image_msg.height),
            "encoding": image_msg.encoding,
        }
        if image_msg.width == 0 or image_msg.height == 0:
            return info
        img = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
            image_msg.height, image_msg.width, 3
        )
        if image_msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), img)
        return info

    def observation_to_dict(self, obs: Observation, image_dir: Path) -> dict[str, Any]:
        left_dir = image_dir / "left"
        center_dir = image_dir / "center"
        right_dir = image_dir / "right"
        for d in (left_dir, center_dir, right_dir):
            d.mkdir(parents=True, exist_ok=True)
        left_path = left_dir / f"{self.step_idx:06d}.png"
        center_path = center_dir / f"{self.step_idx:06d}.png"
        right_path = right_dir / f"{self.step_idx:06d}.png"

        cs = obs.controller_state
        state = {
            "tcp_pose.position.x": float(cs.tcp_pose.position.x),
            "tcp_pose.position.y": float(cs.tcp_pose.position.y),
            "tcp_pose.position.z": float(cs.tcp_pose.position.z),
            "tcp_pose.orientation.x": float(cs.tcp_pose.orientation.x),
            "tcp_pose.orientation.y": float(cs.tcp_pose.orientation.y),
            "tcp_pose.orientation.z": float(cs.tcp_pose.orientation.z),
            "tcp_pose.orientation.w": float(cs.tcp_pose.orientation.w),
            "tcp_velocity.linear.x": float(cs.tcp_velocity.linear.x),
            "tcp_velocity.linear.y": float(cs.tcp_velocity.linear.y),
            "tcp_velocity.linear.z": float(cs.tcp_velocity.linear.z),
            "tcp_velocity.angular.x": float(cs.tcp_velocity.angular.x),
            "tcp_velocity.angular.y": float(cs.tcp_velocity.angular.y),
            "tcp_velocity.angular.z": float(cs.tcp_velocity.angular.z),
            "tcp_error.x": float(cs.tcp_error[0]),
            "tcp_error.y": float(cs.tcp_error[1]),
            "tcp_error.z": float(cs.tcp_error[2]),
            "tcp_error.rx": float(cs.tcp_error[3]),
            "tcp_error.ry": float(cs.tcp_error[4]),
            "tcp_error.rz": float(cs.tcp_error[5]),
        }
        for i, pos in enumerate(obs.joint_states.position):
            state[f"joint_positions.{i}"] = float(pos)

        return {
            "left_image": self._save_image(obs.left_image, left_path),
            "center_image": self._save_image(obs.center_image, center_path),
            "right_image": self._save_image(obs.right_image, right_path),
            "state": state,
            "wrist_wrench": {
                "frame_id": obs.wrist_wrench.header.frame_id,
                "force": {
                    "x": obs.wrist_wrench.wrench.force.x,
                    "y": obs.wrist_wrench.wrench.force.y,
                    "z": obs.wrist_wrench.wrench.force.z,
                },
                "torque": {
                    "x": obs.wrist_wrench.wrench.torque.x,
                    "y": obs.wrist_wrench.wrench.torque.y,
                    "z": obs.wrist_wrench.wrench.torque.z,
                },
            },
            "controller_state": {
                "tcp_error": list(cs.tcp_error),
                "target_mode": int(cs.target_mode.mode),
            },
        }

    def record_step(
        self,
        phase: str,
        task: Task,
        obs: Observation,
        action: MotionUpdate,
        port_tf: Transform,
        plug_tf: Transform,
        gripper_tf: Transform,
        extras: dict[str, Any],
    ) -> None:
        image_dir = self.episode_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        lerobot_action = {
            "position.x": float(action.pose.position.x),
            "position.y": float(action.pose.position.y),
            "position.z": float(action.pose.position.z),
            "orientation.x": float(action.pose.orientation.x),
            "orientation.y": float(action.pose.orientation.y),
            "orientation.z": float(action.pose.orientation.z),
            "orientation.w": float(action.pose.orientation.w),
        }

        record = {
            "step": self.step_idx,
            "time": time.time(),
            "phase": phase,
            "task": self.task_to_dict(task),
            "transforms": {
                "port": self.transform_to_dict(port_tf),
                "plug": self.transform_to_dict(plug_tf),
                "gripper": self.transform_to_dict(gripper_tf),
            },
            "observation": self.observation_to_dict(obs, image_dir),
            "action": self.motion_update_to_dict(action),
            "lerobot_action": lerobot_action,
            "extras": extras,
        }
        with self.steps_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.step_idx += 1

    def record_terminal_step(
        self,
        phase: str,
        task: Task,
        obs: Observation,
        port_tf: Transform,
        plug_tf: Transform,
        gripper_tf: Transform,
        extras: dict[str, Any],
    ) -> None:
        action = MotionUpdate()
        action.header.frame_id = "base_link"
        action.trajectory_generation_mode = TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_UNSPECIFIED
        )
        self.record_step(
            phase=phase, task=task, obs=obs, action=action,
            port_tf=port_tf, plug_tf=plug_tf, gripper_tf=gripper_tf, extras=extras,
        )


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
    ) -> None:
        task_idx = 0 if "sfp" in task.port_type.lower() else 1

        self.dataset.add_frame({
            "observation.state": self._build_state(obs),
            "action": np.array([
                action.pose.position.x, action.pose.position.y, action.pose.position.z,
                action.pose.orientation.x, action.pose.orientation.y,
                action.pose.orientation.z, action.pose.orientation.w,
            ], dtype=np.float32),
            "observation.plug_to_port":    compute_plug_to_port(port_tf, plug_tf),
            "observation.scenario_params": self.scenario_params_vec,
            "observation.images.left_camera":   decode_image(obs.left_image),
            "observation.images.center_camera": decode_image(obs.center_image),
            "observation.images.right_camera":  decode_image(obs.right_image),
            "task": task_idx,
        })

    def record_terminal_step(
        self,
        phase: str,
        task: Task,
        obs: Observation,
        port_tf: Transform,
        plug_tf: Transform,
        gripper_tf: Transform,
        extras: dict[str, Any],
    ) -> None:
        action = MotionUpdate()
        action.header.frame_id = "base_link"
        action.trajectory_generation_mode = TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_UNSPECIFIED
        )
        self.record_step(
            phase=phase, task=task, obs=obs, action=action,
            port_tf=port_tf, plug_tf=plug_tf, gripper_tf=gripper_tf, extras=extras,
        )

    def save_episode(self) -> None:
        self.dataset.save_episode()
