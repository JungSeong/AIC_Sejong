import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Transform


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
                "x": tfm.translation.x,
                "y": tfm.translation.y,
                "z": tfm.translation.z,
            },
            "rotation": {
                "w": tfm.rotation.w,
                "x": tfm.rotation.x,
                "y": tfm.rotation.y,
                "z": tfm.rotation.z,
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
                    "x": mu.pose.position.x,
                    "y": mu.pose.position.y,
                    "z": mu.pose.position.z,
                },
                "orientation": {
                    "w": mu.pose.orientation.w,
                    "x": mu.pose.orientation.x,
                    "y": mu.pose.orientation.y,
                    "z": mu.pose.orientation.z,
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
        return {
            "left_image": self._save_image(obs.left_image, left_path),
            "center_image": self._save_image(obs.center_image, center_path),
            "right_image": self._save_image(obs.right_image, right_path),
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
                "tcp_error": list(obs.controller_state.tcp_error),
                "target_mode": int(obs.controller_state.target_mode.mode),
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
        # obs.header 가 없는 Observation 버전 대응: stamp 는 기본값(0) 사용
        action.trajectory_generation_mode = TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_UNSPECIFIED
        )
        self.record_step(
            phase=phase,
            task=task,
            obs=obs,
            action=action,
            port_tf=port_tf,
            plug_tf=plug_tf,
            gripper_tf=gripper_tf,
            extras=extras,
        )
