import os
from huggingface_hub import snapshot_download
import time
import json
import torch
import numpy as np
import cv2
import draccus
from pathlib import Path
from typing import Dict
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3, Wrench

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

from aic_control_interfaces.msg import (
    MotionUpdate,
    TrajectoryGenerationMode,
)

# LeRobot & Safetensors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from safetensors.torch import load_file


class Baseline(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── [1/5] 경로 확인 ───────────────────────────────────
        # snapshot_download: HF 캐시(~/.cache/huggingface/hub/)에서 자동으로 경로를 반환.
        # CWD에 무관하게 동작하며, 이미 다운로드된 경우 캐시를 그대로 사용.
        self.get_logger().info("[MODEL LOAD 1/5] Resolving model path via HuggingFace cache ...")
        abs_path = Path(
            snapshot_download(
                repo_id="aic-sejong-team/act_AIC",
                allow_patterns=["config.json", "*.safetensors"],
            )
        )
        self.get_logger().info(f"[MODEL LOAD 1/5] Resolved    : {abs_path}")
        self.get_logger().info(
            f"[MODEL LOAD 1/5] Directory OK. Files: {[f.name for f in abs_path.iterdir()]}"
        )

        # ── [2/5] config.json ─────────────────────────────────
        self.get_logger().info("[MODEL LOAD 2/5] Loading config.json ...")
        with open(abs_path / "config.json", "r") as f:
            config_dict = json.load(f)
            config_dict.pop("type", None)
        config = draccus.decode(ACTConfig, config_dict)
        self.get_logger().info(
            f"[MODEL LOAD 2/5] Config OK. chunk_size={getattr(config, 'chunk_size', '?')}"
        )

        # ── [3/5] 가중치 ──────────────────────────────────────
        self.get_logger().info(
            f"[MODEL LOAD 3/5] Loading model.safetensors on {self.device} ..."
        )
        self.policy = ACTPolicy(config)
        self.policy.load_state_dict(load_file(abs_path / "model.safetensors"))
        self.policy.eval()
        self.policy.to(self.device)
        param_count = sum(p.numel() for p in self.policy.parameters())
        self.get_logger().info(f"[MODEL LOAD 3/5] Weights OK. Params: {param_count:,}")

        # ── [4/5] 정규화 통계 ─────────────────────────────────
        stats_file = (
            abs_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )
        self.get_logger().info(
            f"[MODEL LOAD 4/5] Loading normalization stats from {stats_file.name} ..."
        )
        stats = load_file(stats_file)
        self.get_logger().info(f"[MODEL LOAD 4/5] Stats keys: {list(stats.keys())}")

        def get_stat(key, shape):
            if key not in stats:
                raise KeyError(
                    f"Key '{key}' not found in stats. Available: {list(stats.keys())}"
                )
            return stats[key].to(self.device).view(*shape)

        self.img_stats = {
            "left": {
                "mean": get_stat("observation.images.left_camera.mean",   (1, 3, 1, 1)),
                "std":  get_stat("observation.images.left_camera.std",    (1, 3, 1, 1)),
            },
            "center": {
                "mean": get_stat("observation.images.center_camera.mean", (1, 3, 1, 1)),
                "std":  get_stat("observation.images.center_camera.std",  (1, 3, 1, 1)),
            },
            "right": {
                "mean": get_stat("observation.images.right_camera.mean",  (1, 3, 1, 1)),
                "std":  get_stat("observation.images.right_camera.std",   (1, 3, 1, 1)),
            },
        }
        self.state_mean  = get_stat("observation.state.mean", (1, -1))
        self.state_std   = get_stat("observation.state.std",  (1, -1))
        self.action_mean = get_stat("action.mean", (1, -1))
        self.action_std  = get_stat("action.std",  (1, -1))
        self.get_logger().info(
            f"[MODEL LOAD 4/5] Stats OK. "
            f"state_dim={self.state_mean.shape}, action_dim={self.action_mean.shape}"
        )

        self.image_scale = 0.25

        # ── [5/5] 완료 ────────────────────────────────────────
        self.get_logger().info("=" * 50)
        self.get_logger().info("[MODEL LOAD 5/5] ★ MODEL READY ★")
        self.get_logger().info(f"  device     : {self.device}")
        self.get_logger().info(f"  params     : {param_count:,}")
        self.get_logger().info(f"  state_dim  : {self.state_mean.shape[-1]}")
        self.get_logger().info(f"  action_dim : {self.action_mean.shape[-1]}")
        self.get_logger().info(f"  img_scale  : {self.image_scale}")
        self.get_logger().info("=" * 50)

    # ── 이미지 전처리 ─────────────────────────────────────────────
    @staticmethod
    def _img_to_tensor(
        raw_img,
        device: torch.device,
        scale: float,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )
        if scale != 1.0:
            img_np = cv2.resize(img_np, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
        tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )
        return (tensor - mean) / std

    # ── 관측값 전처리 ─────────────────────────────────────────────
    def prepare_observations(self, obs_msg: Observation) -> Dict[str, torch.Tensor]:
        obs = {
            "observation.images.left_camera": self._img_to_tensor(
                obs_msg.left_image, self.device, self.image_scale,
                self.img_stats["left"]["mean"], self.img_stats["left"]["std"],
            ),
            "observation.images.center_camera": self._img_to_tensor(
                obs_msg.center_image, self.device, self.image_scale,
                self.img_stats["center"]["mean"], self.img_stats["center"]["std"],
            ),
            "observation.images.right_camera": self._img_to_tensor(
                obs_msg.right_image, self.device, self.image_scale,
                self.img_stats["right"]["mean"], self.img_stats["right"]["std"],
            ),
        }

        tcp  = obs_msg.controller_state.tcp_pose
        vel  = obs_msg.controller_state.tcp_velocity
        state_np = np.array([
            tcp.position.x,    tcp.position.y,    tcp.position.z,
            tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w,
            vel.linear.x,  vel.linear.y,  vel.linear.z,
            vel.angular.x, vel.angular.y, vel.angular.z,
            *obs_msg.controller_state.tcp_error,
            *obs_msg.joint_states.position[:7],
        ], dtype=np.float32)

        raw_state = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        obs["observation.state"] = (raw_state - self.state_mean) / self.state_std
        return obs

    # ── MotionUpdate 생성 헬퍼 ────────────────────────────────────
    def set_cartesian_twist_target(self, twist: Twist, frame_id: str = "base_link") -> MotionUpdate:
        msg = MotionUpdate()
        msg.velocity = twist
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.target_stiffness = np.diag([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]).flatten()
        msg.target_damping   = np.diag([40.0,  40.0,  40.0,  15.0, 15.0, 15.0]).flatten()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        return msg

    # ── 메인 루프 ─────────────────────────────────────────────────
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self.policy.reset()
        self.get_logger().info(f"Baseline.insert_cable() start. Task: {task}")

        MAX_LINEAR_VEL  = 0.05   # m/s
        MAX_ANGULAR_VEL = 0.3    # rad/s

        start_time = time.time()
        step = 0

        while time.time() - start_time < 30.0:
            loop_start = time.time()

            obs_msg = get_observation()
            if obs_msg is None:
                self.get_logger().info("No observation received.")
                continue

            obs_tensors = self.prepare_observations(obs_msg)

            with torch.inference_mode():
                normalized_action = self.policy.select_action(obs_tensors)

            action = ((normalized_action * self.action_std) + self.action_mean)[0].cpu().numpy()

            if not np.isfinite(action).all():
                self.get_logger().error(f"[step {step}] Non-finite action: {action} — skipping")
                step += 1
                continue

            if step < 5:
                self.get_logger().info(
                    f"[step {step}] lin=({action[0]:.4f},{action[1]:.4f},{action[2]:.4f}) "
                    f"ang=({action[3]:.4f},{action[4]:.4f},{action[5]:.4f})"
                )

            lin = np.clip(action[:3],  -MAX_LINEAR_VEL,  MAX_LINEAR_VEL)
            ang = np.clip(action[3:6], -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)

            if not (np.allclose(lin, action[:3]) and np.allclose(ang, action[3:6])):
                self.get_logger().warn(
                    f"[step {step}] Clamped: {action[:6]} → {np.concatenate([lin, ang])}"
                )

            twist = Twist(
                linear=Vector3(x=float(lin[0]), y=float(lin[1]), z=float(lin[2])),
                angular=Vector3(x=float(ang[0]), y=float(ang[1]), z=float(ang[2])),
            )
            move_robot(motion_update=self.set_cartesian_twist_target(twist))
            send_feedback("in progress...")

            step += 1
            time.sleep(max(0, 0.25 - (time.time() - loop_start)))

        self.get_logger().info(f"Baseline.insert_cable() done. {step} steps in 30s.")
        return True
