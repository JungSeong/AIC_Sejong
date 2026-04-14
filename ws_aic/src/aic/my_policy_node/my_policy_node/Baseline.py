import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import json
import torch
import numpy as np
import cv2
import draccus
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import snapshot_download  # HF 사용 시

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from geometry_msgs.msg import Twist, Vector3, Wrench

# 사용할 정책에 맞게 import 변경
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig


class Baseline(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── 1. 파일 경로 ──────────────────────────────────────────
        # (A) HuggingFace에서 다운로드
        policy_path = Path(snapshot_download(repo_id="JungSeong2/act_AIC"))

        # (B) 로컬 체크포인트 사용 시
        # policy_path = Path("/home/vsc/LLM_TUNE/AIC_Sejong/aic_data/outputs/train/act_cnn_test/checkpoints/last/pretrained_model")

        # ── 2. config.json 로드 ───────────────────────────────────
        with open(policy_path / "config.json") as f:
            config_dict = json.load(f)
            config_dict.pop("type", None)  # draccus 오류 방지
        config = draccus.decode(ACTConfig, config_dict)

        # ── 3. 가중치 로드 ────────────────────────────────────────
        self.policy = ACTPolicy(config)
        self.policy.load_state_dict(load_file(policy_path / "model.safetensors"))
        self.policy.eval().to(self.device)

        # ── 4. 정규화 통계 로드 ───────────────────────────────────
        stats = load_file(
            policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )

        def stat(key, shape):
            return stats[key].to(self.device).view(*shape)

        self.img_stats = {
            "left":   {"mean": stat("observation.images.left_camera.mean",   (1,3,1,1)),
                       "std":  stat("observation.images.left_camera.std",    (1,3,1,1))},
            "center": {"mean": stat("observation.images.center_camera.mean", (1,3,1,1)),
                       "std":  stat("observation.images.center_camera.std",  (1,3,1,1))},
            "right":  {"mean": stat("observation.images.right_camera.mean",  (1,3,1,1)),
                       "std":  stat("observation.images.right_camera.std",   (1,3,1,1))},
        }
        self.state_mean = stat("observation.state.mean", (1, -1))
        self.state_std  = stat("observation.state.std",  (1, -1))
        self.action_mean = stat("action.mean", (1, -1))
        self.action_std  = stat("action.std",  (1, -1))

        self.image_scale = 0.25  # 훈련 시 사용한 스케일과 반드시 동일하게

    # ── 이미지 전처리 ─────────────────────────────────────────────
    def _img_to_tensor(self, raw_img, mean, std):
        img = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )
        if self.image_scale != 1.0:
            img = cv2.resize(img, None, fx=self.image_scale, fy=self.image_scale,
                             interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(img).permute(2,0,1).float().div(255.0).unsqueeze(0).to(self.device)
        return (t - mean) / std

    # ── 관측값 전처리 ─────────────────────────────────────────────
    def prepare_obs(self, obs_msg):
        tcp   = obs_msg.controller_state.tcp_pose
        vel   = obs_msg.controller_state.tcp_velocity
        state = np.array([
            tcp.position.x, tcp.position.y, tcp.position.z,
            tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w,
            vel.linear.x,  vel.linear.y,  vel.linear.z,
            vel.angular.x, vel.angular.y, vel.angular.z,
            *obs_msg.controller_state.tcp_error,           # 6차원
            *obs_msg.joint_states.position[:7],            # 7차원
        ], dtype=np.float32)

        raw_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        return {
            "observation.images.left_camera":   self._img_to_tensor(obs_msg.left_image,   self.img_stats["left"]["mean"],   self.img_stats["left"]["std"]),
            "observation.images.center_camera": self._img_to_tensor(obs_msg.center_image, self.img_stats["center"]["mean"], self.img_stats["center"]["std"]),
            "observation.images.right_camera":  self._img_to_tensor(obs_msg.right_image,  self.img_stats["right"]["mean"],  self.img_stats["right"]["std"]),
            "observation.state": (raw_t - self.state_mean) / self.state_std,
        }

    # ── 메인 루프 ─────────────────────────────────────────────────
    def insert_cable(self, task: Task, get_observation, move_robot, send_feedback):
        self.policy.reset()

        import time
        start = time.time()
        while time.time() - start < 30.0:
            loop_start = time.time()

            obs_msg = get_observation()
            if obs_msg is None:
                continue

            obs = self.prepare_obs(obs_msg)

            with torch.inference_mode():
                norm_action = self.policy.select_action(obs)  # [1, 7]

            action = ((norm_action * self.action_std) + self.action_mean)[0].cpu().numpy()

            twist = Twist(
                linear=Vector3(x=float(action[0]), y=float(action[1]), z=float(action[2])),
                angular=Vector3(x=float(action[3]), y=float(action[4]), z=float(action[5])),
            )

            msg = MotionUpdate()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.velocity = twist
            msg.target_stiffness = np.diag([100.0]*3 + [50.0]*3).flatten()
            msg.target_damping   = np.diag([40.0]*3  + [15.0]*3).flatten()
            msg.feedforward_wrench_at_tip = Wrench()
            msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
            msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY

            move_robot(motion_update=msg)
            send_feedback("running")

            time.sleep(max(0, 0.25 - (time.time() - loop_start)))  # ~4Hz

        return True
