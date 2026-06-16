"""ROS observation을 pose prediction 모델 입력으로 변환하고 추론하는 래퍼."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sensor_msgs.msg import Image

from final_policy.config import FinalPolicyConfig
from final_policy.pose_prediction.model import build_pose_model


JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


class PosePredictor:
    """checkpoint를 로드하고 observation마다 포트 offset/yaw 예측값을 반환한다."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        device: str = FinalPolicyConfig.DEVICE,
        logger=None,
    ) -> None:
        """checkpoint 경로와 device를 설정한 뒤 모델을 즉시 로드한다."""
        self.logger = logger
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.cameras = FinalPolicyConfig.CAMERAS
        self.image_size: int | tuple[int, int] | None = 224
        self.model = None
        self.target_mean = {}
        self.target_std = {}
        self._load()

    def _log(self, level: str, message: str) -> None:
        """logger가 있으면 지정한 레벨로 메시지를 남긴다."""
        if self.logger is None:
            return
        getattr(self.logger, level)(message)

    def _load(self) -> None:
        """checkpoint payload에서 모델 설정, 가중치, target 정규화 통계를 읽는다."""
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Pose prediction checkpoint not found: {self.checkpoint_path}"
            )
        payload = torch.load(self.checkpoint_path, map_location="cpu")
        config = payload.get("config", {})
        state_dict = payload["model_state_dict"]
        self.cameras = tuple(config.get("cameras", self.cameras))
        self.image_size = config.get("image_size", 224)
        self.target_mean = dict(config.get("target_mean", {}))
        self.target_std = dict(config.get("target_std", {}))
        self.model = build_pose_model(
            backbone_name=config.get("backbone", "resnet50"),
            pretrained=False,
            hidden_dim=int(config.get("hidden_dim", 256)),
            force_torque_dim=int(config.get("force_torque_dim", 6)),
            joint_dim=int(config.get("joint_dim", 12)),
            aggregation=config.get("aggregation", "concat"),
            num_views=int(config.get("num_views", len(self.cameras))),
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self._log("info", f"Pose model loaded: {self.checkpoint_path}")

    def _camera_image(self, observation, camera: str) -> Optional[Image]:
        """observation에서 카메라 이름에 맞는 Image 메시지를 꺼낸다."""
        return getattr(observation, f"{camera}_image", None)

    def _image_msg_to_tensor(self, image_msg: Image) -> torch.Tensor:
        """ROS Image 메시지를 모델 입력용 RGB 정규화 텐서로 변환한다."""
        height = int(image_msg.height)
        width = int(image_msg.width)
        encoding = getattr(image_msg, "encoding", "").lower()
        channels = 4 if encoding in {"rgba8", "bgra8"} else 3
        flat = np.frombuffer(image_msg.data, dtype=np.uint8)
        step = int(getattr(image_msg, "step", 0))
        if step > 0 and flat.size >= height * step:
            rows = flat[: height * step].reshape(height, step)
            image = rows[:, : width * channels].reshape(height, width, channels)
        else:
            image = flat[: height * width * channels].reshape(height, width, channels)
        if channels == 4:
            image = image[:, :, :3]
        if encoding in {"bgr8", "bgra8"}:
            image = image[:, :, ::-1]
        tensor = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
        )
        if self.image_size is not None:
            size = (
                (self.image_size, self.image_size)
                if isinstance(self.image_size, int)
                else self.image_size
            )
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std

    def _force_torque(self, observation) -> torch.Tensor:
        """wrist wrench의 force/torque 6차원 값을 텐서로 만든다."""
        wrench = observation.wrist_wrench.wrench
        return torch.tensor(
            [
                wrench.force.x,
                wrench.force.y,
                wrench.force.z,
                wrench.torque.x,
                wrench.torque.y,
                wrench.torque.z,
            ],
            dtype=torch.float32,
        )

    def _joint_positions(self, observation) -> torch.Tensor:
        """UR joint angle을 sin/cos 형태의 12차원 텐서로 변환한다."""
        joint_state = observation.joint_states
        by_name = dict(zip(list(joint_state.name), list(joint_state.position)))
        values = torch.tensor(
            [float(by_name.get(name, 0.0)) for name in JOINT_NAMES],
            dtype=torch.float32,
        )
        return torch.cat([torch.sin(values), torch.cos(values)], dim=0)

    def _denorm(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """checkpoint에 저장된 target mean/std가 있으면 예측값을 원 단위로 복원한다."""
        if name not in self.target_mean or name not in self.target_std:
            return value
        mean = torch.as_tensor(self.target_mean[name], dtype=torch.float32)
        std = torch.as_tensor(self.target_std[name], dtype=torch.float32).clamp_min(1e-6)
        return value.cpu() * std + mean

    @torch.inference_mode()
    def predict(self, observation) -> Optional[dict[str, np.ndarray | float]]:
        """현재 observation으로 port0/port1 offset(m)과 dyaw(rad)를 예측한다."""
        if observation is None:
            return None
        images = []
        for camera in self.cameras:
            image_msg = self._camera_image(observation, camera)
            if image_msg is None:
                self._log("warn", f"Observation missing {camera}_image")
                return None
            images.append(self._image_msg_to_tensor(image_msg))
        image = torch.stack(images, dim=0).unsqueeze(0).to(self.device)
        force_torque = self._force_torque(observation).unsqueeze(0).to(self.device)
        joints = self._joint_positions(observation).unsqueeze(0).to(self.device)
        output = self.model(image, force_torque, joints)
        port0_mm = self._denorm("port0_position", output["port0_position"][0]).numpy()
        port1_mm = self._denorm("port1_position", output["port1_position"][0]).numpy()
        dyaw = float(self._denorm("dyaw", output["dyaw"][0]).item())
        if (
            not np.isfinite(port0_mm).all()
            or not np.isfinite(port1_mm).all()
            or not np.isfinite(dyaw)
        ):
            self._log("warn", "Non-finite pose prediction")
            return None
        port0_m = np.array([port0_mm[0], port0_mm[1], 0.0], dtype=np.float64) / 1000.0
        port1_m = np.array([port1_mm[0], port1_mm[1], 0.0], dtype=np.float64) / 1000.0
        return {
            "port0_position_m": port0_m,
            "port1_position_m": port1_m,
            "dyaw_rad": dyaw,
        }
