"""
StagedPolicy v3: Vision 통합 3단계 State Machine 정책

Stage 1 (이동) — 모션 플래닝 + Vision (신찬희 담당)
  - 포트 좌표 획득: YOLO+스테레오만 사용
  - 목표: 그리퍼를 포트 축선 위 10cm 지점까지 이동
  - 방식: S-curve 직선 보간

Stage 2/3:
  - Stage 1의 Vision 포트 좌표와 vision offset 모델 기반 정렬/삽입

포트/플러그 ground-truth 좌표는 사용하지 않는다.

실행:
  pixi reinstall ros-kilted-motion-planning-node
  pixi run ros2 run aic_model aic_model \\
    --ros-args -p use_sim_time:=true \\
    -p policy:=motion_planning_node.StagedPolicy
"""

from typing import Optional

import numpy as np
import cv2

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform

from motion_planning_node.core.config import Stage1Config
from motion_planning_node.core.stage1 import Stage1Approach
from motion_planning_node.core.stage23 import Stage23Controller
from motion_planning_node.core.vision import VisionPortEstimator


# ═══════════════════════════════════════════════════════════
#  StagedPolicy
# ═══════════════════════════════════════════════════════════

class StagedPolicy(Policy):
    """3단계 State Machine 정책"""

    PORT_AXIS_LOCAL: np.ndarray = np.array([0.0, 0.0, 1.0])

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._task: Optional[Task] = None

        # Vision 모듈
        self._vision = VisionPortEstimator(
            model_path=Stage1Config.DETECTION_MODEL_PATH,
            conf_thresh=Stage1Config.DETECTION_CONF_THRESH,
            logger=self.get_logger(),
        )
        self.get_logger().info("DETECTION 모델 백그라운드 로드 시작")
        self.get_logger().info(f"  DETECTION model path: {Stage1Config.DETECTION_MODEL_PATH}")
        self.get_logger().info(f"  DETECTION conf threshold: {Stage1Config.DETECTION_CONF_THRESH}")

        self._stage1 = Stage1Approach(self, self._vision)
        self._stage23 = Stage23Controller(self)
        self._distance_model = None
        self._distance_device = None
        self._distance_target_mean = np.zeros(3, dtype=np.float32)
        self._distance_target_std = np.ones(3, dtype=np.float32)
        self._load_distance_model()

    def _transform_to_pose(self, tf: Transform) -> Pose:
        return Pose(
            position=Point(
                x=tf.translation.x, y=tf.translation.y, z=tf.translation.z,
            ),
            orientation=Quaternion(
                x=tf.rotation.x, y=tf.rotation.y,
                z=tf.rotation.z, w=tf.rotation.w,
            ),
        )

    def _load_distance_model(self) -> None:
        model_path = Stage1Config.DISTANCE_MODEL_PATH
        try:
            import torch
            import torch.nn as nn
            from torchvision.models import resnet50

            checkpoint = torch.load(model_path, map_location="cpu")
            config = checkpoint.get("config", {})
            self._distance_target_mean = np.asarray(
                config.get("target_mean", [0.0, 0.0, 0.0]), dtype=np.float32
            )
            self._distance_target_std = np.asarray(
                config.get("target_std", [1.0, 1.0, 1.0]), dtype=np.float32
            )
            encoder = resnet50(weights=None)
            in_features = encoder.fc.in_features
            encoder.fc = nn.Identity()

            class PositionModel(nn.Module):
                def __init__(self, encoder, in_dim):
                    super().__init__()
                    self.encoder = encoder
                    self.head = nn.Sequential(
                        nn.Linear(in_dim, 256),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.1),
                        nn.Linear(256, 128),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.1),
                        nn.Linear(128, 3),
                    )

                def forward(self, x):
                    return self.head(self.encoder(x))

            self._distance_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._distance_model = PositionModel(encoder, in_features).to(self._distance_device)
            self._distance_model.load_state_dict(checkpoint["model_state_dict"])
            self._distance_model.eval()
            self.get_logger().info(f"Distance model loaded: {model_path}")
        except Exception as ex:
            self._distance_model = None
            self.get_logger().warn(f"Distance model unavailable: {ex}")

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

    def predict_distance_offset(self, obs) -> Optional[np.ndarray]:
        """Return predicted plug-tip-to-port offset [x,y,z] in meters."""
        if self._distance_model is None or obs is None:
            return None
        bgr = self._image_msg_to_bgr(obs.center_image)
        if bgr is None:
            return None
        try:
            import torch

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            image = resized.astype(np.float32) / 255.0
            image = (
                image - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
            ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
            tensor = tensor.to(self._distance_device)
            with torch.no_grad():
                pred_norm = self._distance_model(tensor).detach().cpu().numpy()[0]
            pred_mm = pred_norm * self._distance_target_std + self._distance_target_mean
            return pred_mm.astype(np.float32) / 1000.0
        except Exception as ex:
            self.get_logger().warn(f"Distance prediction failed: {ex}")
            return None

    # ─────────────────────────────────────────────────────
    #  메인
    # ─────────────────────────────────────────────────────

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(
            f"StagedPolicy 시작\n"
            f"  cable={task.cable_name}, plug={task.plug_name}\n"
            f"  port={task.port_name}, target={task.target_module_name}"
        )
        self._task = task

        # Stage 1
        result = self._stage1.run(get_observation, move_robot, send_feedback)
        self.get_logger().info(
            f"Stage 1 result: success={result.success}, "
            f"source={result.port_source}, "
            f"elapsed={result.elapsed_time:.2f}s, "
            f"reason={result.failure_reason}"
        )

        # Vision 실패 시 Stage 2/3는 항상 None pose만 반환하므로
        # 시간만 낭비. 즉시 종료하여 평가 시스템이 다음 태스크로 넘어갈 수 있도록 함.
        if result.port_source == "none":
            self.get_logger().error(
                "포트 좌표 획득 실패 (Vision) → 조기 종료"
            )
            send_feedback("failed: port not detected (skipping stage 2/3)")
            return False

        # Stage 2/3 — Stage1 result의 Vision port_pose만 전달
        port_pose_for_23 = None
        if result.port_source == "vision":
            port_pose_for_23 = result.port_pose

        try:
            self._stage23.align(move_robot, send_feedback,
                                port_pose_vision=port_pose_for_23,
                                get_observation=get_observation)
            self._stage23.insert(get_observation, move_robot, send_feedback,
                                 port_pose_vision=port_pose_for_23)
        except Exception as ex:
            self.get_logger().warn(f"Stage 2/3 실행 중 예외: {ex}")

        self.get_logger().info("StagedPolicy 완료")
        return True
