"""Approach first, then align and insert with the image distance model."""

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

from distance_prediction_policy.config import DistancePredictionConfig
from distance_prediction_policy.model_feedback import (
    ModelFeedbackController,
    VisionOffsetPredictor,
)
from distance_prediction_policy.yolo_approach import YoloTriangulationApproach
from motion_planning_node.core.config import Stage1Config
from motion_planning_node.core.vision import VisionPortEstimator


class DistancePredictionPolicy(Policy):
    """Policy that uses YOLO triangulation approach followed by model feedback."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._task = None

        self._vision = VisionPortEstimator(
            model_path=Stage1Config.DETECTION_MODEL_PATH,
            conf_thresh=Stage1Config.DETECTION_CONF_THRESH,
            logger=self.get_logger(),
        )
        self.get_logger().info("Preloading DETECTION model for approach fallback...")
        self.get_logger().info(f"  DETECTION model path: {Stage1Config.DETECTION_MODEL_PATH}")
        self.get_logger().info(f"  DETECTION conf threshold: {Stage1Config.DETECTION_CONF_THRESH}")
        self._vision._ensure_loaded()
        if not self._vision._loaded:
            self.get_logger().warn(
                "YOLO preload failed. Approach requires YOLO triangulation, "
                "so this policy may fail until the model path is fixed."
            )

        self.get_logger().info(
            "Loading distance prediction checkpoint: "
            f"{DistancePredictionConfig.CHECKPOINT_PATH}"
        )
        self._distance_predictor = VisionOffsetPredictor(logger=self.get_logger())

        self._approach = YoloTriangulationApproach(self, self._vision)
        self._feedback = ModelFeedbackController(
            self,
            self._distance_predictor,
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(
            "DistancePredictionPolicy start: "
            f"cable={task.cable_name}, plug={task.plug_name}, "
            f"port={task.port_name}, target={task.target_module_name}"
        )
        self._task = task

        self.get_logger().info("Pre-stage settle: 0.8s")
        self.sleep_for(0.8)

        result = self._approach.run(get_observation, move_robot, send_feedback)
        self.get_logger().info(
            "Approach result: "
            f"success={result.success}, "
            f"elapsed={result.elapsed_time:.2f}s, reason={result.failure_reason}"
        )

        if not result.success:
            self.get_logger().error("Approach failed")
            send_feedback("failed: YOLO triangulation approach failed")
            return False

        success = self._feedback.run(
            get_observation,
            move_robot,
            send_feedback,
        )
        self.get_logger().info(f"DistancePredictionPolicy done: success={success}")
        return success
