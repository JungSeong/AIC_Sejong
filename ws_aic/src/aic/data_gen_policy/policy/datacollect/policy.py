"""
DataCollect policy
──────────────────
AutoCapture + 에피소드 시작 전 F/T 센서 자동 Tare.

실행:
  pixi run ros2 run aic_model aic_model --ros-args \
    -p policy:=data_gen_policy.policy.datacollect
"""

import subprocess

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

from data_gen_policy.policy.autocapture import AutoCapture

TARE_SERVICE = "/aic_controller/tare_force_torque_sensor"
TARE_TYPE    = "std_srvs/srv/Trigger"


class DataCollect(AutoCapture):
    """
    AutoCapture를 상속하여 매 에피소드 시작 전 F/T 센서 Tare를 추가.

    환경변수 (AutoCapture에서 상속):
      AIC_CAPTURE_DIR               저장 경로
      AIC_CAPTURE_STEP_SLEEP_SEC    스텝 간격 초 (기본 0.05 = 20Hz)
      AIC_CAPTURE_CHEATCODE_*       CheatCode 동작 파라미터
    """

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        # 1. 시스템 레벨 Tare 시도
        self._tare_sensor()
        
        # 2. 코드 레벨 소프트웨어 Tare (Eval 환경 대응)
        # 초기 오프셋을 캡처하기 위해 잠시 대기 후 관측 데이터 획득
        self.sleep_for(1.0)
        initial_obs = get_observation()
        fz_offset = 0.0
        if initial_obs and hasattr(initial_obs, 'wrist_wrench'):
            fz_offset = initial_obs.wrist_wrench.wrench.force.z
            self.get_logger().info(f"[DataCollect] Initial Fz offset captured: {fz_offset:.4f}")

        # 3. get_observation 콜백 래핑
        def wrapped_get_observation() -> Observation:
            obs = get_observation()
            if obs and hasattr(obs, 'wrist_wrench'):
                # Fz 값에서 초기 오프셋 차감
                obs.wrist_wrench.wrench.force.z -= fz_offset
            return obs

        # 래핑된 콜백을 사용하여 데이터 수집 진행
        return super().insert_cable(task, wrapped_get_observation, move_robot, send_feedback)

    def _tare_sensor(self) -> None:
        """F/T 센서 Tare 서비스 호출."""
        self.get_logger().info(f"[DataCollect] Tare: {TARE_SERVICE}")
        try:
            result = subprocess.run(
                ["ros2", "service", "call", TARE_SERVICE, TARE_TYPE],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                self.get_logger().info("[DataCollect] Tare 완료")
            else:
                self.get_logger().warn(
                    f"[DataCollect] Tare 실패 (code={result.returncode}): "
                    f"{result.stderr.strip()}"
                )
        except subprocess.TimeoutExpired:
            self.get_logger().error("[DataCollect] Tare 타임아웃 (10s)")
        except Exception as e:
            self.get_logger().error(f"[DataCollect] Tare 예외: {e}")
