# AIC 컨트롤러

> 날짜: 2026-04-05
> 원본: `ws_aic/src/aic/docs/aic_controller.md`

---

## 개요

정책(10–30 Hz)의 명령을 받아 로봇 하드웨어(500 Hz)로 전달하는 저수준 제어 시스템. 안전 검사와 스무딩을 거친 후 임피던스 제어를 적용한다.

---

## 제어 파이프라인

```
입력 명령(10~30Hz) → 클램핑 → 보간 → 임피던스 제어 → 중력 보상 → 관절 토크 출력(500Hz)
```

| 단계 | 내용 |
|------|------|
| 명령 클램핑 | URDF 관절 한계 / 사용자 지정 카테시안 한계 내로 제한 |
| 명령 보간 | 저속 정책 명령 → 고속 고품질 setpoint 변환 |
| 임피던스 제어 | 카테시안 또는 관절 임피던스 계산 |
| 중력 보상 | 로봇 링크 중력 토크 계산 |
| 명령 실행 | 임피던스 + 중력 보상 토크를 관절에 전달 |

> **추적 오차 초과 시 타겟 리셋**: 충돌 상태에서 명령을 계속 보내 오차가 누적되는 문제를 방지.

---

## 제어 수식

### 카테시안 임피던스
$$\tau = \mathbf{J}^T \Big[ \mathbf{K}_p (\mathbf{x}_{des} - \mathbf{x}) + \mathbf{K}_d (\dot{\mathbf{x}}_{des} - \dot{\mathbf{x}}) + \mathbf{W}_f \Big] + \tau_{null}$$

### 관절 임피던스
$$\tau = \mathbf{K}_p (\mathbf{q}_{des} - \mathbf{q}) + \mathbf{K}_d (\dot{\mathbf{q}}_{des} - \dot{\mathbf{q}}) + \tau_f$$

---

## ROS 2 인터페이스

### 모드 전환
```bash
# 카테시안 모드로 전환 (기본값)
ros2 service call /aic_controller/change_target_mode \
  aic_control_interfaces/srv/ChangeTargetMode "{target_mode: {mode: 1}}"

# 관절 모드로 전환
ros2 service call /aic_controller/change_target_mode \
  aic_control_interfaces/srv/ChangeTargetMode "{target_mode: {mode: 2}}"
```

### F/T 센서 타링
```bash
ros2 service call /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger
```
> 평가 중에는 이 서비스 호출 불가. 채점 전 시스템이 자동 타링.

---

## MotionUpdate 주요 파라미터

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `header.frame_id` | string | `gripper/tcp` (상대) 또는 `base_link` (절대) |
| `pose` | `geometry_msgs/Pose` | 타겟 카테시안 포즈 (MODE_POSITION) |
| `velocity` | `geometry_msgs/Twist` | 타겟 속도 (MODE_VELOCITY) |
| `target_stiffness` | float64[36] | 6×6 강성 행렬 (클수록 강성 제어) |
| `target_damping` | float64[36] | 6×6 감쇠 행렬 (진동 억제) |
| `feedforward_wrench_at_tip` | `geometry_msgs/Wrench` | TCP에 가할 추가 힘/토크 |
| `trajectory_generation_mode` | enum | `MODE_POSITION`(2) 또는 `MODE_VELOCITY`(1) |

## JointMotionUpdate 주요 파라미터

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `target_state.positions` | float64[] | 타겟 관절 위치 (MODE_POSITION) |
| `target_state.velocities` | float64[] | 타겟 관절 속도 (MODE_VELOCITY) |
| `target_stiffness` | float64[] | 관절별 강성 |
| `target_damping` | float64[] | 관절별 감쇠 |

---

> 컨트롤러 파라미터 설정은 `aic_controller_parameters.yaml` 참조. 평가 중 설정은 고정되며 모든 참가자 동일 적용.
