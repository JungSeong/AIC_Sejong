# AIC 인터페이스 정의

> 날짜: 2026-04-05
> 원본: `ws_aic/src/aic/docs/aic_interfaces.md`

---

## 개요

챌린지에서 사용하는 표준 ROS 2 인터페이스와 커스텀 인터페이스를 정의한다. 커스텀 인터페이스는 `aic_interfaces` 패키지에 정의되어 있다.

---

## 입력 — 센서 토픽

| 토픽 | 메시지 타입 | 설명 |
|------|------------|------|
| `/left_camera/image` | `sensor_msgs/Image` | 좌측 손목 카메라 (보정된 이미지) |
| `/left_camera/camera_info` | `sensor_msgs/CameraInfo` | 좌측 카메라 캘리브레이션 |
| `/center_camera/image` | `sensor_msgs/Image` | 중앙 손목 카메라 |
| `/center_camera/camera_info` | `sensor_msgs/CameraInfo` | 중앙 카메라 캘리브레이션 |
| `/right_camera/image` | `sensor_msgs/Image` | 우측 손목 카메라 |
| `/right_camera/camera_info` | `sensor_msgs/CameraInfo` | 우측 카메라 캘리브레이션 |
| `/fts_broadcaster/wrench` | `geometry_msgs/WrenchStamped` | 힘/토크 센서 데이터 |
| `/joint_states` | `sensor_msgs/JointState` | 로봇 관절 상태 |
| `/gripper_state` | `sensor_msgs/JointState` | 그리퍼 상태 |
| `/tf` | `tf2_msgs/TFMessage` | 동적 좌표 변환 |
| `/tf_static` | `tf2_msgs/TFMessage` | 정적 좌표 변환 |

## 입력 — 액션 서버

| 액션 | 타입 | 설명 |
|------|------|------|
| `/insert_cable` | `aic_task_interfaces/action/InsertCable` | 케이블 삽입 태스크 트리거 |

## 입력 — 컨트롤러 상태

| 토픽 | 메시지 타입 | 설명 |
|------|------------|------|
| `/aic_controller/controller_state` | `aic_control_interfaces/ControllerState` | TCP 포즈·속도, 추적 오차, 관절 토크 등 |

---

## 출력 — 로봇 명령 토픽

| 토픽 | 메시지 타입 | 설명 |
|------|------------|------|
| `/aic_controller/pose_commands` | `aic_control_interfaces/MotionUpdate` | 카테시안 공간 타겟 포즈/속도 |
| `/aic_controller/joint_commands` | `aic_control_interfaces/JointMotionUpdate` | 관절 공간 타겟 위치/속도 |

> 컨트롤러는 **한 번에 하나의 모드**만 처리. 모드 전환은 `/aic_controller/change_target_mode` 서비스 사용.

---

## 컨트롤러 서비스

| 서비스 | 타입 | 설명 |
|--------|------|------|
| `/aic_controller/change_target_mode` | `aic_control_interfaces/srv/ChangeTargetMode` | 카테시안(1) / 관절(2) 모드 전환 |
| `/aic_controller/tare_force_torque_sensor` | `std_srvs/srv/Trigger` | F/T 센서 타링 (평가 중 사용 불가) |

---

## 커스텀 인터페이스 목록

| 파일 | 설명 |
|------|------|
| `aic_task_interfaces/action/InsertCable.action` | 삽입 태스크 트리거 액션 |
| `aic_task_interfaces/msg/Task.msg` | 케이블 삽입 태스크 파라미터 |
| `aic_control_interfaces/msg/MotionUpdate.msg` | 카테시안 제어 타겟 |
| `aic_control_interfaces/msg/JointMotionUpdate.msg` | 관절 제어 타겟 |
| `aic_model_interfaces/msg/Observation.msg` | 센서 통합 관측값 스냅샷 |

---

*관련 문서: `aic_controller.md` (컨트롤러 상세) / `policy.md` (정책 API)*
