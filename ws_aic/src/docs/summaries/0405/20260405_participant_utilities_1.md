# 참가자 유틸리티

> 날짜: 2026-04-05
> 원본: `ws_aic/src/aic/docs/participant_utilities.md`

---

## 텔레오퍼레이션

| 도구 | 설명 |
|------|------|
| **aic_teleoperation** | 키보드 기반 관절/카테시안 공간 제어 |
| **lerobot_robot_aic** | LeRobot 기반 텔레오퍼레이션 (키보드 또는 SpaceMouse). `lerobot-record`로 데이터셋 녹화 가능 |
| **test_impedance.py** | 특정 포즈/관절 설정으로 로봇 이동 예시 스크립트 |
| **home_robot.py** | 로봇 홈 포즈 복귀 스크립트 |

---

## 데이터 수집 & 훈련

- **lerobot_robot_aic** (`lerobot-record`) — LeRobot 포맷으로 텔레오퍼레이션 데이터셋 녹화
- **LeRobot 통합** — HuggingFace LeRobot과 AIC 연동으로 정책 훈련 파이프라인 구성 가능

---

## 시각화 도구

| 도구 | 용도 |
|------|------|
| **RViz** | ROS 2 공식 시각화 도구. 기본 설정: 중앙 카메라 스트림 표시. 추가 카메라 뷰 직접 설정 가능 |
| **PlotJuggler** | ROS 토픽 시계열 데이터 실시간 시각화 |

---

## ROS 2 CLI 주요 명령

```bash
# 노드 목록
ros2 node list

# 토픽 데이터 실시간 확인
ros2 topic echo /aic_controller/state

# 발행 주기 확인
ros2 topic hz /left_camera/image

# 파라미터 목록
ros2 param list /aic_controller

# 데이터 녹화
ros2 bag record -o my_recording /aic_controller/state /center_camera/image

# 서비스 호출
ros2 service call /aic_controller/change_target_mode \
  aic_control_interfaces/srv/ChangeTargetMode "{target_mode: {mode: 1}}"

# 인터페이스 타입 확인
ros2 interface show aic_control_interfaces/msg/MotionUpdate
```

---

*텔레오퍼레이션 상세: `aic_utils/aic_teleoperation/README.md` / 씬 탐색: `scene_description.md`*
