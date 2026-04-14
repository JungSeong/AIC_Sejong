# aic_bringup README 요약

**날짜:** 2026-04-07
**주제:** aic_bringup 패키지 — 시뮬레이션 런치 및 설정
**파일:** `ws_aic/src/aic/aic_bringup/README.md`

---

## 핵심 한 줄 요약

`aic_bringup`은 Gazebo 시뮬레이션 환경 전체를 띄우는 ROS 2 런치 패키지로, 로봇·태스크보드·케이블 스폰 및 평가 엔진 실행을 담당한다.

---

## 패키지 역할

- Gazebo 시뮬레이션 + UR5e 로봇 실행
- 태스크 보드 스폰 (LC/SFP/SC 마운트, SC 포트, NIC 카드 마운트)
- 케이블 스폰 및 그리퍼 연결
- AIC Controller (임피던스 제어) 시작
- `aic_engine` 연동으로 시험(trial) 자동 오케스트레이션
- ROS-Gazebo 브릿지 구성

---

## 런치 파일 3종

### 1. `aic_gz_bringup.launch.py` — 메인 런치

```bash
# 기본 실행
ros2 launch aic_bringup aic_gz_bringup.launch.py

# 자격심사(Qualification) 환경
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=false \
  start_aic_engine:=true

# 개발 모드 (Ground Truth 활성화)
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true \
  spawn_task_board:=true
```

**주요 파라미터 그룹:**

| 그룹 | 대표 파라미터 | 기본값 |
|------|-------------|--------|
| 로봇 위치 | `robot_x/y/z`, `robot_roll/pitch/yaw` | x=-0.2, y=0.2, z=1.14, yaw=-3.141 |
| 태스크 보드 | `spawn_task_board`, `task_board_x/y/z` | false, x=0.15, y=-0.2, z=1.14 |
| 케이블 | `spawn_cable`, `cable_type`, `attach_cable_to_gripper` | false, sfp_sc_cable, false |
| Gazebo | `world_file`, `gazebo_gui` | aic.sdf, true |
| 시각화 | `launch_rviz`, `rviz_config_file` | false, view_robot.rviz |
| Ground Truth | `ground_truth` | false |
| AIC 엔진 | `start_aic_engine`, `shutdown_on_aic_engine_exit` | false, false |

**케이블 타입:**
- `sfp_sc_cable` → cable_z 기본값 `1.518`
- `sfp_sc_cable_reversed` → cable_z를 `1.508`으로 변경 필요

---

### 2. `spawn_task_board.launch.py` — 태스크 보드 단독 스폰

기존 Gazebo 시뮬레이션에 태스크 보드를 추가로 스폰할 때 사용.

**마운트 레일 구성 (총 6개):**

| 레일 | 커넥터 종류 | 파라미터 예시 |
|------|------------|---------------|
| rail_0 (좌측) | LC, SFP, SC | `lc_mount_rail_0_present:=true` |
| rail_1 (우측) | LC, SFP, SC | `sfp_mount_rail_1_present:=true` |

각 마운트마다: `present`, `translation` (범위: **±0.09625m**), `roll/pitch/yaw` 설정 가능.

**포트 레일:**
- SC Port: 0, 1번 레일
- NIC Card Mount: 0~4번 레일 (5개)

> **평가 시 고정 조건:** 태스크 보드 roll/pitch = 0.0, SC 포트 yaw = 0.0
> 학습용 도메인 랜덤화를 위해 임의 방향 설정 가능

---

### 3. `spawn_cable.launch.py` — 케이블 단독 스폰

기존 시뮬레이션에 케이블만 추가로 스폰.

```bash
ros2 launch aic_bringup spawn_cable.launch.py
```

기본 위치: x=-0.35, y=0.4, z=1.15 / `attach_cable_to_gripper:=false`

---

## 임피던스 컨트롤러 테스트

```bash
# 1. 시뮬레이션 시작
ros2 launch aic_bringup aic_gz_bringup.launch.py

# 2. Joint + Cartesian 타겟 전송 테스트
ros2 run aic_bringup test_impedance.py
```

`test_impedance.py`는 `JointMotionUpdate` (관절), `MotionUpdate` (카르테시안), `ChangeTargetMode` 서비스를 모두 테스트한다.

---

## 핵심 포인트 (3줄 요약)

1. **메인 런치 파일 하나**(`aic_gz_bringup.launch.py`)로 로봇·태스크보드·케이블·평가엔진을 모두 제어하며, `ground_truth:=true`로 개발 디버깅, `start_aic_engine:=true`로 평가 모드 전환.
2. **태스크 보드 마운트**는 LC/SFP/SC × rail_0/1 조합으로 최대 6개 설정 가능하고, 이동 범위는 ±0.09625m로 제한됨 (충돌 방지).
3. **도메인 랜덤화** 목적으로 위치·방향을 자유롭게 바꿀 수 있지만, 실제 평가 시에는 roll/pitch=0, SC 포트 yaw=0으로 고정됨.
