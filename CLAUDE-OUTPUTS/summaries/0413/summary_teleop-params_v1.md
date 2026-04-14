# 원격 조종(Teleoperation) 파라미터 정리

> AIC 로봇 텔레오퍼레이션 방법 2가지 — `aic_teleoperation`(단순 조종)과 `lerobot-teleoperate`(데이터 녹화 연동) — 의 파라미터 및 키 매핑 정리.
> 소스: `aic_utils/aic_teleoperation/README.md`, `aic_utils/lerobot_robot_aic/README.md`

---

## 1. aic_teleoperation (단순 키보드 조종)

데이터 녹화 없이 로봇만 직접 움직일 때 사용.

### 실행

```bash
cd ~/ws_aic/src/aic

# 관절 공간 제어
pixi run ros2 run aic_teleoperation joint_keyboard_teleop

# Cartesian 공간 제어
pixi run ros2 run aic_teleoperation cartesian_keyboard_teleop
```

### 관절 공간 키 매핑

| 키 | 관절 | 방향 |
|----|------|------|
| `q` / `a` | shoulder_pan | − / + |
| `w` / `s` | shoulder_lift | − / + |
| `e` / `d` | elbow | − / + |
| `r` / `f` | wrist_1 | − / + |
| `t` / `g` | wrist_2 | − / + |
| `y` / `h` | wrist_3 | − / + |
| `k` | 저속 모드 | 0.075 rad/s |
| `l` | 고속 모드 | 0.200 rad/s |
| `ESC` | 종료 | |

### Cartesian 공간 키 매핑

| 키 | 동작 |
|----|------|
| `a` / `d` | X축 − / + |
| `w` / `s` | Y축 − / + |
| `r` / `f` | Z축 − / + |
| `Shift+s` / `Shift+w` | Angular X − / + |
| `Shift+a` / `Shift+d` | Angular Y − / + |
| `q` / `e` | Angular Z − / + |
| `n` | 기준 프레임 → `gripper/tcp` (툴 기준) |
| `m` | 기준 프레임 → `base_link` (월드 기준) |
| `k` | 저속 모드 (linear: 0.02 m/s, angular: 0.02 rad/s) |
| `l` | 고속 모드 (linear: 0.10 m/s, angular: 0.10 rad/s) |
| `ESC` | 종료 | |

---

## 2. lerobot-teleoperate (LeRobot 연동 조종 + 녹화)

데이터셋 녹화(`lerobot-record`)와 동일한 드라이버를 사용하는 텔레오퍼레이션.

### 실행

```bash
cd ~/ws_aic/src/aic
pixi run lerobot-teleoperate \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=<teleop-type> --teleop.id=aic \
  --robot.teleop_target_mode=<mode> \
  --robot.teleop_frame_id=<frame_id> \
  --display_data=true
```

### 핵심 파라미터

| 파라미터 | 선택지 | 설명 |
|----------|--------|------|
| `--teleop.type` | `aic_keyboard_ee` | Cartesian 키보드 조종 |
| | `aic_keyboard_joint` | 관절 공간 키보드 조종 |
| | `aic_spacemouse` | Cartesian SpaceMouse 조종 |
| `--robot.teleop_target_mode` | `cartesian` | `aic_keyboard_ee`, `aic_spacemouse` 사용 시 |
| | `joint` | `aic_keyboard_joint` 사용 시 |
| `--robot.teleop_frame_id` | `base_link` | 월드(로봇 베이스) 기준 Cartesian 제어 |
| | `gripper/tcp` | 툴 끝단 기준 Cartesian 제어 (기본값) |

> **주의:** `--teleop.type`과 `--robot.teleop_target_mode`는 **반드시 쌍으로** 맞춰야 한다.
> 드라이버(`AICRobotAICController`)가 `teleop.type`에 접근할 수 없어 `teleop_target_mode`를 따로 지정해야 한다.

### Cartesian 키 매핑 (lerobot, aic_keyboard_ee 기준)

| 키 | 동작 |
|----|------|
| `w` / `s` | linear Y − / + |
| `a` / `d` | linear X − / + |
| `r` / `f` | linear Z − / + |
| `q` / `e` | angular Z − / + |
| `Shift+w` / `Shift+s` | angular X + / − |
| `Shift+a` / `Shift+d` | angular Y − / + |
| `t` | 저속/고속 토글 |

> Shift 계열 키: **Shift를 먼저 누른 채로** 방향키를 누르고, **방향키를 먼저 떼고** Shift를 뗄 것. 순서 반대면 계속 회전함.

### 관절 공간 키 매핑 (lerobot, aic_keyboard_joint 기준)

| 키 | 관절 |
|----|------|
| `q` / `a` | shoulder_pan − / + |
| `w` / `s` | shoulder_lift − / + |
| `e` / `d` | elbow − / + |
| `r` / `f` | wrist_1 − / + |
| `t` / `g` | wrist_2 − / + |
| `y` / `h` | wrist_3 − / + |
| `u` | 저속/고속 토글 |

### SpaceMouse 설정 (선택)

```bash
# USB 권한 설정 (최초 1회)
sudo tee /etc/udev/rules.d/99-spacemouse.rules << 'EOF'
KERNEL=="hidraw*", ATTRS{idVendor}=="046d", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="046d", MODE="0666", GROUP="plugdev"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger
```

---

## 3. 두 방식 비교

| 항목 | aic_teleoperation | lerobot-teleoperate |
|------|-------------------|---------------------|
| 데이터 녹화 | 불가 | `lerobot-record`로 전환 시 가능 |
| 실행 방식 | `ros2 run` | `pixi run lerobot-teleoperate` |
| 프레임 전환 | `n` / `m` 키 | `--robot.teleop_frame_id` 파라미터 |
| 속도 조절 | `k` / `l` 키 | `t` / `u` 토글 키 |
| 권장 용도 | 빠른 테스트·탐색 | **훈련 데이터 수집 전 연습** |
