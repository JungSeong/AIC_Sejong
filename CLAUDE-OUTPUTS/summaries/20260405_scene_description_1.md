# 씬 설명

> 날짜: 2026-04-05
> 원본: `ws_aic/src/aic/docs/scene_description.md`

---

## 로봇 구성

| 구성요소 | 모델 |
|----------|------|
| 로봇 팔 | Universal Robots UR5e |
| 그리퍼 | Robotiq Hand-E |
| 힘/토크 센서 | ATI AXIA80-M20 |
| 카메라 (×3) | Basler acA2440-20gc + Edmunds lens 58-000 (1152×1024, 20 FPS) |

카메라는 손목에 좌/중/우 3대 장착. TCP 프레임: `gripper/tcp`.

---

## 환경 커스터마이징

```bash
# 예시: 다양한 컴포넌트가 있는 태스크보드 스폰
/entrypoint.sh \
  spawn_task_board:=true \
  task_board_x:=0.3 task_board_y:=-0.1 task_board_z:=1.2 \
  task_board_yaw:=0.785 \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  spawn_cable:=true cable_type:=sfp_sc_cable \
  attach_cable_to_gripper:=true \
  ground_truth:=true start_aic_engine:=false
```

**주요 파라미터:**

| 파라미터 | 설명 |
|----------|------|
| `ground_truth:=true` | 디버깅용 ground truth TF 프레임 활성화 |
| `start_aic_engine:=false` | 자동 Trial 오케스트레이션 비활성화 (자유 탐색) |
| `cable_type` | `sfp_sc_cable` 또는 `sfp_sc_cable_reversed` |

---

## 월드 상태 내보내기 (AI 훈련용)

시뮬레이션 월드 상태가 자동으로 `/tmp/aic.sdf`에 저장됨.

```bash
# 시나리오 저장
cp /tmp/aic.sdf ~/training_scenarios/scenario_001.sdf
```

- **IsaacLab**: `aic_utils/aic_isaac/` 가이드 참조
- **MuJoCo**: `aic_utils/aic_mujoco/` 가이드 참조 (MJCF 변환 지원)

---

## 훈련 전 F/T 센서 타링 (필수)

```bash
ros2 service call /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger
```
> 텔레오퍼레이션 또는 케이블 스폰 전에 반드시 실행.

---

*컨트롤러 안내: `aic_controller.md` / 태스크보드 상세: `task_board_description.md`*
