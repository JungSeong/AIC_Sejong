# 월드 시나리오 파라미터 & 다양화 전략

> 평가에서 Trial 1~3이 무엇이 다른지, 그리고 훈련 데이터 다양화를 위해
> `/entrypoint.sh` 파라미터를 어떻게 조합해야 하는지 정리.
> 소스: `aic_engine/config/sample_config.yaml`, `aic_bringup/README.md`, `docs/task_board_description.md`, `docs/scoring.md`

---

## 1. Trial 1 vs Trial 2 vs Trial 3 차이

`sample_config.yaml`에 정의된 공식 평가 시나리오.

| 항목 | Trial 1 | Trial 2 | Trial 3 |
|------|---------|---------|---------|
| **태스크 보드 위치** | x=0.15, y=-0.2 | x=0.15, y=-0.2 | x=0.17, **y=0.0** |
| **태스크 보드 yaw** | 3.1415 (180°) | 3.1415 (180°) | **3.0** (~172°) |
| **삽입 대상 포트** | **SFP** (NIC 카드) | **SFP** (NIC 카드) | **SC** 포트 |
| **NIC 카드 레일** | **rail 0**, translation=0.036 | **rail 1**, translation=0.036 | 없음 |
| **SC 포트 레일** | rail 0, translation=0.042 | rail 0, translation=0.042 | **rail 1**, translation=-0.055 |
| **케이블 타입** | `sfp_sc_cable` | `sfp_sc_cable` | `sfp_sc_cable_reversed` |
| **플러그 종류** | SFP 끝 → 그리퍼 | SFP 끝 → 그리퍼 | **SC 끝 → 그리퍼** |

**핵심 차이 요약:**
- Trial 1 → 2: **NIC 카드 레일 위치만 변경** (rail 0 → rail 1). 태스크 유형은 동일(SFP 삽입).
- Trial 2 → 3: **완전히 다른 태스크.** SFP→NIC에서 SC→SC포트로 전환. 케이블 방향도 반전.

---

## 2. 변경 가능한 파라미터 전체 정리

### 태스크 보드 위치/자세

| 파라미터 | 기본값 | 평가 범위 | 설명 |
|----------|--------|-----------|------|
| `task_board_x` | 0.15 | - | 좌우 위치 (m) |
| `task_board_y` | -0.2 | - | 전후 위치 (m) |
| `task_board_z` | 1.14 | 고정 | 높이 (m) |
| `task_board_yaw` | 0.0 | **0.0 고정** (평가 시) | 회전각 (rad). 훈련 시 자유롭게 변경 가능 |
| `task_board_roll` | 0.0 | **0.0 고정** | 평가 시 항상 0 |
| `task_board_pitch` | 0.0 | **0.0 고정** | 평가 시 항상 0 |

> **평가 시 주의:** roll, pitch는 항상 0. yaw도 0으로 고정. 훈련 시에는 다양화 가능.

---

### NIC 카드 마운트 (Zone 1 — SFP 삽입 대상)

레일 0~4 중 하나에 NIC 카드 배치. 각 레일에 독립 파라미터.

| 파라미터 | 범위 | 설명 |
|----------|------|------|
| `nic_card_mount_N_present` | true/false | N번 레일에 NIC 카드 존재 여부 |
| `nic_card_mount_N_translation` | **[-0.0215, 0.0234] m** | 레일 위 NIC 카드 위치 |
| `nic_card_mount_N_yaw` | **[-10°, +10°] (rad)** | NIC 카드 회전 |

```bash
# 예: NIC rail 2에 중앙보다 오른쪽으로 배치
nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.015 nic_card_mount_2_yaw:=0.1
```

---

### SC 포트 (Zone 2 — SC 삽입 대상)

레일 0, 1에 각각 SC 포트 배치 가능.

| 파라미터 | 범위 | 설명 |
|----------|------|------|
| `sc_port_N_present` | true/false | N번 레일에 SC 포트 존재 여부 |
| `sc_port_N_translation` | **[-0.06, 0.055] m** | 레일 위 SC 포트 위치 |
| `sc_port_N_yaw` | **0.0 고정** (평가 시) | 평가 시 항상 0, 훈련 시 변경 가능 |

```bash
# 예: SC rail 1에 왼쪽으로 배치
sc_port_1_present:=true sc_port_1_translation:=-0.04
```

---

### 케이블 타입

| `cable_type` 값 | 그리퍼에 잡히는 플러그 | 삽입 대상 |
|-----------------|----------------------|-----------|
| `sfp_sc_cable` | **SFP 끝** | NIC 카드 SFP 포트 |
| `sfp_sc_cable_reversed` | **SC 끝** | SC 포트 |

```bash
# SFP 삽입 시나리오
cable_type:=sfp_sc_cable attach_cable_to_gripper:=true

# SC 삽입 시나리오
cable_type:=sfp_sc_cable_reversed attach_cable_to_gripper:=true
```

---

### 픽 위치 마운트 (Zone 3 & 4 — 픽업 위치)

| 파라미터 | 범위 | 설명 |
|----------|------|------|
| `lc_mount_rail_N_present` | true/false | LC 마운트 존재 여부 |
| `sfp_mount_rail_N_present` | true/false | SFP 마운트 존재 여부 |
| `sc_mount_rail_N_present` | true/false | SC 마운트 존재 여부 |
| `*_mount_rail_N_translation` | **[-0.09425, 0.09425] m** | 레일 위 마운트 위치 |
| `*_mount_rail_N_yaw` | **[-60°, +60°] (rad)** | 마운트 회전 |

---

## 3. 훈련 시나리오 다양화 전략

스코어링 구조상 **Tier 3(삽입 성공)이 최대 75점**으로 압도적. 모델이 다양한 위치에서도 삽입할 수 있도록 아래 변수들을 무작위화해야 한다.

### 다양화 우선순위

| 우선순위 | 변수 | 이유 |
|---------|------|------|
| ★★★ | **NIC/SC 위치 (translation)** | 평가마다 달라지는 핵심 변수. 이걸 못 따라가면 0점 |
| ★★★ | **케이블 타입 (sfp vs sc)** | Trial 3은 완전히 다른 태스크 |
| ★★ | **NIC 레일 번호 (0~4)** | Trial 1→2 차이의 핵심 |
| ★★ | **태스크 보드 yaw** | 보드 정면 방향이 달라지면 접근 방향 전체가 바뀜 |
| ★ | **태스크 보드 x, y** | 거리 차이 → 궤적 효율 점수에 영향 |

### 시나리오 예시

```bash
# ── SFP 삽입 시나리오 A (NIC rail 0, 왼쪽) ──
/entrypoint.sh \
  spawn_task_board:=true \
  task_board_x:=0.15 task_board_y:=-0.2 task_board_z:=1.14 \
  task_board_yaw:=3.1415 \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.02 \
  spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \
  ground_truth:=true start_aic_engine:=false

# ── SFP 삽입 시나리오 B (NIC rail 2, 오른쪽, 보드 각도 변경) ──
/entrypoint.sh \
  spawn_task_board:=true \
  task_board_x:=0.17 task_board_y:=-0.15 task_board_z:=1.14 \
  task_board_yaw:=2.8 \
  nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.02 \
  spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \
  ground_truth:=true start_aic_engine:=false

# ── SC 삽입 시나리오 (Trial 3 유형) ──
/entrypoint.sh \
  spawn_task_board:=true \
  task_board_x:=0.17 task_board_y:=0.0 task_board_z:=1.14 \
  task_board_yaw:=3.0 \
  sc_port_1_present:=true sc_port_1_translation:=-0.04 \
  spawn_cable:=true cable_type:=sfp_sc_cable_reversed attach_cable_to_gripper:=true \
  ground_truth:=true start_aic_engine:=false
```

### 월드 저장 및 재사용

```bash
# 시나리오 생성 후 자동으로 /tmp/aic.sdf에 저장됨
cp /tmp/aic.sdf ~/aic_sejong/aic_data/scenarios/sfp_rail2_right.sdf

# IsaacLab, MuJoCo 등에서 재사용 가능
```

---

## 4. 평가 vs 훈련 파라미터 차이

| 항목 | 평가 시 | 훈련 시 |
|------|---------|---------|
| `roll`, `pitch` | **항상 0.0** | 자유롭게 변경 가능 |
| `task_board_yaw` | 3.0~3.1415 범위 | 어떤 각도든 가능 |
| `sc_port_yaw` | **항상 0.0** | 자유롭게 변경 가능 |
| `start_aic_engine` | `true` | `false` (탐색 시), `true` (점수 확인 시) |
| `ground_truth` | `false` | `true` (디버깅), `false` (실제 평가와 동일) |
