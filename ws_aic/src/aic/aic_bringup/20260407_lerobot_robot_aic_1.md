# lerobot_robot_aic README 요약

**날짜:** 2026-04-07
**주제:** lerobot_robot_aic — LeRobot 인터페이스 (텔레오퍼레이션 · 데이터 수집 · 학습)
**파일:** `ws_aic/src/aic/aic_utils/lerobot_robot_aic/README.md`

---

## 핵심 한 줄 요약

`lerobot_robot_aic`는 HuggingFace LeRobot을 AIC 로봇에 연결하는 인터페이스 패키지로, 키보드/SpaceMouse 텔레오퍼레이션 → 데이터 수집 → 정책 학습 전 과정을 Pixi 환경에서 수행한다.

---

## 주요 기능 3가지

### 1. 텔레오퍼레이션 (`lerobot-teleoperate`)

```bash
cd ~/ws_aic/src/aic
pixi run lerobot-teleoperate \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=<teleop-type> --teleop.id=aic \
  --robot.teleop_target_mode=<mode> --robot.teleop_frame_id=<frame_id> \
  --display_data=true
```

**텔레오퍼레이션 타입 조합:**

| `--teleop.type` | `--teleop_target_mode` | 설명 |
|----------------|------------------------|------|
| `aic_keyboard_ee` | `cartesian` | 키보드 카르테시안 제어 |
| `aic_spacemouse` | `cartesian` | SpaceMouse 카르테시안 제어 |
| `aic_keyboard_joint` | `joint` | 키보드 관절 제어 |

> ⚠️ `--teleop.type`과 `--robot.teleop_target_mode`는 **반드시 함께** 설정해야 함.

**카르테시안 기준 프레임 (`--robot.teleop_frame_id`):**
- `gripper/tcp` — 그리퍼 TCP 기준 (기본값)
- `base_link` — 로봇 베이스 기준

---

#### 키보드 카르테시안 키맵

| 키 | 동작 |
|----|------|
| w/s | -/+ linear y |
| a/d | -/+ linear x |
| r/f | -/+ linear z |
| q/e | -/+ angular z |
| Shift+w/s | +/- angular x |
| Shift+a/d | -/+ angular y |
| t | slow/fast 속도 토글 |

> ⚠️ Shift+키 종료 시: Shift보다 키를 **먼저** 놓아야 함 (반대 순서면 회전 지속됨)

#### 키보드 관절 키맵

| 키 | 관절 |
|----|------|
| q/a | shoulder_pan |
| w/s | shoulder_lift |
| e/d | elbow |
| r/f | wrist_1 |
| t/g | wrist_2 |
| y/h | wrist_3 |
| u | slow/fast 속도 토글 |

#### SpaceMouse
- 3Dconnexion SpaceMouse + `pyspacemouse` 라이브러리 사용
- USB 권한 설정 필요 (`/etc/udev/rules.d/99-spacemouse.rules`)
- 키보드보다 반응이 느릴 수 있음 (실 사용 경험 기준)

---

### 2. 학습 데이터 수집 (`lerobot-record`)

```bash
cd ~/ws_aic/src/aic
pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=<teleop-type> --teleop.id=aic \
  --robot.teleop_target_mode=<mode> --robot.teleop_frame_id=<frame_id> \
  --dataset.repo_id=<hf-repo> \
  --dataset.single_task=<task-prompt> \
  --dataset.push_to_hub=false \
  --dataset.private=true \
  --display_data=true
```

**레코딩 조작 키:**

| 키 | 동작 |
|----|------|
| → | 다음 에피소드 |
| ← | 현재 에피소드 취소 후 재녹화 |
| ESC | 녹화 종료 |

> `WARN Watchdog Validator ...` 메시지는 무시해도 됨. `INFO ... Recording episode 0` 확인이 정상 시작.

---

### 3. 학습 (`lerobot-train`)

```bash
cd ~/ws_aic/src/aic
pixi run lerobot-train \
  --dataset.repo_id=${HF_USER}/your_dataset \
  --policy.type=your_policy_type \
  --output_dir=outputs/train/act_your_dataset \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/act_policy
```

데이터셋 준비 후 LeRobot 공식 튜토리얼 가이드라인을 따름.

---

## 핵심 포인트 (3줄 요약)

1. **Pixi 환경** 안에서 `lerobot-teleoperate` → `lerobot-record` → `lerobot-train` 순서로 전체 워크플로우가 완결된다.
2. **텔레오퍼레이션 타입**(`--teleop.type`)과 **제어 모드**(`--teleop_target_mode`)는 항상 쌍으로 지정해야 하며, 카르테시안/관절 제어에 따라 키맵이 완전히 달라진다.
3. **데이터셋은 HuggingFace Hub에 업로드**(`push_to_hub`)하거나 로컬 저장(`false`) 선택 가능하며, 학습 후 정책도 HF Hub(`policy.repo_id`)에 배포할 수 있다.
