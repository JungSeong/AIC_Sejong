# LeRobot 데이터 녹화 → 훈련 → 모델 생성 정리

> `lerobot-record`로 녹화 시 저장되는 데이터의 내용과 위치, 그리고 `lerobot-train`으로 ACT 모델을 만드는 전체 흐름을 정리한다.
> 소스: `aic_utils/lerobot_robot_aic/` 코드 및 README, `docs/participant_utilities.md`

---

## 1. 데이터 녹화 명령 (lerobot_rec)

```bash
# 시작 전 F/T 센서 반드시 tare
pixi run ros2 service call /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger

pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee --teleop.id=aic \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=base_link \
  --dataset.repo_id=JungSeong2/AIC \
  --dataset.single_task="Simple Test with LeRobot" \
  --dataset.push_to_hub=false \
  --dataset.private=true \
  --play_sounds=false \
  --display_data=true
```

**녹화 중 키 조작:**

| 키 | 동작 |
|----|------|
| `→` (Right Arrow) | 에피소드 완료 → 다음 에피소드 |
| `←` (Left Arrow) | 현재 에피소드 취소 후 재녹화 |
| `ESC` | 녹화 종료 |

---

## 2. 저장 위치

**`--dataset.push_to_hub=false` 설정 시 로컬 저장:**

```
~/.cache/huggingface/lerobot/JungSeong2/AIC/
```

> `--dataset.repo_id`의 값이 곧 로컬 경로 하위 폴더가 된다.
> `--dataset.push_to_hub=true`로 설정하면 HuggingFace Hub에 자동 업로드.

---

## 3. 저장되는 데이터 구조

LeRobot은 HuggingFace `datasets` 라이브러리 형식으로 저장한다. 에피소드별로 아래 데이터가 기록된다.

### 3-1. Observation (상태 데이터)

`get_observation()`에서 수집. **총 27개 scalar + 카메라 3채널.**

**로봇 상태 (27 scalar):**

| 키 | 내용 | 차원 |
|----|------|------|
| `tcp_pose.position.{x,y,z}` | TCP 위치 | 3 |
| `tcp_pose.orientation.{x,y,z,w}` | TCP 방향 (quaternion) | 4 |
| `tcp_velocity.linear.{x,y,z}` | TCP 선속도 | 3 |
| `tcp_velocity.angular.{x,y,z}` | TCP 각속도 | 3 |
| `tcp_error.{x,y,z,rx,ry,rz}` | TCP 위치/자세 오차 | 6 |
| `joint_positions.{0~6}` | 관절 위치 (7개 관절) | 7 |

**카메라 이미지 (3채널):**

| 카메라 | 원본 해상도 | 저장 해상도 (scale=0.25) |
|--------|------------|--------------------------|
| `left_camera` | 1152 × 1024 | **288 × 256** |
| `center_camera` | 1152 × 1024 | **288 × 256** |
| `right_camera` | 1152 × 1024 | **288 × 256** |

> 이미지 축소 비율은 `AICRobotAICControllerConfig.camera_image_scaling`에서 조정 가능 (기본 0.25).

### 3-2. Action (액션 데이터)

텔레오퍼레이션 입력이 그대로 기록된다. **`teleop_target_mode`에 따라 형식이 다르다.**

**Cartesian 모드 (`--robot.teleop_target_mode=cartesian`):**

| 키 | 내용 |
|----|------|
| `linear.x`, `linear.y`, `linear.z` | TCP 선속도 명령 |
| `angular.x`, `angular.y`, `angular.z` | TCP 각속도 명령 |

**Joint 모드 (`--robot.teleop_target_mode=joint`):**

| 키 | 내용 |
|----|------|
| `shoulder_pan_joint` ~ `wrist_3_joint` | 관절 속도 명령 (6개) |

### 3-3. 파일 포맷 (LeRobot 표준)

```
~/.cache/huggingface/lerobot/JungSeong2/testv1/
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet   ← 상태/액션 시계열 (tabular)
│   │   ├── ...
├── videos/
│   ├── chunk-000/
│   │   ├── observation.images.left_camera_episode_000000.mp4
│   │   ├── observation.images.center_camera_episode_000000.mp4
│   │   ├── observation.images.right_camera_episode_000000.mp4
│   │   ├── ...
├── meta/
│   ├── info.json       ← 데이터셋 메타 (fps, shape, feature 정보 등)
│   ├── stats.json      ← 각 feature의 mean/std (훈련 정규화에 사용)
│   ├── tasks.jsonl     ← task prompt 목록
│   └── episodes.jsonl  ← 에피소드별 메타
```

> **`stats.json`**: 훈련 시 RunACT의 `policy_preprocessor_step_3_normalizer_processor.safetensors`와 같은 역할. 이미지/상태/액션의 mean·std가 여기서 나온다.

---

## 4. 훈련 과정 (lerobot-train)

데이터셋이 준비되면 아래 명령으로 바로 훈련 가능.

```bash
cd ~/ws_aic/src/aic
pixi run lerobot-train \
  --dataset.repo_id=JungSeong2/testv1 \
  --policy.type=act \
  --output_dir=outputs/train/act_testv1 \
  --job_name=act_testv1 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=JungSeong2/act_testv1
```

**주요 파라미터:**

| 파라미터 | 설명 |
|----------|------|
| `--dataset.repo_id` | 녹화한 데이터셋 경로 (로컬 캐시 or HF Hub) |
| `--policy.type` | 정책 아키텍처 (`act`, `diffusion`, `tdmpc` 등) |
| `--output_dir` | 체크포인트 저장 경로 |
| `--policy.device` | `cuda` or `cpu` |
| `--wandb.enable` | WandB 로깅 여부 |
| `--policy.repo_id` | 완료 후 HF Hub 업로드 대상 |

**훈련 출력 구조:**

```
outputs/train/act_testv1/
├── checkpoints/
│   ├── 000500/               ← 스텝별 체크포인트
│   │   ├── pretrained_model/
│   │   │   ├── config.json
│   │   │   └── model.safetensors   ← RunACT에서 로드하는 파일
│   │   └── training_state/
│   └── last/                 ← 최신 체크포인트
└── ...
```

---

## 5. 전체 흐름 요약

```
[텔레오퍼레이션 녹화]
  lerobot-record
       │
       ▼
~/.cache/huggingface/lerobot/<repo_id>/
  ├── data/*.parquet   (상태 27차원 + 액션 6차원)
  ├── videos/*.mp4     (카메라 3채널, 288×256)
  └── meta/stats.json  (정규화 통계)
       │
       ▼
[훈련]
  lerobot-train --policy.type=act
       │
       ▼
outputs/train/.../checkpoints/.../
  └── model.safetensors   ← RunACT.py에서 로드
       │
       ▼
[추론 / 평가]
  pixi run ros2 run aic_model aic_model \
    --ros-args -p policy:=aic_example_policies.ros.RunACT
```

---

## 참고: RunACT ↔ 녹화 데이터 대응 관계

| lerobot-record 저장 키 | RunACT 입력 텐서 |
|------------------------|-----------------|
| `tcp_pose.*` + `tcp_velocity.*` + `tcp_error.*` + `joint_positions.*` | `observation.state` (26차원) |
| `left_camera` 이미지 | `observation.images.left_camera` |
| `center_camera` 이미지 | `observation.images.center_camera` |
| `right_camera` 이미지 | `observation.images.right_camera` |
| `linear.*`, `angular.*` (액션) | 모델 출력 → Twist 명령 |
