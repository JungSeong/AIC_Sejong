# LeRobot 데이터 녹화 → 훈련 → 모델 생성 정리 (v2)

> `lerobot-record`로 녹화 시 저장되는 데이터의 내용·위치, 올바른 녹화를 위한 주의사항,
> 저장 경로 변경 방법, `policy.type` 선택지까지 포함한 전체 흐름 정리.
> 소스: `lerobot_robot_aic/` 코드, `lerobot/utils/constants.py`, `lerobot/policies/factory.py`, `lerobot/configs/train.py`

---

## 1. 데이터 녹화 명령

```bash
# 매 에피소드 시작 전 F/T 센서 반드시 tare
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
| `→` | 에피소드 완료 → 다음 에피소드 |
| `←` | **현재 에피소드 취소 후 재녹화** |
| `ESC` | 전체 녹화 종료 |

---

## 2. 올바른 녹화를 위해 중요한 것

훈련 품질은 데이터 품질에 직결된다. 아래 항목을 반드시 지킬 것.

| 항목 | 이유 |
|------|------|
| **에피소드 시작 전 F/T 센서 tare** | 케이블 무게 등 외란 보정 안 되면 학습 데이터 오염 |
| **동일한 텔레오퍼레이션 파라미터 유지** | `teleop_target_mode`, `frame_id`가 바뀌면 액션 분포가 달라져 학습 불가 |
| **실패 에피소드는 `←`로 즉시 취소** | 나쁜 궤적이 데이터셋에 섞이면 모델이 실패 동작을 학습 |
| **부드럽고 일관된 동작** | 급격한 조작·정지는 이상치로 학습을 방해 |
| **케이블이 그리퍼에 제대로 잡힌 상태에서 시작** | 초기 상태의 일관성이 policy의 일반화에 중요 |
| **충분한 에피소드 수 확보** | ACT 기준 최소 수십 개 이상 권장 |

---

## 3. 저장되는 데이터

### 3-1. 저장 위치 (기본값)

```
~/.cache/huggingface/lerobot/JungSeong2/AIC/
```

결정 구조:

```
HF_HOME (기본: ~/.cache/huggingface)
  └── lerobot/
        └── <dataset.repo_id>/    ← 여기에 저장
              JungSeong2/AIC/
```

### 3-2. 저장 경로 변경 방법

| 환경변수 | 역할 | 예시 |
|----------|------|------|
| `HF_LEROBOT_HOME` | **데이터셋 저장 루트 경로** 변경 | `export HF_LEROBOT_HOME=/data/lerobot` |
| `HF_HOME` | HuggingFace 전체 캐시 루트 변경 | `export HF_HOME=/data/huggingface` |
| ~~`LEROBOT_HOME`~~ | deprecated → `HF_LEROBOT_HOME` 사용 | |

적용 예시:

```bash
export HF_LEROBOT_HOME=/media/swlinux/Etc\ Data/lerobot_datasets
pixi run lerobot-record ...
```

> `~/.bashrc`나 `~/.zshrc`에 추가하면 영구 적용.

### 3-3. 저장 파일 구조

```
~/.cache/huggingface/lerobot/JungSeong2/AIC/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet   ← 상태 + 액션 시계열 (tabular)
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.left_camera_episode_000000.mp4
│       ├── observation.images.center_camera_episode_000000.mp4
│       ├── observation.images.right_camera_episode_000000.mp4
│       └── ...
└── meta/
    ├── info.json        ← fps, shape, feature 목록
    ├── stats.json       ← mean/std (훈련 정규화 통계)
    ├── tasks.jsonl      ← task prompt
    └── episodes.jsonl   ← 에피소드별 메타
```

### 3-4. 저장 데이터 내용

**로봇 상태 (27 scalar, Parquet):**

| 키 | 내용 | 차원 |
|----|------|------|
| `tcp_pose.position.{x,y,z}` | TCP 위치 | 3 |
| `tcp_pose.orientation.{x,y,z,w}` | TCP 방향 | 4 |
| `tcp_velocity.linear.{x,y,z}` | TCP 선속도 | 3 |
| `tcp_velocity.angular.{x,y,z}` | TCP 각속도 | 3 |
| `tcp_error.{x,y,z,rx,ry,rz}` | TCP 오차 | 6 |
| `joint_positions.{0~6}` | 관절 위치 | 7 |

**액션 (Cartesian 모드, 6 scalar, Parquet):**

| 키 | 내용 |
|----|------|
| `linear.{x,y,z}` | TCP 선속도 명령 |
| `angular.{x,y,z}` | TCP 각속도 명령 |

**카메라 이미지 (mp4):**

| 카메라 | 저장 해상도 (scale 0.25 적용) |
|--------|-------------------------------|
| `left_camera` | 288 × 256 |
| `center_camera` | 288 × 256 |
| `right_camera` | 288 × 256 |

---

## 4. 훈련 (lerobot-train)

```bash
cd ~/ws_aic/src/aic
pixi run lerobot-train \
  --dataset.repo_id=JungSeong2/AIC \
  --policy.type=act \
  --output_dir=outputs/train/act_AIC \
  --job_name=act_AIC \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=JungSeong2/act_AIC
```

### 훈련 출력 위치 (기본값)

`--output_dir` 미지정 시 **명령 실행 디렉토리** 기준으로 자동 생성:

```
<실행 위치>/outputs/train/<job_name>/
  └── checkpoints/
        ├── 000500/
        │   └── pretrained_model/
        │       ├── config.json
        │       └── model.safetensors   ← RunACT가 로드하는 파일
        └── last/
```

> **출력 경로를 바꾸려면:** `--output_dir=/원하는/경로` 지정.

---

## 5. policy.type 선택지

`lerobot/policies/factory.py`의 `get_policy_class(name)`에서 문자열로 매핑된다.

| `policy.type` 값 | 설명 |
|-----------------|------|
| **`act`** | **Action Chunking with Transformers** — AIC 기본 예시 (RunACT) |
| `diffusion` | Diffusion Policy |
| `pi0` | π₀ (물리 기반 VLA) |
| `pi05` | π₀.5 |
| `smolvla` | SmolVLA (소형 VLA) |
| `tdmpc` | TD-MPC (Model-based RL) |
| `vqbet` | VQ-BeT |
| `sac` | Soft Actor-Critic (온라인 RL) |
| `groot` | GR00T |
| `wall_x` | Wall-X |
| `xvla` | xVLA |

> AIC 케이블 삽입 태스크에는 **`act`** 가 기본 선택. 데이터셋이 클수록 `diffusion`, `pi0` 계열도 고려 가능.

---

## 6. 전체 흐름 요약

```
[녹화]
  lerobot-record
  (F/T tare → 조작 → → 저장 / ← 취소)
        │
        ▼
$HF_LEROBOT_HOME/JungSeong2/AIC/
  ├── data/*.parquet    (상태 27차원 + 액션 6차원)
  ├── videos/*.mp4      (카메라 3채널, 288×256)
  └── meta/stats.json   (정규화 통계 → 훈련에 자동 사용)
        │
        ▼
[훈련]
  lerobot-train --policy.type=act
        │
        ▼
outputs/train/act_AIC/checkpoints/.../
  └── model.safetensors
        │
        ▼
[추론]
  RunACT.py → snapshot_download("JungSeong2/act_AIC")
  → policy.select_action() → Cartesian Twist → 로봇 이동
```
