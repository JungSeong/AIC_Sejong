# AIC Sejong

[![Documentation](https://img.shields.io/badge/Documentation-GitHub%20Pages-0A66C2)](https://jungseong.github.io/contests/aic-sejong/)
[![Staged Policy](https://img.shields.io/badge/Staged%20Policy-Cable%20Insertion-0A66C2)](https://jungseong.github.io/contests/aic-sejong/#core-workflows)
[![Vision Pipeline](https://img.shields.io/badge/Vision%20Pipeline-YOLO%20%2B%20Stereo-5B5FC7)](https://jungseong.github.io/contests/aic-sejong/#core-workflows)
[![Data Workflow](https://img.shields.io/badge/Data%20Workflow-Recording%20%26%20Training-FFB000)](https://jungseong.github.io/contests/aic-sejong/#core-workflows)

[한국어](readme/README.ko.md) | [English](readme/README.en.md)

> AI for Industry Challenge (AIC) 참가를 위한 로컬 작업 공간.
> UR5e 로봇 팔로 케이블 삽입 태스크를 수행하는 정책을 개발·훈련·평가한다.

## Documentation

- [Getting Started](https://jungseong.github.io/contests/aic-sejong/#getting-started)
- [Core Workflows](https://jungseong.github.io/contests/aic-sejong/#core-workflows)

---

## 디렉토리 구조

```
AIC_Sejong/
├── README.md
│
└── ws_aic/                  ← 워크스페이스 루트
    ├── model/               ← 훈련된 모델 가중치
    │   ├── ais_yolo/        ← YOLO 모델 가중치
    │   │   └── weights/
    │   └── ais_distance_prediction/
    │
    └── src/                 ← 소스 루트 (pixi.toml 위치)
        ├── pixi.toml        ← 전체 워크스페이스 환경 정의
        ├── pixi.lock        ← 의존성 고정 파일
        │
        ├── aic/             ← AIC 공식 저장소 (git submodule)
        │   ├── aic_model/                  ← aic_model 노드 (정책 로더)
        │   ├── aic_adapter/                ← 모델 노드 어댑터 (C++)
        │   ├── aic_example_policies/       ← 예제 정책 (WaveArm, CheatCode, RunACT 등)
        │   ├── aic_bringup/                ← 시뮬레이션 launch 파일
        │   ├── aic_engine/                 ← 태스크 오케스트레이션 + 스코어링
        │   ├── aic_controller/             ← 로봇 팔 임피던스 컨트롤러
        │   ├── aic_interfaces/             ← ROS 2 메시지/서비스 인터페이스
        │   │   ├── aic_control_interfaces
        │   │   ├── aic_engine_interfaces
        │   │   ├── aic_model_interfaces
        │   │   └── aic_task_interfaces
        │   ├── aic_utils/
        │   │   ├── lerobot_robot_aic/      ← LeRobot ↔ AIC 연결 드라이버
        │   │   ├── aic_teleoperation/      ← 텔레오퍼레이션 유틸리티
        │   │   └── aic_mujoco/             ← MuJoCo 시뮬레이터 지원
        │
        ├── ais/             ← ★ 팀 자체 개발 패키지
        │   ├── ais_auto_capture/       ← YOLO 학습용 자동 데이터 수집
        │   ├── ais_early_prediction/   ← 조기 실패 예측 (Transformer 기반)
        │   ├── ais_eda/                ← 멀티뷰 편향 탐색적 데이터 분석
        │   ├── ais_encoder/            ← 멀티모달 표현 학습 (Vision + Touch)
        │   ├── ais_load_model_from_hf/ ← HuggingFace 모델 로드/업로드 유틸
        │   ├── ais_motion_planning/    ← YOLO + 스테레오 기반 포트 검출 및 모션 플래닝
        │   ├── docker/                     ← Dockerfile, docker-compose.yaml
        │   └── ais_ours_policy/        ← ROS 2 노드 래퍼
        │       ├── data_gen_node/      ← 데이터 생성 노드
        │       └── motion_planning_node/ ← 모션 플래닝 노드
        │
        ├── data/            ← 데이터셋
        │   ├── lerobot/     ← LeRobot 형식 데이터셋 (master 브랜치)
        │   └── yolo/        ← YOLO 학습 데이터 (날짜별, 예: 20260426/)
        │
        ├── docs/            ← 문서
            └── summaries/   ← Claude 세션별 요약 (0405, 0409, ... 0423)
```

---

## 초기 환경 설정

### 0. 시스템 요구사항

| 항목 | 최소 사양 |
|------|-----------|
| OS | Ubuntu 24.04 |
| CPU | 4코어 이상 |
| RAM | 32GB 이상 |
| GPU | NVIDIA RTX 2070 이상 |
| VRAM | 8GB 이상 |

---

### 1. Docker 설치

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # 로그아웃 후 재로그인 필요
```

---

### 2. NVIDIA Container Toolkit 설치 (GPU 사용 시)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

### 3. Distrobox 설치

```bash
sudo apt install distrobox
```

---

### 4. Pixi 설치

```bash
curl -fsSL https://pixi.sh/install.sh | sh
# 설치 후 터미널 재시작
```

---

### 5. 저장소 클론 및 의존성 설치

```bash
git clone https://github.com/JungSeong/AIC_Sejong.git ~/AIC_Sejong
cd ~/AIC_Sejong/ws_aic/src
pixi install
```

---

### 6. Eval 컨테이너 준비 (Docker eval 워크플로우)

```bash
export DBX_CONTAINER_MANAGER=docker

# 이미지 다운로드 및 컨테이너 생성 (최초 1회)
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
```

---

### 7. 환경변수 설정 (`~/.bashrc`에 등록 권장)

```bash
# HuggingFace 토큰 (모델 push/pull 시 필요)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

# LeRobot 데이터셋 저장 위치
export HF_LEROBOT_HOME=~/AIC_Sejong/ws_aic/src/data/lerobot

# HuggingFace 캐시 위치
export HF_HOME=~/.cache/huggingface
```

---

### 8. HuggingFace 로그인

```bash
cd ~/AIC_Sejong/ws_aic/src
pixi run hf auth login --token $HF_TOKEN
pixi run hf auth whoami   # 확인
```

---

### 9. 정책 노드 빌드

```bash
cd ~/AIC_Sejong/ws_aic/src
pixi install   # 변경 사항 반영
```

> `Baseline.py` 또는 `StagedPolicy.py` 수정 후에도 재실행 필요.

---

## 주요 워크플로우

### A. 시뮬레이션 + 정책 실행

```bash
# Terminal 1 — eval 컨테이너 (Gazebo + 엔진 + Zenoh 라우터)
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true

# Terminal 2 — StagedPolicy (Vision 통합 3단계 State Machine)
cd ~/AIC_Sejong/ws_aic/src
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=my_policy_node.StagedPolicy

# Terminal 2 (대안) — Baseline (ACT 기반)
cd ~/AIC_Sejong/ws_aic/src
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=my_policy_node.Baseline
```

**StagedPolicy 동작 방식:**

| 단계 | 방식 | 설명 |
|------|------|------|
| Stage 1 (이동) | YOLO + 스테레오 삼각측량 | 포트 위 10cm 지점까지 S-curve 보간 이동 |
| Stage 2 (삽입 접근) | Ground truth (임시) | 정밀 어프로치 |
| Stage 3 (삽입) | Ground truth (임시) | 케이블 삽입 완료 |

> `ground_truth=true` 시: TF로 포트 좌표 직접 읽음 (오차 0)
> `ground_truth=false` 시: YOLO 검출 + 스테레오 삼각측량 (오차 ~17mm)

---

### B. YOLO 학습 데이터 수집

```bash
# Terminal 1 — eval 컨테이너 (ground_truth=true 필수)
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=true start_aic_engine:=true

# Terminal 2 — 자동 캡처 노드
cd ~/AIC_Sejong/ws_aic/src
pixi run python ais/ais_motion_planning/collect_dataset.py
```

수집 데이터는 `data/yolo/<날짜>/` 아래 YOLO 형식으로 저장된다.

---

### C. LeRobot 훈련 데이터 녹화

```bash
cd ~/AIC_Sejong/ws_aic/src

# [중요!!] F/T 센서 tare (매 에피소드 전 필수)
pixi run ros2 service call /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger

pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee --teleop.id=aic \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=base_link \
  --dataset.repo_id=aic-sejong-team/AIC \
  --dataset.single_task="Cable insertion" \
  --dataset.push_to_hub=true \
  --dataset.private=true \
  --play_sounds=false \
  --display_data=true
```

| 키 | 동작 |
|----|------|
| `→` | 에피소드 완료 |
| `←` | 에피소드 취소 후 재녹화 |
| `ESC` | 녹화 종료 |

---

### D. ACT 모델 훈련

```bash
cd ~/AIC_Sejong/ws_aic/src
pixi run lerobot-train \
  --dataset.repo_id=aic-sejong-team/AIC \
  --policy.type=act \
  --output_dir=./model/ais_act \
  --job_name=act_AIC \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=aic-sejong-team/act_AIC
```

---

### E. YOLO 모델 훈련

```bash
cd ~/AIC_Sejong/ws_aic/src
pixi run python ais/ais_motion_planning/train_yolo.py
```

---

### F. 제출용 Docker 이미지 빌드

```bash
cd ~/AIC_Sejong/ws_aic/src/aic

# 빌드
docker compose -f docker/docker-compose.yaml build model

# 로컬 검증
docker compose -f docker/docker-compose.yaml up
```

---

## 참고 문서

| 문서 | 위치 |
|------|------|
| 공식 시작 가이드 | `ws_aic/src/aic/docs/getting_started.md` |
| 스코어링 규칙 | `ws_aic/src/aic/docs/scoring.md` |
| 정책 통합 가이드 | `ws_aic/src/aic/docs/policy.md` |
| LeRobot 연동 | `ws_aic/src/aic/aic_utils/lerobot_robot_aic/README.md` |
| Stage 1-B 케이블 장력 보상 실험 노트 | `ws_aic/src/aic/my_policy_node/STAGE1_MP2_NOTES.md` |
| 멀티모달 인코더 (Vision + Touch) | `ws_aic/src/ais/ais_encoder/README.md` |
| 모션 플래닝 패키지 | `ws_aic/src/ais/ais_motion_planning/` |
| 세션별 요약 | `ws_aic/src/docs/summaries/` |
| 논문 요약 | `ws_aic/src/docs/paper/` |
