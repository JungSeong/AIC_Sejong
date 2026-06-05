# AIC Sejong

[![Documentation](https://img.shields.io/badge/Documentation-GitHub%20Pages-0A66C2)](https://jungseong.github.io/contests/aic-sejong/)
[![Staged Policy](https://img.shields.io/badge/Staged%20Policy-Cable%20Insertion-0A66C2)](https://jungseong.github.io/contests/aic-sejong/#core-workflows)
[![Vision Pipeline](https://img.shields.io/badge/Vision%20Pipeline-YOLO%20%2B%20Stereo-5B5FC7)](https://jungseong.github.io/contests/aic-sejong/#core-workflows)
[![Data Workflow](https://img.shields.io/badge/Data%20Workflow-Recording%20%26%20Training-FFB000)](https://jungseong.github.io/contests/aic-sejong/#core-workflows)

[한국어](readme/README.ko.md) | [English](readme/README.en.md)

AI for Industry Challenge 참가를 위한 로컬 작업 공간. UR5e 로봇 팔로 케이블 삽입 태스크를 수행하는 정책을 개발·훈련·평가한다.

## 문서

- [시작하기](https://jungseong.github.io/contests/aic-sejong/#getting-started)
- [주요 워크플로우](https://jungseong.github.io/contests/aic-sejong/#core-workflows)

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
        │   └── aic_utils/
        │       ├── lerobot_robot_aic/      ← LeRobot ↔ AIC 연결 드라이버
        │       ├── aic_teleoperation/      ← 텔레오퍼레이션 유틸리티
        │       └── aic_mujoco/             ← MuJoCo 시뮬레이터 지원
        │
        ├── ais/             ← ★ 팀 자체 개발 패키지
        │   ├── ais_auto_capture/       ← YOLO 학습용 자동 데이터 수집
        │   ├── ais_early_prediction/   ← 조기 실패 예측 (Transformer 기반)
        │   ├── ais_eda/                ← 멀티뷰 편향 탐색적 데이터 분석
        │   ├── ais_encoder/            ← 멀티모달 표현 학습 (Vision + Touch)
        │   ├── ais_load_model_from_hf/ ← HuggingFace 모델 로드/업로드 유틸
        │   ├── ais_motion_planning/    ← YOLO + 스테레오 기반 포트 검출 및 모션 플래닝
        │   ├── docker/                 ← Dockerfile, docker-compose.yaml
        │   └── ais_ours_policy/        ← ROS 2 노드 래퍼
        │       ├── data_gen_node/      ← 데이터 생성 노드
        │       └── motion_planning_node/ ← 모션 플래닝 노드
        │
        ├── data/            ← 데이터셋
        │   ├── lerobot/     ← LeRobot 형식 데이터셋 (master 브랜치)
        │   └── yolo/        ← YOLO 학습 데이터 (날짜별, 예: 20260426/)
        │
        └── docs/            ← 문서
            └── summaries/   ← Claude 세션별 요약 (0405, 0409, ... 0423)
```

## 시작하기

### 요구사항

- Ubuntu 24.04
- NVIDIA GPU 권장
- Docker, NVIDIA Container Toolkit, Distrobox, Pixi

### 의존성 설치

```bash
git clone https://github.com/JungSeong/AIC_Sejong.git ~/AIC_Sejong
cd ~/AIC_Sejong/ws_aic/src
pixi install
```

### Eval 컨테이너 준비

```bash
export DBX_CONTAINER_MANAGER=docker
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
```

## 주요 워크플로우

### 시뮬레이션 + 정책 실행

```bash
# Terminal 1
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true

# Terminal 2
cd ~/AIC_Sejong/ws_aic/src
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=my_policy_node.StagedPolicy
```

### YOLO 데이터 수집

```bash
cd ~/AIC_Sejong/ws_aic/src
pixi run python ais/ais_motion_planning/collect_dataset.py
```

### ACT 훈련

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

## 참고 문서

- 공식 시작 가이드: `ws_aic/src/aic/docs/getting_started.md`
- 스코어링 규칙: `ws_aic/src/aic/docs/scoring.md`
- 정책 가이드: `ws_aic/src/aic/docs/policy.md`
- 모션 플래닝 패키지: `ws_aic/src/ais/ais_motion_planning/`
