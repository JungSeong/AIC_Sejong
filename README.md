# AIC Sejong — 작업 공간 가이드

> AI for Industry Challenge (AIC) 참가를 위한 로컬 작업 공간.
> UR5e 로봇 팔로 케이블 삽입 태스크를 수행하는 정책을 개발·훈련·평가한다.

---

## 디렉토리 구조

```
aic_sejong/
├── ABOUT-ME/
│   ├── about-me.md          ← 개인 배경 및 목표
│   └── working-rules.md     ← Claude 작업 규칙 (응답 형식, 파일 저장 규칙 등)
│
├── CLAUDE.md                ← Claude 컨텍스트 진입점 (about-me + working-rules 참조)
│
├── CLAUDE-OUTPUTS/
│   └── summaries/
│       ├── 0405/            ← docs/ 전체 문서 요약 (getting_started, scoring 등 20개)
│       ├── 0409/            ← 맵 파일, 예제 정책, LeRobot 녹화·훈련 요약
│       ├── 0412/            ← 정책 노드 생성, eval 환경 설정, 훈련 파라미터 요약
│       └── 0413/            ← Docker eval 실행 가이드
│
├── TEMPLATES/
│   └── basic-templates.md   ← 코드/문서 작성 템플릿
│
├── aic_data/
│   ├── huggingface/
│   │   ├── hub/
│   │   │   └── models--JungSeong2--act_AIC/   ← HF에서 받은 모델 캐시
│   │   └── lerobot/
│   │       └── calibration/                   ← LeRobot 캘리브레이션 데이터
│   └── outputs/
│       └── train/
│           └── act_AIC/                       ← lerobot-train 훈련 출력 (체크포인트)
│
└── ws_aic/
    └── src/
        └── aic/                               ← AIC 공식 저장소 (git submodule)
            ├── pixi.toml                      ← 전체 환경 의존성 정의
            ├── pixi.lock                      ← 의존성 고정 파일 (Docker 빌드에 필수)
            ├── docs/                          ← 공식 문서 (overview, scoring 등)
            ├── aic_model/                     ← aic_model 노드 (정책 로더)
            ├── aic_example_policies/          ← 예제 정책 (WaveArm, CheatCode, RunACT 등)
            ├── aic_bringup/                   ← 시뮬레이션 launch 파일
            ├── aic_engine/                    ← 태스크 오케스트레이션 + 스코어링
            ├── aic_controller/                ← 로봇 팔 임피던스 컨트롤러
            ├── aic_interfaces/                ← ROS 2 메시지/서비스 인터페이스
            ├── aic_utils/
            │   └── lerobot_robot_aic/         ← LeRobot ↔ AIC 연결 드라이버
            ├── docker/                        ← Dockerfile, docker-compose.yaml
            └── my_policy_node/                ← ★ 내 정책 패키지
                ├── pixi.toml                  ← 패키지 의존성
                ├── package.xml
                └── my_policy_node/
                    └── Baseline.py            ← ACT 기반 정책 구현체
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
# Docker Engine 설치
curl -fsSL https://get.docker.com | sh

# 비루트 사용자 권한 추가 (로그아웃 후 재로그인 필요)
sudo usermod -aG docker $USER
```

---

### 2. NVIDIA Container Toolkit 설치 (GPU 사용 시)

```bash
# 툴킷 설치
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit

# Docker에 NVIDIA 런타임 등록
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
cd ~
git clone https://github.com/JungSeong/AIC_Sejong.git
cd ~/AIC_Sejong/ws_aic/src/aic
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

### 7. 환경변수 설정 (`~/.bashrc`에 등록해서 관리하시는 것을 권장합니다)

```bash
# HuggingFace 토큰 (lerobot-record push / snapshot_download 에 필요)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

# LeRobot 데이터셋 저장 위치 변경 (기본: ~/.cache/huggingface/lerobot)
export HF_LEROBOT_HOME=/home/swlinux/aic_sejong/aic_data/huggingface/lerobot

# HuggingFace 캐시 위치 변경 (기본: ~/.cache/huggingface)
export HF_HOME=/home/swlinux/aic_sejong/aic_data/huggingface

# 훈련 출력 위치는 lerobot-train 실행 시 --output_dir 으로 지정
# 예: --output_dir=/home/swlinux/aic_sejong/aic_data/outputs/train/act_AIC
```

---

### 8. HuggingFace 로그인

```bash
cd ~/aic_sejong/ws_aic/src/aic
pixi run hf auth login --token $HF_TOKEN

# 확인
pixi run hf auth whoami
```

---

### 9. 내 정책 노드 빌드

```bash
cd ~/aic_sejong/ws_aic/src/aic
pixi reinstall ros-kilted-my-policy-node
```

> 코드(`Baseline.py`) 수정 후에도 반드시 재실행.

---

## 주요 워크플로우

### A. 시뮬레이션 + 정책 실행

```bash
# Terminal 1 — eval 컨테이너 (Gazebo + 엔진 + Zenoh 라우터 포함)
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true

# Terminal 2 — 내 정책 (30초 이내 실행, source 불필요)
cd ~/aic_sejong/ws_aic/src/aic
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=my_policy_node.Baseline
```

---

### B. 훈련 데이터 녹화

```bash
cd ~/aic_sejong/ws_aic/src/aic

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

### C. 모델 훈련

```bash
cd ~/aic_sejong/ws_aic/src/aic
pixi run lerobot-train \
  --dataset.repo_id=JungSeong2/AIC \
  --policy.type=act \
  --output_dir=/home/swlinux/aic_sejong/aic_data/outputs/train/act_AIC \
  --job_name=act_AIC \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=aic-sejong-team/act_AIC
```

---

### D. 제출용 Docker 이미지 빌드

```bash
cd ~/aic_sejong/ws_aic/src/aic

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
| 시나리오 생성 파라미터 정리 | `CLAUDE-OUTPUTS/summaries/0413/summary_world-scenario-params_v1.md` |
| 훈련 모델 기반 정책 노드 생성 가이드 | `CLAUDE-OUTPUTS/summaries/0412/summary_trained-model-to-policy_v1.md` |
| LeRobot 연동 | `ws_aic/src/aic/aic_utils/lerobot_robot_aic/README.md` |
| LeRobot 녹화·훈련 가이드 | `CLAUDE-OUTPUTS/summaries/0409/summary_lerobot-record-and-train_v2.md` |
| LeRobot 녹화 파라미터 | `CLAUDE-OUTPUTS/summaries/0413/summary_teleop-params_v1.md` |
| LeRobot 훈련 파라미터 | `CLAUDE-OUTPUTS/summaries/0412/summary_lerobot-train-params_v1.md` |
