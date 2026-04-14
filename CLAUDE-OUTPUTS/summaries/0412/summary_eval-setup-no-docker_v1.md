# 평가 환경 셋업 (도커 없이 소스 빌드) 요약

> **요약 (2-3줄):** AIC 평가 환경은 기본적으로 `aic_eval` Docker 컨테이너 사용을 권장하지만, 고급 사용자는 Ubuntu 24.04에서 소스로 직접 빌드할 수 있다. 핵심은 ROS 2 Kilted + Gazebo 소스 빌드 + `rmw_zenoh_cpp` 미들웨어 설정이며, 3개의 터미널을 통해 Zenoh 라우터 → 시뮬레이션 환경 → 폴리시 노드 순으로 실행한다. 폴리시 노드 자체는 Docker 없이 `pixi run`으로 로컬 실행 가능하다.

---

## 1. 전체 아키텍처

| 컴포넌트 | 담당 | 실행 방식 |
|---------|------|----------|
| **Evaluation Component** | 시뮬레이션, 로봇, 센서, 채점 | Docker(`aic_eval`) 또는 소스 빌드 |
| **Participant Model** | 폴리시 노드 (직접 구현) | `pixi run` 또는 커스텀 Docker |

> **공식 평가**: 참가자 폴리시만 Docker 이미지로 제출. Evaluation Component 수정은 점수에 반영 안 됨.

---

## 2. 소스 빌드 방법 (`build_eval.md`)

### 사전 조건

| 항목 | 버전 |
|------|------|
| OS | Ubuntu 24.04 (Noble Numbat) |
| ROS 2 | Kilted Kaiju |

> **주의:** 기존에 ROS 2 Kilted 바이너리가 설치되어 있다면 **반드시 먼저 제거**해야 함.

```bash
# 기존 바이너리 제거 (충돌 방지)
sudo apt purge ros-kilted-ros2-control* ros-kilted-control* ros-kilted-kinematics* \
  ros-kilted-joint-state-publisher ros-kilted-realtime-tools ros-kilted-gz*
```

---

### Step 1 – Gazebo 리포지토리 추가

```bash
sudo curl https://packages.osrfoundation.org/gazebo.gpg \
  --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) \
  signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
  http://packages.osrfoundation.org/gazebo/ubuntu-stable \
  $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

sudo apt-get update
```

---

### Step 2 – 워크스페이스 클론 및 빌드

```bash
# 워크스페이스 생성
sudo apt update && sudo apt upgrade -y
mkdir -p ~/ws_aic/src
cd ~/ws_aic/src

# 리포지토리 클론
git clone https://github.com/intrinsic-dev/aic

# 의존성 임포트
vcs import . < aic/aic.repos --recursive

# Gazebo 의존성 설치
sudo apt -y install $(sort -u $(find . -iname 'packages-'`lsb_release -cs`'.apt' \
  -o -iname 'packages.apt' | grep -v '/\.git/') | sed '/gz\|sdf/d' | tr '\n' ' ')

# ROS 2 의존성 설치
cd ~/ws_aic
sudo rosdep init  # 최초 1회만
rosdep install --from-paths src --ignore-src --rosdistro kilted -yr \
  --skip-keys "gz-cmake3 DART libogre-dev libogre-next-2.3-dev rosetta"

# rmw_zenoh_cpp 및 추가 의존성
sudo apt install -y ros-kilted-rmw-zenoh-cpp python3-pynput

# 빌드
source /opt/ros/kilted/setup.bash
GZ_BUILD_FROM_SOURCE=1 colcon build \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --merge-install \
  --packages-ignore lerobot_robot_aic
```

---

### Step 3 – 환경 변수 설정

```bash
# ~/.bashrc에 추가
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true;\
transport/shared_memory/transport_optimization/pool_size=536870912'

source ~/.bashrc
```

> **중요:** **모든 터미널**에서 `RMW_IMPLEMENTATION=rmw_zenoh_cpp` 설정 필수.

---

## 3. 시스템 실행 순서 (3개 터미널)

각 터미널에서 먼저 워크스페이스 소싱:
```bash
source ~/ws_aic/install/setup.bash
```

| 터미널 | 명령어 | 역할 |
|--------|--------|------|
| **Terminal 1** | `ros2 run rmw_zenoh_cpp rmw_zenohd` | Zenoh 라우터 시작 |
| **Terminal 2** | `ros2 launch aic_bringup aic_gz_bringup.launch.py ground_truth:=false start_aic_engine:=true` | Gazebo + aic_engine 실행 |
| **Terminal 3** | `ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=<your_policy>` | 폴리시 노드 실행 |

> Terminal 2 실행 후 `aic_engine`이 `aic_model` 노드를 **30초** 내에 찾지 못하면 타임아웃.  
> **Terminal 1 → 2 → 3 순서** 반드시 준수.

---

## 4. 폴리시 노드 로컬 실행 (pixi 사용)

Docker 없이 로컬에서 폴리시를 실행할 때는 `pixi run` 사용:

```bash
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm
```

> `pixi run`은 자체 환경을 생성하므로 **Docker/distrobox 외부에서 실행 가능**.

---

## 5. 커스텀 Dockerfile (폴리시 컨테이너화)

`aic_model`을 사용하지 않는 경우 커스텀 Dockerfile 필요.

### 필수 요구사항

| 항목 | 내용 |
|------|------|
| **ROS 2 미들웨어** | `rmw_zenoh_cpp` 필수 (`RMW_IMPLEMENTATION=rmw_zenoh_cpp`) |
| **Zenoh 연결** | `AIC_MODEL_ROUTER_ADDR` 환경변수로 주어진 라우터에 연결 |
| **인증** | user=`model`, password=`AIC_MODEL_PASSWD` 로 user-password 인증 |
| **Entrypoint** | 폴리시 노드를 시작하는 명령 (추가 인자 없음) |

### Zenoh 연결 설정 예시

```bash
ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$AIC_MODEL_ROUTER_ADDR"'"];\
transport/auth/usrpwd/user="model";\
transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'";\
transport/auth/usrpwd/dictionary_file="/credentials.txt"'

# credentials 파일 생성
echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
```

### 평가 시 자동 주입되는 환경 변수

| 변수 | 설명 |
|------|------|
| `RMW_IMPLEMENTATION` | 항상 `rmw_zenoh_cpp` |
| `ZENOH_ROUTER_CHECK_ATTEMPTS` | 항상 `-1` (라우터보다 먼저 시작해도 에러 방지) |
| `AIC_MODEL_ROUTER_ADDR` | 연결할 Zenoh 라우터 주소 |
| `AIC_MODEL_PASSWD` | user-password 인증에 사용할 패스워드 |

---

## 6. 의존성 관리 (pixi)

| 유형 | 명령어 |
|------|--------|
| ROS 패키지 | `pixi add ros-kilted-<패키지명>` |
| PyPI 패키지 | `pixi add --pypi <패키지명>` |
| 패키지 업데이트 반영 | **`pixi reinstall <패키지명>`** (변경사항 자동 추적 안 됨) |

> **핵심:** pixi 환경 내 변경사항은 `pixi reinstall` 없이는 적용되지 않음.

---

## 7. 트러블슈팅 주요 사항

| 문제 | 해결책 |
|------|--------|
| Gazebo RTF 낮음 (듀얼 GPU) | `sudo prime-select nvidia` → 로그아웃/로그인 |
| GPU 없을 때 RTF 낮음 | `aic.sdf`에서 GlobalIllumination `<enabled>false</enabled>` 설정 |
| RTX 50xx PyTorch 비호환 | `pixi.toml`에 `torch = ">=2.7.1"`, `torchvision = ">=0.22.1"` 오버라이드 |
| `distrobox enter -r aic_eval` 오류 | `export DBX_CONTAINER_MANAGER=docker` 먼저 실행 |
| Zenoh Shared Memory 경고 | 무시해도 됨 (기능 정상 동작) |

---

## 8. 참고 문서 링크

| 문서 | 경로 |
|------|------|
| 소스 빌드 가이드 | `docs/build_eval.md` |
| 커스텀 Dockerfile | `docs/custom_dockerfile.md` |
| Getting Started | `docs/getting_started.md` |
| 제출 가이드 | `docs/submission.md` |
| 트러블슈팅 | `docs/troubleshooting.md` |
| 폴리시 통합 | `docs/policy.md` |
