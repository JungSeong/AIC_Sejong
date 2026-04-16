# MuJoCo & Isaac Sim 맵 저장·변환·실행 가이드

> Gazebo에서 생성한 시뮬레이션 월드를 MuJoCo(MJCF) 및 Isaac Lab(USD) 포맷으로
> 변환하고, 각 플랫폼에서 정책을 실행하는 전체 과정을 정리.
> 소스: `aic_utils/aic_mujoco/README.md`, `aic_utils/aic_isaac/README.md`

---

## 1. 왜 다른 시뮬레이터를 쓰는가?

| 시뮬레이터 | 주요 장점 | 협력사 |
|-----------|----------|--------|
| **Gazebo (기본)** | 평가 환경과 동일, ROS2 완전 통합 | Intrinsic / Open Robotics |
| **MuJoCo** | 경량·고속 물리, 강화학습·연구 친화적 | Google DeepMind |
| **Isaac Lab** | GPU 병렬 시뮬, 강화학습 프레임워크 완비 | NVIDIA |

> **핵심:** 세 환경 모두 **동일한 ROS2 토픽 인터페이스**를 사용하므로
> 정책 코드 수정 없이 시뮬레이터를 교체할 수 있다.

---

## 2. 공통 출발점 — Gazebo에서 맵 추출

모든 변환의 시작점은 Gazebo가 자동 저장하는 **`/tmp/aic.sdf`** 파일이다.

```bash
# Gazebo 실행 (원하는 도메인 랜덤화 파라미터 지정)
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  spawn_task_board:=true spawn_cable:=true \
  cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \
  ground_truth:=true
# → 실행 즉시 /tmp/aic.sdf 에 현재 월드 저장됨
```

시나리오를 재사용하고 싶다면 백업해 두자:

```bash
cp /tmp/aic.sdf ~/aic_sejong/aic_data/scenarios/sfp_rail0_center.sdf
```

---

## 3. MuJoCo 변환 & 실행 (Part 1 — 뷰어만)

### 3-1. 사전 준비

```bash
# MuJoCo 관련 ROS2 저장소 가져오기 (최초 1회)
cd ~/ws_aic/src
vcs import < aic/aic_utils/aic_mujoco/mujoco.repos
# 추가되는 패키지: gz-mujoco (sdformat_mjcf), mujoco_vendor, mujoco_ros2_control

# sdformat → MJCF 변환에 필요한 Python 바인딩 설치
sudo apt install -y python3-sdformat16 python3-gz-math9

# 변환 도구 빌드
cd ~/ws_aic
source /opt/ros/kilted/setup.bash
colcon build --packages-select sdformat_mjcf
source install/setup.bash
```

### 3-2. SDF 파일 수정 (2가지 URI 버그 수정)

Gazebo 내보내기 과정에서 발생하는 XML 파싱 오류를 수동으로 수정해야 한다.

```bash
# 버그 1: <urdf-string> 태그 오염 제거
sed -i 's|file://<urdf-string>/model://|model://|g' /tmp/aic.sdf

# 버그 2: 상대 경로 mesh URI 복원
sed -i 's|file:///lc_plug_visual.glb|model://LC Plug/lc_plug_visual.glb|g' /tmp/aic.sdf
sed -i 's|file:///sc_plug_visual.glb|model://SC Plug/sc_plug_visual.glb|g' /tmp/aic.sdf
sed -i 's|file:///sfp_module_visual.glb|model://SFP Module/sfp_module_visual.glb|g' /tmp/aic.sdf
```

> **주의:** Gazebo를 재시작하고 새로 내보낼 때마다 위 수정을 반복해야 한다.

### 3-3. SDF → MJCF 변환

```bash
source ~/ws_aic/install/setup.bash
mkdir -p ~/aic_mujoco_world
sdf2mjcf /tmp/aic.sdf ~/aic_mujoco_world/aic_world.xml
# → ~/aic_mujoco_world/ 에 XML + mesh 에셋(.obj, .png) 생성
```

### 3-4. 에셋 복사

```bash
cp ~/aic_mujoco_world/* ~/ws_aic/src/aic/aic_utils/aic_mujoco/mjcf/
```

### 3-5. MJCF 정제 (add_cable_plugin.py)

변환된 단일 XML을 로봇/월드/씬 파일로 분리하고, 누락된 구성요소를 추가:

```bash
# ROS2 workspace를 source하지 않은 새 터미널에서 실행
cd ~/ws_aic/src/aic/aic_utils/aic_mujoco/
python3 scripts/add_cable_plugin.py \
  --input  mjcf/aic_world.xml \
  --output mjcf/aic_world.xml \
  --robot_output mjcf/aic_robot.xml \
  --scene_output mjcf/scene.xml

cd ~/ws_aic && colcon build --packages-select aic_mujoco
```

**스크립트가 자동으로 처리하는 항목들:**

| 항목 | 내용 |
|------|------|
| 파일 분리 | `scene.xml` (최상위) + `aic_robot.xml` + `aic_world.xml` |
| 액추에이터 추가 | UR5e 6관절 + Robotiq 그리퍼 position 제어 |
| 그리퍼 mimic 조인트 | 왼손가락↔오른손가락 equality 제약 |
| FT 센서 | `AtiForceTorqueSensor` 사이트에 force/torque 센서 부착 |
| `gripper_tcp` 사이트 | 정책에서 사용할 TCP 기준점 삽입 |
| 쿼터니언 정규화 | 로봇 링크의 잡음 있는 쿼터니언 보정 |
| 카메라 설정 | fov, 해상도, orientation (center/left/right) |
| 케이블 플러그인 | `mujoco.elasticity.cable` 강성·감쇠 설정 |
| 충돌 제외 | 테이블↔shoulder, 그리퍼 손가락, sc_port↔sc_plug 등 |

### 3-6. MuJoCo 뷰어로 확인

```bash
pixi shell

# 방법 A: 드래그 앤 드롭용 빈 뷰어
python -m mujoco.viewer

# 방법 B: 스크립트로 직접 열기
python src/aic/aic_utils/aic_mujoco/scripts/view_scene.py ~/aic_mujoco_world/scene.xml
```

> Space: 시뮬레이션 시작/정지 | Backspace: 리셋

---

## 4. MuJoCo + ROS2 Control (Part 2 — 정책 실행)

### 4-1. 빌드

```bash
cd ~/ws_aic
rosdep install --from-paths src --ignore-src --rosdistro kilted -yr \
  --skip-keys "gz-cmake3 DART libogre-dev libogre-next-2.3-dev"

GZ_BUILD_FROM_SOURCE=1 colcon build \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --merge-install --symlink-install \
  --packages-ignore lerobot_robot_aic
```

### 4-2. 실행

```bash
# 터미널 1: Zenoh 라우터
source ~/ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true'
ros2 run rmw_zenoh_cpp rmw_zenohd

# 터미널 2: MuJoCo + ros2_control 실행
source ~/ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true'
ros2 launch aic_mujoco aic_mujoco_bringup.launch.py
```

### 4-3. 정책 실행 (Gazebo와 동일)

ros2_control이 동일한 컨트롤러 인터페이스를 제공하므로 정책 코드 변경 불필요:

```bash
# 텔레오퍼레이션
source ~/ws_aic/install/setup.bash
ros2 run aic_teleoperation cartesian_keyboard_teleop

# 학습된 정책 실행 (예시)
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true -p policy:=my_policy_node.Baseline
```

---

## 5. Isaac Lab 변환 & 실행

> **중요 차이점:** Isaac Lab은 Gazebo SDF를 직접 변환하는 파이프라인이
> **미구현** 상태 (Future Work). 대신 **NVIDIA가 준비한 USD 에셋**을 사용.

### 5-1. 설치 흐름

```bash
# 호스트에서 (컨테이너 외부)
cd ~
git clone git@github.com:isaac-sim/IsaacLab.git   # Isaac Lab v2.3.2 필수

cd ~/IsaacLab
git clone git@github.com:intrinsic-dev/aic.git     # AIC 저장소 내부에 클론
```

### 5-2. NVIDIA 준비 에셋 배치

[에셋 다운로드](https://developer.nvidia.com/downloads/Omniverse/learning/Events/Hackathons/Intrinsic_assets.zip) 후 압축 해제:

```bash
# 배치 경로
~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/Intrinsic_assets/
```

**USD 에셋 목록:**

| 파일 | 설명 |
|------|------|
| `aic_unified_robot_cable_sdf.usd` | 로봇 + 케이블 통합 씬 |
| `assets/NIC Card/*.usd` | NIC 카드 비주얼 |
| `assets/SC Plug/*.usd` | SC 플러그 비주얼 |
| `assets/SC Port/*.usd` | SC 포트 비주얼 |
| `assets/Task Board Base/*.usd` | 태스크 보드 기반 |
| `scene/aic.usd` | 전체 씬 구성 |

### 5-3. Docker 컨테이너 빌드 & 진입

```bash
cd ~/IsaacLab

# Isaac Lab base 이미지 빌드 (최초 1회, 시간 소요)
./docker/container.py build base

# 컨테이너 시작 후 진입
./docker/container.py start base
./docker/container.py enter base

# 컨테이너 내부에서 aic_task 설치
python -m pip install -e aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task
```

### 5-4. 텔레오퍼레이션 & 데이터 수집

```bash
# (컨테이너 내부에서 실행)

# 텔레오퍼레이션 (키보드)
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/teleop.py \
  --task AIC-Task-v0 --num_envs 1 --teleop_device keyboard --enable_cameras

# 시연 데이터 수집 (HDF5 포맷)
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/record_demos.py \
  --task AIC-Task-v0 --teleop_device keyboard --enable_cameras \
  --dataset_file ./datasets/dataset.hdf5 --num_demos 10

# 수집 데이터 재생 확인
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/replay_demos.py \
  --dataset_file ./datasets/dataset.hdf5
```

### 5-5. 강화학습 (rsl-rl)

```bash
# PPO 기반 강화학습 (컨테이너 내부)
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task AIC-Task-v0 --num_envs 1 --enable_cameras
```

---

## 6. 플랫폼 비교 요약

| 항목 | Gazebo | MuJoCo | Isaac Lab |
|------|--------|--------|-----------|
| **평가 환경과 동일** | ✅ (공식) | ✅ (동일 ROS 토픽) | ❌ (별도 인터페이스) |
| **SDF 변환 필요** | — | ✅ (`sdf2mjcf`) | ❌ (USD 에셋 사용) |
| **변환 버그 수정** | — | ✅ (2가지 sed 수정) | — |
| **ROS2 연동** | ✅ 완전 통합 | ✅ (ros2_control) | ❌ (독립 환경) |
| **강화학습** | 어려움 | 가능 | ✅ (rsl-rl 내장) |
| **병렬 시뮬레이션** | ❌ | 제한적 | ✅ GPU 병렬 |
| **데이터 포맷** | LeRobot | LeRobot | HDF5 |
| **추천 용도** | 최종 검증 / 제출 | 빠른 데이터 수집 | 강화학습 연구 |

---

## 7. 전체 워크플로우 한눈에 보기

```
[Gazebo 실행]
    │  /entrypoint.sh 파라미터로 도메인 랜덤화
    │  → /tmp/aic.sdf 자동 저장
    │
    ├─→ [MuJoCo 변환]
    │       1. sed로 URI 버그 수정
    │       2. sdf2mjcf 변환
    │       3. add_cable_plugin.py 정제
    │       4. pixi shell → python -m mujoco.viewer (뷰어 확인)
    │       5. ros2 launch aic_mujoco aic_mujoco_bringup.launch.py
    │       6. 정책 실행 (Gazebo와 동일 명령어)
    │
    └─→ [Isaac Lab 변환]
            1. NVIDIA USD 에셋 다운로드 & 배치
            2. ./docker/container.py build base
            3. ./docker/container.py enter base
            4. isaaclab -p teleop.py 텔레오퍼레이션
            5. isaaclab -p record_demos.py 데이터 수집
            6. isaaclab -p train.py 강화학습
```
