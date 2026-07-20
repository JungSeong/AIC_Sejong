# AIC Sejong

[한국어](readme/README.ko.md) | [English](readme/README.en.md)

Intrinsic 및 Open Robotics가 주관한 AI for Industry Challenge의 솔루션 코드입니다 <br>

## 대회 설명
AI for Industry Challenge는 Universal Robots(UR5e) 로봇 팔이 케이블을 지정된 포트에 삽입하는 Peg-In-Hole Task입니다.

<details>
<summary><strong>Task Board Overview</strong></summary>

<table>
  <tr>
    <th colspan="2">SFP</th>
  </tr>
  <tr>
    <td width="50%"><img src="readme/photo/SFP1.png" alt="SFP task board view 1" width="100%"></td>
    <td width="50%"><img src="readme/photo/SFP2.png" alt="SFP task board view 2" width="100%"></td>
  </tr>
  <tr>
    <th colspan="2">SC</th>
  </tr>
  <tr>
    <td width="50%"><img src="readme/photo/SC1.png" alt="SC task board view 1" width="100%"></td>
    <td width="50%"><img src="readme/photo/SC2.png" alt="SC task board view 2" width="100%"></td>
  </tr>
</table>

</details>

<details>
<summary><strong>Task Board Randomization</strong></summary>

매 Trial마다 Task Board의 XY/yaw, 카드의 위치, 삽입 포트 종류가 달라집니다

| 파라미터 | Trial 1/2 (NIC/SFP) | Trial 3 (SC) |
|----------|---------------------|--------------|
| `task_board_x` | [0.13, 0.17] m | [0.15, 0.19] m |
| `task_board_y` | [-0.25, -0.15] m | [-0.05, 0.05] m |
| `task_board_yaw` | [0.0, 3.1415] rad | [0.0, 3.1415] rad |

<br>

| 랜덤화 요소 | 정확한 범위 및 구성 |
|-------------|----------------------|
| SFP Port | `location`: rail 0~4 중 활성화<br>`translation`: [-0.0215, 0.0234] m<br>`yaw`: [-0.1745, 0.1745] rad ([-10°, +10°]) |
| SC Port | `location`: rail 0~1 중 활성화<br>`translation`: [-0.06, 0.055] m<br>`yaw`: 0.0 |
| Cable/gripper perturbation | `cable_type`: `sfp_sc_cable`, `sfp_sc_cable_reversed`<br>`gripper_offset_noise`: [-0.002, 0.002] m

</details>

<details>
<summary><strong>케이블 삽입 태스크 및 정책 구성</strong></summary>

참가자는 카메라 관측, 로봇 상태, 힘/토크(Force/Torque) 센서 정보를 활용하여 포트 위치와 자세를 추정하고, 케이블 삽입을 수행하는 정책을 개발해야 합니다.

본 솔루션은 YOLO 기반 포트 검출, 멀티뷰 위치 추정, pose/yaw 보정, 힘 센서 기반 재시도 로직을 하나의 최종 정책으로 통합했습니다.

</details>

## Key Contributions

```
1. Gazebo Simulator 기반 랜덤화 환경 데이터 수집용 ROS2 노드 개발
2. YOLO Keypoint Detection 및 Multiview Triangulation 기반 포트 3-DoF 추론 로직 구현, 5개 Case에서 MAE ~5mm 내외로 예측 정확도 달성
3. 케이블 정렬을 위한 MultiView Bilinear Cross-Attention Regressor 구현, Fine-Grained한 Case에서 Worst Case 대비 XYZ MAE 21.3% 및 RPY MAE 16.5% 향상
```

## 시작하기

### Requirements

| 항목 | 요구 사항 |
|------|-----------|
| OS | Ubuntu 24.04 |
| ROS 2 | Kilted Kaiju |
| Package manager | Pixi |
| Container | Docker, Distrobox |
| Simulator | Gazebo |
| Middleware | `rmw_zenoh_cpp` |
| Hardware | NVIDIA RTX 2070+ / 8 GB VRAM, RAM 32 GB+ |

정책 노드는 호스트의 Pixi 환경, Gazebo/RViz 및 scoring engine은 eval 컨테이너에서 실행

### 1. Pixi 환경 설정
```bash
git clone https://github.com/JungSeong/AIC_Sejong.git ~/AIC_Sejong
cd ~/AIC_Sejong/ws_aic/src
pixi install
```

### 2. Distrobox 설정
```bash
export DBX_CONTAINER_MANAGER=docker

docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
```

### 3. 데이터 수집 노드 실행

데이터 수집은 `aic_model`에 데이터 수집용 policy를 지정해서 실행합니다. Gazebo와 scoring engine은 eval 컨테이너에서 실행하고, policy 노드는 호스트 Pixi 환경에서 실행합니다.

#### 3-1. 시뮬레이터 실행

데이터 수집에서는 GT TF와 task board 랜덤화 파라미터가 필요하므로 `ground_truth:=true`로 실행합니다.

```bash
export DBX_CONTAINER_MANAGER=docker

distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true
```

#### 3-2. 기본 LeRobot 데이터셋 수집

`data_gen_node.LeRobot`은 로봇 상태, action, 카메라 영상, plug-to-port 상태, stiffness/damping, phase, insertion 결과를 LeRobot 포맷으로 저장합니다.

```bash
export AIC_LEROBOT_OUT_DIR=~/AIC_Sejong/data/lerobot
export AIC_LEROBOT_REPO_ID=aic-sejong-team/aic-dataset
export AIC_LEROBOT_VERSION=v1.0

cd ~/AIC_Sejong/ws_aic/src
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=data_gen_node.LeRobot
```

#### 3-3. Vision-Offset 정렬 데이터셋 수집

`data_gen_node.PortOffsetCollect`은 entrance frame 기준 포트 상대 offset/RPY, YOLO keypoint, triangulation, visibility 정보를 저장합니다. 현재 기본 출력 경로는 `AIC_VISION_OFFSET_DATASET_DIR`이며, 지정하지 않으면 노드 내부 기본 경로를 사용합니다.

```bash
export AIC_VISION_OFFSET_DATASET_DIR=~/AIC_Sejong/data/vision_offset_dataset
export AIC_VISION_OFFSET_RECORD=1
export AIC_VISION_OFFSET_PUSH_TO_HUB=false
export AIC_VISION_OFFSET_REPO_ID=aic-sejong-team/aic-vision-offset-dataset
export AIC_VISION_OFFSET_HF_REVISION=main
export AIC_VISION_OFFSET_UPLOAD_ON_PORT_TYPE=sc

cd ~/AIC_Sejong/ws_aic/src
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=data_gen_node.PortOffsetCollect
```

이때 Hugging Face 업로드를 사용할 경우 Pixi 환경에서 먼저 로그인해야 합니다.

```bash
cd ~/AIC_Sejong/ws_aic/src
pixi run hf auth login
pixi run hf auth whoami
```

`401 Repository Not Found`가 발생하면 repo가 없거나 현재 token이 해당 dataset repo에 대한 write 권한을 갖고 있지 않은 상태입니다. 이 경우 `AIC_VISION_OFFSET_REPO_ID`를 본인 계정/조직의 dataset repo로 바꾸거나, 권한이 있는 token으로 다시 로그인합니다.

| Policy | 용도 | 저장 위치 | Output format |
|--------|------|-----------|---------------|
| `data_gen_node.LeRobot` | 기본 에피소드 수집 | `$AIC_LEROBOT_OUT_DIR/$AIC_LEROBOT_VERSION`, `/tmp/aic_episodes/<episode>/episode_summary.json` | LeRobot dataset (`meta/*.json`, `data/*.parquet`, `videos/*/*.mp4`) |
| `data_gen_node.PortOffsetCollect` | vision-offset 정렬 학습 샘플 수집 | `$AIC_VISION_OFFSET_DATASET_DIR` | YOLO-style image/label dataset + metadata JSON/JSONL |

#### 3-4. 자동 수집 스크립트

여러 세트를 반복 수집하고 Gazebo 실행까지 자동화하려면 `ais_auto_capture` 스크립트를 사용합니다. LeRobot 에피소드 수집과 PortOffsetCollect 정렬 데이터 수집은 서로 다른 스크립트를 사용합니다.

```bash
cd ~/AIC_Sejong/ws_aic/src

# LeRobot episode 자동 수집
pixi run python ais/ais_auto_capture/collect_lerobot_data.py \
  --sets 10 \
  --data-policy LeRobot \
  --lerobot-out-dir ~/AIC_Sejong/data/lerobot \
  --lerobot-repo-id aic-sejong-team/aic-dataset \
  --lerobot-version v1.0 \
  --no-push-to-hub

# PortOffsetCollect 기반 vision-offset 정렬 데이터 자동 수집
pixi run python ais/ais_auto_capture/collect_portoffset_randomization.py \
  --trials 20 \
  --samples-per-trial 24 \
  --port-types sfp,sc \
  --port-order round_robin \
  --dataset-version v1.0 \
  --no-push-to-hub \
  --headless \
  --rootless-distrobox \
  --cleanup
```

aarch64 소스 빌드 환경에서 LeRobot episode를 수집할 때는 동일한 인자로 `collect_lerobot_data_aarch.py`를 사용합니다. YOLO 접근 데이터셋은 `collect_yolo_data_aarch.py`를 사용합니다.

#### PortOffsetCollect 자동 수집 파라미터

`collect_portoffset_randomization.py`는 Gazebo를 trial마다 rootless distrobox로 자동 실행하고, SFP/SC target과 simulator 조명을 함께 랜덤화합니다. 각 trial 시작 시 `Task Board`, `Port`, `Cable / Robot`, `Simulator / Lighting` 카테고리별 랜덤화 값이 색상/볼드 로그로 출력됩니다.

| 파라미터 종류 | 범위 | 역할 |
|---|---|---|
| Target port type | `--port-types sfp`, `sc`, `sfp,sc` 기본 `sfp,sc` | 수집할 포트 계열을 선택합니다. 기본은 SFP와 SC를 모두 수집합니다. |
| Target port order | `--port-order round_robin` 또는 `random`, 기본 `round_robin` | SFP/SC trial 배치 순서를 결정합니다. |
| Trial count | `--trials`, 기본 `20` | 생성 및 수집할 Gazebo trial 개수입니다. |
| Samples per trial | `--samples-per-trial`, 기본 `24` | 한 trial에서 저장할 vision-offset sample 수입니다. |
| SFP task board X | `0.13 ~ 0.17 m` | SFP/NIC trial에서 task board의 world X 위치를 랜덤화합니다. |
| SFP task board Y | `-0.25 ~ -0.20 m` | SFP/NIC trial에서 task board의 world Y 위치를 랜덤화합니다. |
| SFP task board yaw | `0.55 ~ 0.80 rad` | SFP/NIC trial에서 task board 회전각을 랜덤화합니다. |
| SC task board X | `0.15 ~ 0.19 m` | SC trial에서 task board의 world X 위치를 랜덤화합니다. |
| SC task board Y | `-0.05 ~ 0.05 m` | SC trial에서 task board의 world Y 위치를 랜덤화합니다. |
| SC task board yaw | `0.0 ~ 3.1415 rad` | SC trial에서 task board 회전각을 랜덤화합니다. |
| Task board Z / roll / pitch | `z=1.14 m`, `roll=0`, `pitch=0` | 보드 높이와 기울기는 고정해 xy/yaw 중심의 scene variation만 적용합니다. |
| SFP NIC rail | `0 ~ 4` 중 1개 | SFP target이 들어갈 NIC rail을 선택합니다. |
| SFP port index | `sfp_port_0` 또는 `sfp_port_1` | SFP target port를 선택합니다. |
| SFP NIC translation | `-0.0215 ~ 0.0234 m` | 선택된 NIC card mount의 rail 방향 위치를 랜덤화합니다. |
| SFP NIC yaw | `-10 ~ +10 deg` | 선택된 NIC card mount의 yaw를 랜덤화합니다. |
| SC rail | `0 ~ 1` 중 1개 | SC target이 들어갈 SC rail을 선택합니다. |
| SC translation | `-0.06 ~ 0.055 m` | 선택된 SC mount의 rail 방향 위치를 랜덤화합니다. |
| Gripper offset noise | 각 축 `-0.002 ~ +0.002 m` | cable gripper offset 기준값에 미세 오차를 더합니다. SFP 기준값은 `[0, 0.015385, 0.04245] m`, SC 기준값은 `[0, 0.015385, 0.04045] m`입니다. |
| Cable RPY noise | `--cable-rpy-noise-deg`, 기본 `±20 deg` | cable 초기 roll/pitch/yaw 기준값에 회전 오차를 더합니다. |
| Robot home joint noise | `--robot-joint-noise-deg`, 기본 `±4 deg` | robot home joint 초기값에 관절별 오차를 더합니다. |
| Collect XYZ range | X/Y 기본 `-50 ~ +50 mm`, Z 기본 `0 ~ 100 mm` | PortOffsetCollect가 포트 기준 위치 offset sample을 생성하는 범위입니다. `--dx-*`, `--dy-*`, `--dz-*`로 축별 override할 수 있습니다. |
| Collect RPY range | roll/pitch 기본 `±25 deg`, yaw 기본 `±35 deg` | PortOffsetCollect가 포트 기준 자세 offset sample을 생성하는 범위입니다. `--roll-*`, `--pitch-*`, `--yaw-*`로 축별 override할 수 있습니다. |
| RPY norm cap | `--rpy-norm-max-rad`, 기본 미사용 | sampling된 RPY vector magnitude를 제한합니다. |
| Actual RPY norm filter | `--actual-rpy-norm-max-rad`, 기본 `--rpy-norm-max-rad` 정책값 사용 | 저장 직전 실제 plug-port quaternion angle이 큰 sample을 제외합니다. |
| Capture settle time | `--capture-settle-s`, 기본 `0.25 s` | offset 적용 후 이미지/metadata를 저장하기 전 안정화 대기 시간입니다. |
| Visibility filter | `--min-visible-cameras` 기본 `1`, `--visibility-margin-px` 기본 `8 px` | 포트가 충분히 보이는 sample만 저장하도록 카메라 visibility 기준을 정합니다. |
| Lighting randomization | 기본 on, `--no-randomize-lighting`으로 off | trial마다 Gazebo world SDF를 생성해 조명/배경을 랜덤화합니다. |
| Light intensity scale | `0.65 ~ 1.35` | `enclosure_light`, `ceiling_01`, `ceiling_02` intensity에 곱할 scale 범위입니다. |
| Light color jitter | 기본 `±0.12` | 각 light diffuse RGB를 channel별로 흔듭니다. |
| Light pose jitter | XY 기본 `±0.25 m`, Z 기본 `±0.20 m` | 각 light 위치를 약간 이동시켜 highlight/shadow 위치를 바꿉니다. |
| Ambient / background | ambient `0.0 ~ 0.08`, background `0.08 ~ 0.20` | scene ambient와 background 밝기를 랜덤화합니다. |
| Gazebo mode | `--headless` on/off | Gazebo GUI/RViz 실행 여부를 결정합니다. |
| Color log | 기본 on, `--no-color-log` 또는 `NO_COLOR`로 off | trial별 랜덤화 로그를 색상/볼드로 구분해 출력합니다. |
| Trial timeout | 기본 `time-limit-s + 180 s` | `episode_summary.json` 생성 대기 제한 시간입니다. |
| Dataset version | `--dataset-version`, 기본 빈 문자열 | `data/ais_portoffset_randomization/{version}` 하위에 저장할 버전을 지정합니다. |
| Hugging Face upload | 기본 off, `--push-to-hub`로 on, `--no-push-to-hub`로 off | PortOffsetCollect policy에 `AIC_VISION_OFFSET_PUSH_TO_HUB`를 명시 전달합니다. 실험 중 의도치 않은 업로드를 막기 위해 runner 기본값은 off입니다. |
| Hugging Face target | `--vision-offset-repo-id`, `--vision-offset-hf-revision`, `--vision-offset-hf-path-in-repo`, `--hf-private` | 업로드할 HF dataset repo, revision, repo 내부 경로, private repo 생성 여부를 지정합니다. |
| Upload port filter | `--upload-on-port-type` 값 `sfp`, `sc`, 또는 빈 문자열 기본 빈 문자열 | 특정 포트 타입 trial에서만 업로드하도록 제한합니다. 빈 문자열이면 PortOffsetCollect가 포트 타입 제한 없이 판단합니다. |


### 4. 최종 Policy 실행

#### 4-1. 시뮬레이터 실행

최종 정책 평가에서는 scoring engine을 켜고 policy 노드를 별도 터미널에서 실행합니다.

```bash
export DBX_CONTAINER_MANAGER=docker

distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=false \
  start_aic_engine:=true
```

#### 4-2. 모델 다운로드 및 경로 설정

FinalPolicy는 먼저 환경변수로 지정한 로컬 모델을 찾고, 없으면 기본 Hugging Face repo에서 `~/AIC_Sejong/model` 하위로 다운로드합니다. private repo를 사용하는 경우 먼저 로그인합니다.

```bash
cd ~/AIC_Sejong/ws_aic/src
pixi run hf auth login
pixi run hf auth whoami
```

로컬 모델을 직접 지정할 경우:

```bash
export AIC_SFP_YOLO_MODEL_PATH=~/AIC_Sejong/model/approach/SFP/weights/best.pt
export AIC_SC_YOLO_MODEL_PATH=~/AIC_Sejong/model/approach/SC/weights/best.pt
export AIC_SFP_VISION_OFFSET_MODEL_PATH=~/AIC_Sejong/model/align/SFP/cross_attention_bilinear/cross_attention_bilinear_best.pt
export AIC_SC_VISION_OFFSET_MODEL_PATH=~/AIC_Sejong/model/align/SC/cross_attention_bilinear/cross_attention_bilinear_best.pt
```

모델 다운로드/로드 로그 색상이 필요 없으면 `AIC_MODEL_LOG_COLOR=0`을 설정합니다.

#### 4-3. 최종 정책 실행

```bash
export AIC_DEBUG_SAVE_DIR=~/AIC_Sejong/debug
export AIC_SFP_YOLO_PORT_INDEX_FLIP=1

cd ~/AIC_Sejong/ws_aic/src
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=final_policy.FinalPolicy
```

FinalPolicy는 `lift_up_detect -> approach -> vision_offset_align -> insert` 순서로 동작합니다. detect 결과와 triangulation debug 이미지는 `AIC_DEBUG_SAVE_DIR` 아래에 저장됩니다.

## 디렉토리 구조

| 경로 | 역할 |
|------|------|
| `data/` | 대회 train/dev/test 메타데이터와 submission 파일 |
| `model/` | 정책 실행에 필요한 모델 체크포인트 기본 위치 |
| `ws_aic/src/pixi.toml` | Pixi 환경 및 로컬 editable 패키지 정의 |
| `ws_aic/src/aic/` | AIC 공식 ROS 2 평가 환경, 인터페이스, 예제 정책 |
| `ws_aic/src/ais/ais_policy/final_policy/` | 최종 정책 `final_policy.FinalPolicy` |
| `ws_aic/src/ais/ais_policy/data_gen_node/` | 데이터 수집용 Policy (`LeRobot`, `PortOffsetCollect`) |
| `ws_aic/src/ais/ais_policy/motion_planning_node/` | YOLO 기반 포트 검출 및 접근 모듈 |
| `ws_aic/src/ais/ais_policy/distance_prediction/` | distance/offset 예측 기반 정렬 모듈 |
| `ws_aic/src/ais/ais_pose_prediction/` | 통합 pose/yaw 예측 모델 코드 |
| `ws_aic/src/ais/ais_auto_capture/` | Gazebo 기반 자동 데이터 수집 |
| `ws_aic/src/ais/ais_yolo_train/` | YOLO 학습 데이터 수집 및 평가 |
| `ws_aic/src/ais/ais_retry_classifier/` | 삽입 실패 감지 및 재시도 판단 실험 |
| `ws_aic/src/ais/ais_model_evaluation/` | 정책 평가 실행 및 결과 정리 유틸리티 |
| `ws_aic/src/docs/` | 실험 문서와 세션별 요약 |
| `readme/` | 한국어/영문 README 문서 |

## TODOs
