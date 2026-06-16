# AIC Sejong

[한국어](readme/README.ko.md) | [English](readme/README.en.md)

Intrinsic 및 Open Robotics가 주관한 AI for Industry Challenge의 솔루션 코드입니다 (70th/166 Teams) <br>

## 대회 설명
AI for Industry Challenge는 Universal Robots(UR5e) 로봇 팔이 케이블을 지정된 포트에 삽입하는 Peg-In-Hole Task입니다.

<details>
<summary><strong>Task Board Randomization</strong></summary>

매 Trial마다 Task Board의 XY/yaw, 카드의 yaw 및 위치, 삽입 포트 종류가 달라집니다

| 파라미터 | Trial 1/2 (NIC/SFP) | Trial 3 (SC) |
|----------|---------------------|--------------|
| `task_board_yaw` | [0.0, 3.1415] rad | [0.0, 3.1415] rad |
| `task_board_x` | [0.13, 0.17] m | [0.15, 0.19] m |
| `task_board_y` | [-0.25, -0.15] m | [-0.05, 0.05] m |

<br>

| 랜덤화 요소 | 정확한 범위 및 구성 |
|-------------|----------------------|
| NIC/SFP target | `nic_card_mount_N_present`: rail 0~4 중 활성화<br>`nic_card_mount_N_translation`: [-0.0215, 0.0234] m<br>`nic_card_mount_N_yaw`: [-0.1745, 0.1745] rad ([-10°, +10°]) |
| SC target | `sc_port_N_present`: rail 0~1 중 활성화<br>`sc_port_N_translation`: [-0.06, 0.055] m<br>`sc_port_N_yaw`: 0.0 |
| Cable/gripper perturbation | `cable_type`: `sfp_sc_cable` 또는 `sfp_sc_cable_reversed`<br>`gripper_offset_noise`: [-0.002, 0.002] m<br>NIC base offset: [0.0, 0.015385, 0.04245] m<br>SC base offset: [0.0, 0.015385, 0.04045] m |

</details>

<details>
<summary><strong>케이블 삽입 태스크 및 정책 구성</strong></summary>

참가자는 카메라 관측, 로봇 상태, 힘/토크(Force/Torque) 센서 정보를 활용하여 포트 위치와 자세를 추정하고, 케이블 삽입을 수행하는 정책을 개발해야 합니다.

본 솔루션은 YOLO 기반 포트 검출, 멀티뷰 위치 추정, pose/yaw 보정, 힘 센서 기반 재시도 로직을 하나의 최종 정책으로 통합했습니다.

</details>

## Key Contributions

```
1.
2.
3.
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
<br>

### 3. 최종 Policy 실행

#### 1. 시뮬레이터 실행
시뮬레이터는 항상 먼저 실행해야 합니다
```bash
export DBX_CONTAINER_MANAGER=docker

distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true
```

#### 2. 최종 정책 실행
```bash
cd ~/AIC_Sejong/ws_aic/src
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=final_policy.FinalPolicy
```
<br>

### 4. 데이터 수집 Policy 실행

#### 4-1. 시뮬레이터 실행
시뮬레이터는 항상 먼저 실행해야 합니다
```bash
export DBX_CONTAINER_MANAGER=docker
export AIC_POSE_HF_PATH=~/AIC_Sejong/model/
export AIC_SFP_YOLO_HF_PATH=~/AIC_Sejong/model/approach/SFP
export AIC_SC_YOLO_HF_PATH=~/AIC_Sejong/model/approach/SC

distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=true start_aic_engine:=true
```

#### 4-2.1. 기본 LeRobot 데이터셋 수집

```bash
export AIC_LEROBOT_OUT_DIR=~/AIC_Sejong/data/lerobot
export AIC_LEROBOT_REPO_ID=aic-sejong-team/aic-dataset
export AIC_LEROBOT_VERSION=v1.0
export AIC_LEROBOT_PUSH_TO_HUB=false

cd ~/AIC_Sejong/ws_aic/src
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=data_gen_node.LeRobot
```

#### 4-2.2. Entrance frame 기준 YOLO/Vision-Offset 데이터 수집
```bash
export AIC_LEROBOT_OUT_DIR=~/AIC_Sejong/data/lerobot
export AIC_LEROBOT_REPO_ID=aic-sejong-team/aic-dataset
export AIC_LEROBOT_VERSION=v1.0
export AIC_LEROBOT_PUSH_TO_HUB=false

cd ~/AIC_Sejong/ws_aic/src
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=data_gen_node.PortOffsetCollect
```

 `AIC_LEROBOT_OUT_DIR`과 `AIC_LEROBOT_REPO_ID`를 entrance dataset 경로로 바꾼 뒤 policy를 `data_gen_node.DataCollect2`로 지정합니다.

| Policy | 용도 | 저장 위치 | Output format | 주요 필드 |
|--------|------|-----------|---------------|-----------|
| `data_gen_node.LeRobot` | 기본 LeRobot 데이터 세트 수집 | `$AIC_LEROBOT_OUT_DIR/$AIC_LEROBOT_VERSION`, `/tmp/aic_episodes/<episode>/episode_summary.json` | LeRobot dataset (`meta/*.json`, `data/*.parquet`, `videos/*/*.mp4`) + episode summary JSON | `observation.state` float32[35], `action` float32[7], `observation.plug_to_port` float32[7], `observation.images.{left,center,right}_camera` video 256x288x3, `observation.scenario_params` float32[11], `observation.stiffness` float32[6], `observation.damping` float32[6], `phase`, `insertion_success` |
| `data_gen_node.DataCollect2` | entrance frame 기준 YOLO/vision-offset 데이터 수집 | `$AIC_LEROBOT_OUT_DIR/$AIC_LEROBOT_VERSION`, `$AIC_LEROBOT_OUT_DIR/$AIC_LEROBOT_VERSION/vision_offset_dataset`, `$AIC_LEROBOT_OUT_DIR/$AIC_LEROBOT_VERSION/debug`, `/tmp/aic_episodes/<episode>/episode_summary.json` | LeRobot dataset + `vision_offset_dataset/samples.jsonl` + `vision_offset_dataset/images/{left,center,right}/.../*.png` + YOLO debug image/video | LeRobot 공통 필드, JSONL `sample_id`, `task_id`, `task_type`, `phase`, `images`, `label.plug_tip_to_port`, `label.ports`, `label.insertion_wrist`, `command`, `collect`, `triangulation`, `yolo.cameras` |
| `data_gen_node.PortOffsetCollect` | 포트 상대 offset/RPY 및 YOLO keypoint 샘플 수집 | `$AIC_LEROBOT_OUT_DIR/$AIC_LEROBOT_VERSION/vision_offset_dataset` 하위 RPY dataset 디렉터리 | YOLO-style image/label dataset + metadata JSON/JSONL | `images/<split>/*.jpg`, `yolo/images/<split>/*.jpg`, `yolo/labels/<split>/*.txt`, `metadata/<split>/*.json`, `metadata.jsonl`; JSON에는 `label.plug_tip_to_port`, `label.ports`, `actual.plug_reference_to_port`, `collect.local_{x,y,z,roll,pitch,yaw}`, `triangulation`, `visibility`, `yolo.keypoints` 포함 |

여러 세트를 반복 수집하고 Gazebo 실행까지 자동화하려면 `ais_auto_capture/collect_data.py`를 사용합니다.

```bash
cd ~/AIC_Sejong/ws_aic/src
pixi run python ais/ais_auto_capture/collect_data.py \
  --sets 10 \
  --data-policy DataCollect2 \
  --lerobot-out-dir ~/AIC_Sejong/data/aic-entrance-dataset \
  --lerobot-repo-id aic-sejong-team/aic-entrance-dataset \
  --lerobot-version v1.0 \
  --no-push-to-hub
```

## 저장소 구조

| 경로 | 역할 |
|------|------|
| `data/` | 대회 train/dev/test 메타데이터와 submission 파일 |
| `model/` | 정책 실행에 필요한 모델 체크포인트 기본 위치 |
| `ws_aic/src/pixi.toml` | Pixi 환경 및 로컬 editable 패키지 정의 |
| `ws_aic/src/aic/` | AIC 공식 ROS 2 평가 환경, 인터페이스, 예제 정책 |
| `ws_aic/src/ais/ais_policy/final_policy/` | 최종 정책 `final_policy.FinalPolicy` |
| `ws_aic/src/ais/ais_policy/data_gen_node/` | 데이터 수집용 Policy (`LeRobot`, `DataCollect2`, `PortOffsetCollect`) |
| `ws_aic/src/ais/ais_policy/motion_planning_node/` | YOLO 기반 포트 검출 및 접근 모듈 |
| `ws_aic/src/ais/ais_policy/distance_prediction/` | distance/offset 예측 기반 정렬 모듈 |
| `ws_aic/src/ais/ais_pose_prediction/` | 통합 pose/yaw 예측 모델 코드 |
| `ws_aic/src/ais/ais_auto_capture/` | Gazebo 기반 자동 데이터 수집 |
| `ws_aic/src/ais/ais_yolo_train/` | YOLO 학습 데이터 수집 및 평가 |
| `ws_aic/src/ais/ais_retry_classifier/` | 삽입 실패 감지 및 재시도 판단 실험 |
| `ws_aic/src/ais/ais_model_evaluation/` | 정책 평가 실행 및 결과 정리 유틸리티 |
| `ws_aic/src/docs/` | 실험 문서와 세션별 요약 |
| `readme/` | 한국어/영문 README 문서 |

## TODO
