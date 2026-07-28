# AIC Sejong

[한국어](readme/README.ko.md) | [English](readme/README.en.md)

Intrinsic 및 Open Robotics가 주관한 AI for Industry Challenge의 솔루션 코드입니다 <br>

![Final Policy](readme/gif/FinalPolicy1.gif)

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

매 Trial마다 Task Board의 XY/yaw, 카드의 위치, 삽입 포트 종류가 달라집니다.

| 파라미터 | Trial 1/2 (NIC/SFP) | Trial 3 (SC) |
|---|---|---|
| `task_board_x` | [0.13, 0.17] m | [0.15, 0.19] m |
| `task_board_y` | [-0.25, -0.15] m | [-0.05, 0.05] m |
| `task_board_yaw` | [0.0, 3.1415] rad | [0.0, 3.1415] rad |

| 랜덤화 요소 | 범위 및 구성 |
|---|---|
| SFP Port | rail 0~4, translation [-0.0215, 0.0234] m, yaw [-10°, +10°] |
| SC Port | rail 0~1, translation [-0.06, 0.055] m, yaw 0.0 |
| Cable/gripper perturbation | cable 방향 및 gripper offset noise 랜덤화 |

</details>

<details>
<summary><strong>케이블 삽입 태스크 및 정책 구성</strong></summary>

참가자는 카메라 관측, 로봇 상태, 힘/토크(Force/Torque) 센서 정보를 활용하여 포트 위치와 자세를 추정하고, 케이블 삽입을 수행하는 정책을 개발해야 합니다.

본 솔루션은 YOLO 기반 포트 검출, 멀티뷰 위치 추정, pose/yaw 보정, 힘 센서 기반 재시도 로직을 하나의 최종 정책으로 통합했습니다.

</details>

## Key Contributions

```text
1. Gazebo Simulator 기반 랜덤화 환경 데이터 수집용 ROS2 노드 개발
2. YOLO Keypoint Detection 및 Multiview Triangulation 기반 포트 3-DoF 추론 로직 구현, 5개 Case에서 MAE ~5mm 내외로 예측 정확도 달성
3. 케이블 정렬을 위한 MultiView Bilinear Cross-Attention Regressor 구현, Fine-Grained한 Case에서 Worst Case 대비 XYZ MAE 21.3% 및 RPY MAE 16.5% 향상
```

## 시작하기

### Requirements

| 항목 | 요구 사항 |
|---|---|
| OS | Ubuntu 24.04 |
| ROS 2 | Kilted Kaiju |
| Package manager | Pixi |
| Container | Docker, Distrobox |
| Simulator | Gazebo |
| Middleware | `rmw_zenoh_cpp` |
| Hardware | NVIDIA RTX 2070+ / 8 GB VRAM, RAM 32 GB+ |

정책 노드는 호스트의 Pixi 환경, Gazebo/RViz 및 scoring engine은 eval 컨테이너에서 실행합니다.

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
distrobox create -r --nvidia \
  -i ghcr.io/intrinsic-dev/aic/aic_eval:latest \
  aic_eval
```

### 3. 데이터 자동 수집 노드

데이터 자동 수집은 Gazebo trial 실행, 시나리오 랜덤화, dataset 저장 및 선택적 rosbag 기록 및 Hugging Face 업로드를 한 번에 수행합니다.

```bash
cd ~/AIC_Sejong/ws_aic/src

pixi run python ais/ais_auto_capture/collect_portoffset_randomization_data.py \
  --trials 20 \
  --samples-per-trial 24 \
  --port-types sfp,sc \
  --dataset-version 0726-001 \
  --push-to-hub false \
  --record-rosbag true \
  --headless \
  --cleanup
```

시나리오 랜덤화 분포 plot은 다음 명령으로 생성합니다.

```bash
cd ~/AIC_Sejong/ws_aic/src
pixi run python ais/ais_auto_capture/plot_scenario_randomization.py
```

plot 기본 출력 위치는 [scenario_randomization_distributions.png](readme/photo/scenario_randomization_distributions.png)입니다.

전체 데이터 세트 수집 파라미터, 수집 시각 일치 검사, rosbag 및 offline sample 검증은 [ais_auto_capture 상세 문서](ws_aic/src/ais/ais_auto_capture/README.md)를 참고하십시오.

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
|---|---|
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