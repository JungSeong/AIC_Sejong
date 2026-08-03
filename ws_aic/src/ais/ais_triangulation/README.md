# ais_triangulation

## 스크립트 설명

| 파일 | 환경 | 역할 |
|---|---|---|
| `run_triangulation_cases.py` | host Pixi → Distrobox | PortOffset와 동일한 범위에서 case YAML을 생성하고 `aic_eval` simulator를 실행 |
| `evaluate_triangulation_euclidean.py` | host Pixi + ROS 2 | prediction XYZ와 port entrance GT TF의 축별 오차 및 3D Euclidean 오차를 계산·저장 |

전체 워크플로우 : `run_triangulation_cases.py`로 Gazebo Simulator 환경을 생성 및 실행하고, `evaluate_triangulation_euclidean.py`로 FinalPolicy의 multi-camera triangulation 결과를 평가하는 구조

## 1. run_triangulation_cases.py - Test Cases 생성 및 simulator 실행

`run_triangulation_cases.py`는 PortOffsetCollector의 Task Board, module, cable randomization 범위에서 지정한 seed와 case 수만큼 case YAML을 생성한 뒤, 생성한 YAML을 `aic_engine_config_file`로 전달하여 `aic_eval` simulator를 실행한다. Robot은 camera common FOV 계산이 가능한 고정 observation pose를 사용한다.

```bash
cd /home/swlinux/Desktop/workspace/AIC_Sejong/ws_aic/src

pixi run python ais/ais_triangulation/run_triangulation_cases.py \
  --seed 20 \
  --num_cases 50 \
  --launch_rviz false
```


한편 위 명령은 아래의 두 작업을 연속으로 수행한다.

1. `ais_auto_capture/portoffset_randomization/scenario.py | make_trial_config()`로 candidate case를 생성한다.
2. 고정 robot home에서 target entrance가 최소 두 camera의 안전 margin 안에 투영되는 candidate만 채택한다.
3. `ais_triangulation/cases/YYYYMMDD_triangulation_cases.yaml`에 저장한 뒤, 그 절대 경로를 `aic_engine_config_file`로 넘겨 `aic_eval` Distrobox를 실행한다.

위 runner가 실제로 실행하는 simulator 명령은 아래와 같다.

```bash
export DBX_CONTAINER_MANAGER=docker

distrobox enter aic_eval -- /entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  gazebo_gui:=false \
  launch_rviz:=true \
  spawn_task_board:=false \
  spawn_cable:=false \
  aic_engine_config_file:=/home/swlinux/Desktop/workspace/AIC_Sejong/ws_aic/src/ais/ais_triangulation/cases/YYYYMMDD_triangulation_cases.yaml
```

시뮬레이션 YAML 파일만 생성하고 실제로 실행하지 않으려면 `--generate-only` 옵션을 추가한다. 같은 날짜에 다시 생성하면 같은 날짜 파일을 덮어쓴다. 같은 seed와 옵션은 같은 case 순서를 만든다.

### 1-1. 주요 파라미터

| 옵션 | 기본값 | 역할 |
|---|---:|---|
| `--seed` | `30` | Python random seed |
| `--num_cases`, `--num-cases` | `20` | 생성하고 실행할 case 수, 1 이상 |
| `--port-types` | `sfp,sc` | 생성할 port 종류 |
| `--port-order` | `round_robin` | `round_robin` 또는 `random` |
| `--time-limit-s` | `600` | 각 task 제한 시간 |
| `--robot-joint-noise-deg` | `0` | common-FOV 보장을 위해 `0`만 허용 |
| `--cable-rpy-noise-deg` | `20` | cable roll/pitch/yaw별 uniform noise |
| `--min-visible-cameras` | `2` | target entrance가 들어와야 하는 최소 camera 수, `2` 또는 `3` |
| `--visibility-margin-px` | `64` | image 경계에서 제외할 안전 margin |
| `--distrobox` | `aic_eval` | 실행할 Distrobox 이름 |
| `--headless` | 꺼짐 | 사용 시 `gazebo_gui:=false` |
| `--ground_truth {true,false}` | `true` | `ground_truth:=...`를 entrypoint에 전달 |
| `--start_aic_engine {true,false}` | `true` | `start_aic_engine:=...`를 entrypoint에 전달 |
| `--gazebo_gui {true, false}` | `false` | `gazebo_gui:=...`를 entrypoint에 전달 |
| `--launch_rviz {true,false}` | `false` | `launch_rviz:=...`를 entrypoint에 전달 |
| `--spawn_task_board {true,false}` | `false` | `spawn_task_board:=...`를 entrypoint에 전달 |
| `--spawn_cable {true,false}` | `false` | `spawn_cable:=...`를 entrypoint에 전달 |
| `--sim-arg NAME=VALUE` | 없음 | 그 밖의 entrypoint launch parameter 전달. 반복 가능 |
| `--output PATH` | 날짜 기반 경로 | 생성 YAML 경로를 직접 지정 |
| `--generate-only` | 꺼짐 | YAML 생성 후 Distrobox를 실행하지 않음 |

### 1-2. 고정 observation pose와 camera common FOV 보장

Runner는 Task Board 중심이 아니라 YAML의 board, rail/module, port entrance transform을 모두 합성해 실제 target entrance의 `base_link` XYZ를 계산한다. SFP와 SC의 계산 chain은 각각 다음과 같다.

```text
SFP: world → task_board → nic_card_mount → nic_card → sfp_port → entrance
SC : world → task_board → sc_port → sc_port_base → entrance
```

고정 `BASE_ROBOT_HOME`에서 URDF로 산출한 `base_link → {left,center,right}_camera/optical` transform과 simulator camera 설정인 `1152×1024`, horizontal FOV `0.8718 rad`, near clip `0.07 m`를 사용한다. 각 camera에서는 다음 식으로 entrance를 pixel에 투영한다.

```text
p_camera = inverse(T_base_camera) · p_base
u = fx · X / Z + cx
v = fy · Y / Z + cy
fx = fy = width / (2 · tan(horizontal_fov / 2))
```

기본 채택 조건은 다음과 같다.

```text
0.07 m ≤ Z ≤ 20 m
64 ≤ u < 1152 - 64
64 ≤ v < 1024 - 64
위 조건을 만족하는 camera 수 ≥ 2
```

조건을 통과하지 못한 candidate는 YAML에 넣지 않고 같은 port 순서로 다시 sampling한다. 한 case에서 10,000회 연속 실패하면 무한 loop 대신 오류로 종료한다. `--sim-arg robot_x=...`처럼 robot world pose를 바꾸면 같은 값이 visibility 계산에도 반영된다.

이 필터가 보장하는 것은 **target entrance 중심점의 기하학적 common FOV**다. Mesh 가림, 조명, lens distortion, YOLO confidence까지 보장하지는 않으므로 실제 detection 성공률은 별도 batch 결과로 확인해야 한다.

## 2. constants.py 및 scenario.py - Test Case randomization 범위 정의

`ais_auto_capture/portoffset_randomization/constants.py`는 Task Board, module, cable 및 robot home의 randomization 범위를 정의하고, `scenario.py`는 이 범위에서 각 test case의 scene 설정을 sampling한다. `run_triangulation_cases.py`가 두 코드를 직접 사용하므로 PortOffsetCollector의 범위 변경은 triangulation case 생성에도 그대로 반영된다.

| 대상 | Translation | Rotation |
|---|---|---|
| SFP Task Board | X `0.13 ~ 0.17 m`, Y `-0.25 ~ -0.20 m`, Z `1.14 m` | roll/pitch `0`, yaw `3.10 ~ 3.1415 rad` |
| SC Task Board | X `0.15 ~ 0.19 m`, Y `-0.05 ~ 0.05 m`, Z `1.14 m` | roll/pitch `0`, yaw `3.10 ~ 3.1415 rad` |
| SFP NIC module | rail translation `-0.0215 ~ 0.0234 m` | local yaw `-10 ~ +10 deg` |
| SC port module | rail translation `-0.06 ~ 0.055 m` | local roll/pitch/yaw `0` |
| Cable gripper offset | port별 기준 X/Y/Z에서 각 축 `±2 mm` | 해당 없음 |
| Cable pose | 해당 없음 | 기준 roll/pitch/yaw에서 각 축 기본 `±20 deg` |
| Robot home | 해당 없음 | `BASE_ROBOT_HOME` 고정, joint noise 없음 |

SFP는 NIC rail 5개와 port 2개, SC는 rail 2개 중 target을 선택한다. Task Board의 translation과 rotation도 매 case에서 새로 추출된다.

고정 Robot home pose는 모든 trial에 공유된다. Task Board, target module, cable pose는 각 trial마다 독립적으로 다시 추출되며 common-FOV 조건을 통과한 조합만 채택된다. PortOffset의 조명 randomization은 YAML scene 필드가 아니라 수집 runner가 world 파일을 trial별로 수정하는 기능이므로 이 runner에는 포함되지 않는다.

## 3. evaluate_triangulation_euclidean.py - Triangulation XYZ 오차 평가

`evaluate_triangulation_euclidean.py`는 ROS 2 evaluator node를 실행하는 Python script다. `--prediction-topic`으로 지정한 `/final_policy/triangulated_port_xyz` Topic을 구독하고, prediction timestamp에 해당하는 port entrance GT TF를 조회하여 XYZ 오차를 계산한다. 이때 case YAML의 `port_type`, `target_module_name`, `port_name`으로 평가 대상 entrance TF frame 이름을 구성한다. Prediction과 GT를 같은 timestamp와 base frame으로 맞춘 뒤 `dx/dy/dz` 및 3D Euclidean distance를 계산한다.

Prediction의 `header.frame_id`가 비어 있거나 `header.stamp`가 0이면 해당 sample은 평가하지 않는다. 정확한 timestamp의 TF를 얻지 못하면 `--sync-threshold-ms` 이내인 최신 TF만 대체값으로 허용하고, threshold를 넘으면 결과를 저장하지 않는다.

생성된 YAML 경로를 확인한 뒤 evaluator를 별도 터미널에서 시작한다. `--cases`를 생략하면 `cases/`에서 파일명 기준 가장 최신인 `*_triangulation_cases.yaml`을 선택한다. `--all-cases`에서는 YAML의 모든 cases를 평가한다.

```bash
cd /home/swlinux/Desktop/workspace/AIC_Sejong/ws_aic/src

pixi run python ais/ais_triangulation/evaluate_triangulation_euclidean.py \
  --all-cases \
  --prediction-topic /final_policy/triangulated_port_xyz \
  --base-frame base_link \
  --fixed-frame world \
  --overwrite
```

다른 터미널에서 FinalPolicy를 활성화한다.

```bash
cd /home/swlinux/Desktop/workspace/AIC_Sejong/ws_aic/src

export AIC_PUBLISH_TRIANGULATED_PORT_XYZ=1
export AIC_TRIANGULATION_EVAL_ONLY=1
export AIC_YOLO_DEVICE=cpu
export AIC_TRIANGULATED_PORT_XYZ_TOPIC=/final_policy/triangulated_port_xyz
export AIC_TRIANGULATED_PORT_XYZ_FRAME_ID=base_link
export AIC_DETECTION_DEBUG_RVIZ=1
export AIC_DETECTION_DEBUG_IMAGE_TOPIC_PREFIX=/final_policy/detection_debug
export AIC_TRIANGULATION_DEBUG_RVIZ=1
export AIC_TRIANGULATION_DEBUG_IMAGE_TOPIC_PREFIX=/final_policy/triangulation_debug
export AIC_TRIANGULATION_DEBUG_MARKER_TOPIC=/final_policy/triangulation_debug/markers
export AIC_DEBUG_IMAGE_REPUBLISH_HZ=1
export AIC_TRIANGULATION_SYNC_THRESHOLD_MS=30

pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=final_policy.FinalPolicy
```

평가 결과는 다음 위치에 저장된다.

- `ais/ais_triangulation/results/triangulation_xyz_results.csv`
- `ais/ais_triangulation/results/triangulation_xyz_results.jsonl`
- `ais/ais_triangulation/results/triangulation_xyz_summary.json`

단일 case를 평가할 때는 생성 YAML의 `trials` key 중 하나를 `--case-name`에 지정하고 `--once`를 사용한다.

```bash
pixi run python ais/ais_triangulation/evaluate_triangulation_euclidean.py \
  --cases ais/ais_triangulation/cases/YYYYMMDD_triangulation_cases.yaml \
  --case-name trial_0000_sfp \
  --prediction-topic /final_policy/triangulated_port_xyz \
  --fixed-frame world \
  --once
```

### 3-1. 주요 파라미터

| 옵션 | 기본값 | 역할 |
|---|---|---|
| `--cases PATH` | `cases/`의 최신 날짜 YAML | case와 target entrance frame을 읽을 AIC engine YAML |
| `--case-name NAME` | 없음 | 단일 ROS 평가 대상 trial 이름 |
| `--target-frame FRAME` | 없음 | YAML 대신 target TF frame을 직접 지정 |
| `--base-frame FRAME` | `base_link` | prediction을 변환하고 GT를 조회할 공통 좌표계 |
| `--prediction-topic TOPIC` | `/final_policy/triangulated_port_xyz` | `PointStamped` 또는 `PoseStamped` prediction 입력 topic |
| `--message-type` | `point` | topic message를 `point` 또는 `pose`로 해석 |
| `--pred-xyz X Y Z` | 없음 | topic 대신 단일 prediction 값을 직접 입력. `--all-cases`와 함께 사용 불가 |
| `--predictions PATH` | 없음 | CSV, JSONL 또는 JSON을 읽는 offline 평가 모드 |
| `--tf-timeout SEC` | `3.0` | GT TF 조회 제한 시간 |
| `--sync-threshold-ms MS` | `30.0` | 정확 시각 TF가 없을 때 허용할 인접 TF 최대 시간차 |
| `--fixed-frame FRAME` | `world` | direct TF 조회 실패 시 full lookup에 사용할 고정 frame |
| `--all-cases` | 꺼짐 | 모든 trial을 YAML 순서대로 평가 |
| `--once` | 꺼짐 | 단일 case에서 prediction 하나를 저장한 뒤 종료 |
| `--output-dir PATH` | `ais_triangulation/results` | CSV, JSONL, summary JSON 저장 디렉터리 |
| `--overwrite` | 꺼짐 | 기존 CSV를 읽어 누적하지 않고 새 결과로 시작 |

#### 3-1-1. `--base-frame`과 `--fixed-frame`의 차이

`--base-frame`은 prediction과 GT를 변환하여 오차를 계산·저장하는 최종 좌표계다. 기본값은 `base_link`다.

`--fixed-frame`은 `ais_triangulation/evaluate_triangulation_euclidean.py | lookup_synced_transform()`에서 direct TF 조회가 실패했을 때 TF2 full lookup이 경유하는 기준 frame이다. Direct 조회가 성공하면 사용되지 않으며, 결과 좌표계도 변경하지 않는다. 따라서 `--base-frame base_link --fixed-frame world`는 결과를 `base_link` 기준으로 비교하되 exact-time TF fallback에서 `world`를 고정 기준으로 사용한다는 의미다.

Prediction과 GT를 같은 timestamp에서 조회하므로 TF tree가 정상이라면 `--fixed-frame world`와 `--fixed-frame base_link`의 XYZ 결과는 일반적으로 같다. `world`가 안정적으로 연결된 simulator에서는 `world`를 권장하며, `world` TF가 없거나 조회할 수 없을 때만 `base_link`를 사용한다.

Offline 모드에서는 입력 record마다 `case_name` 또는 `target_frame`, prediction XYZ와 GT XYZ가 모두 필요하다. 지원 필드명은 `ais_triangulation/evaluate_triangulation_euclidean.py | prediction_xyz_from_record()`와 `ais_triangulation/evaluate_triangulation_euclidean.py | gt_xyz_from_record()`에 정의되어 있다.

```bash
pixi run python ais/ais_triangulation/evaluate_triangulation_euclidean.py \
  --cases ais/ais_triangulation/cases/YYYYMMDD_triangulation_cases.yaml \
  --predictions /path/to/predictions.jsonl \
  --overwrite
```

### 3-2. vision.py 및 debug.py - RViz detection·triangulation debug 실시간 발행

`vision.py`와 `debug.py`는 YOLO detection overlay와 triangulation 재투영 결과를 원본 camera timestamp와 optical frame을 유지한 `sensor_msgs/Image`로 발행한다. 최종 prediction, GT 및 두 점 사이의 오차선은 `base_link` 기준 `visualization_msgs/MarkerArray`로도 발행한다.

#### 3-2-1. Camera pair 선택과 재투영 RMS

`vision.py`는 검출 가능한 `left-center`, `center-right`, `left-right` 중 사용할 수 있는 모든 pair와 같은 `point_name`의 detection 조합을 각각 DLT로 triangulation한다.

각 3D 후보를 해당 point가 검출된 모든 camera에 다시 투영하고, 실제 YOLO UV와 재투영 UV 사이의 pixel distance를 계산한다.

```text
e_camera = sqrt((u_detect - u_projected)^2 + (v_detect - v_projected)^2)
reprojection_rms = sqrt(mean(e_camera^2))
```

Camera별 오차가 `30 px`를 넘는 후보는 폐기한다. 남은 후보는 전체 camera의 `reprojection_rms`가 작은 순서로 정렬하며, 값이 같을 때만 board center 거리와 detection confidence 기반 score를 보조 기준으로 사용한다. 서로 `10 mm` 이내인 중복 3D 후보는 이 정렬 뒤 제거하므로 여러 pair가 같은 포트를 계산했을 때 재투영 RMS가 가장 작은 pair의 결과가 남는다.

```text
모든 사용 가능한 camera pair
→ pair별 DLT 3D 후보
→ 검출된 모든 camera에 재투영
→ camera별 30 px gate
→ 재투영 RMS 최소 후보 선택
→ 10 mm 중복 제거
```

이 방식은 가장 정확한 two-view pair를 고르는 로직이다. 세 camera 관측을 하나의 최적화 식으로 다시 계산하는 three-view refinement는 수행하지 않는다.

FinalPolicy는 center camera timestamp를 기준 시각으로 사용하고, left/center/right 전체 timestamp span이 `AIC_TRIANGULATION_SYNC_THRESHOLD_MS` 이내인 Observation만 triangulation한다. 기본 threshold는 `30 ms`다. 범위를 넘은 Observation은 경고 로그를 남기고 버리며 다음 관측을 기다린다. OpenCV debug의 GT TF도 center camera timestamp에서 조회한다.

YOLO detection topic은 JPEG로 저장되는 detection debug와 같은 bbox, keypoint, target 표시, confidence 및 UV 좌표를 실시간으로 보여준다. Overlay에는 원본 camera timestamp와 세 camera의 `sync_span`도 표시된다.

| Camera | YOLO detection Image topic |
|---|---|
| Left | `/final_policy/detection_debug/left/image` |
| Center | `/final_policy/detection_debug/center/image` |
| Right | `/final_policy/detection_debug/right/image` |

| Camera | Triangulation reprojection Image topic |
|---|---|
| Left | `/final_policy/triangulation_debug/left/image` |
| Center | `/final_policy/triangulation_debug/center/image` |
| Right | `/final_policy/triangulation_debug/right/image` |

3D debug topic은 `/final_policy/triangulation_debug/markers`다. RViz에서 `Add` → `By topic`을 선택해 각 image topic은 `Image`, 3D topic은 `MarkerArray` display로 추가한다. RViz의 `Fixed Frame`은 `base_link` 또는 `world`로 설정한다.

Detection과 triangulation debug image는 마지막 frame을 camera별로 캐시하고, RViz 구독자가 있는 동안 기본 `1 Hz`로 계속 재발행한다. 따라서 계산이 끝난 뒤 RViz display를 추가해도 마지막 결과를 볼 수 있다. 재발행 메시지는 원본 camera `header.stamp`와 `frame_id`를 유지하므로 새로운 관측처럼 timestamp를 변경하지 않는다. 주기는 `AIC_DEBUG_IMAGE_REPUBLISH_HZ`로 조절하며 `0` 이하이면 재발행을 끈다.

Marker 색상은 다음과 같다.

- 노랑 sphere: 최종 triangulation prediction
- 자홍 sphere: 같은 timestamp에서 조회한 GT port entrance
- 흰색 line: prediction과 GT 사이의 3D 오차

각 영상 overlay에는 다음 시간 정보가 표시된다.

- `stamp`: 해당 camera 원본 이미지의 ROS timestamp
- `sync_span`: 같은 Observation에 포함된 left/center/right timestamp의 최대 차이(ms)

최종 `/final_policy/triangulated_port_xyz`의 `PointStamped.header.stamp`는 publish 시각이 아니라 triangulation에 사용된 center camera 관측 시각을 우선 사용한다. center timestamp가 유효하지 않으면 left, right 순으로 대체한다. evaluator의 결과 파일도 이 prediction header 시각을 `timestamp`로 기록한다. center 영상과 최종 결과의 header를 함께 보면 동일 관측 결과인지 확인할 수 있다.

결과 CSV/JSONL에는 동기화 감사용으로 `prediction_frame`, `prediction_transform_mode`, `prediction_tf_sync_delta_ms`, `gt_tf_timestamp`, `gt_sync_delta_ms`도 저장된다. `prediction_transform_mode`가 `identity`면 prediction이 이미 base frame이었고, `exact` 또는 `exact_via_fixed_frame`이면 prediction timestamp에서 TF 변환한 것이다. `nearby_latest`는 정확 시각 TF가 없어 threshold 이내의 인접 TF를 사용했음을 뜻한다.

```bash
ros2 topic echo /final_policy/triangulation_debug/center/image --field header
ros2 topic echo /final_policy/triangulated_port_xyz --field header
ros2 topic echo /final_policy/triangulation_debug/markers --once
```

RViz는 현재 들어오는 frame을 실시간 표시하는 도구이므로 과거 시점으로 이동하는 timeline은 제공하지 않는다. 시간축 재생이 필요하면 위 image topic과 XYZ topic을 rosbag2로 함께 기록한 뒤 RViz에서 `/clock`과 함께 재생한다.

```bash
ros2 bag record \
  /clock \
  /final_policy/detection_debug/left/image \
  /final_policy/detection_debug/center/image \
  /final_policy/detection_debug/right/image \
  /final_policy/triangulation_debug/left/image \
  /final_policy/triangulation_debug/center/image \
  /final_policy/triangulation_debug/right/image \
  /final_policy/triangulation_debug/markers \
  /final_policy/triangulated_port_xyz
```

## 4. FinalPolicy.py 및 debug.py - 결과 좌표와 OpenCV debug 좌표 공유

`FinalPolicy.py`와 `debug.py`는 동일한 최종 3D triangulation 값을 prediction topic으로 발행하고 OpenCV debug 이미지의 `PRED` 좌표로 재투영한다.

두 종류의 OpenCV 이미지가 있으므로 구분해야 한다.

- Detection debug: YOLO keypoint group의 2D 중심점이며 triangulation의 입력이다.
- Triangulation debug: 최종 3D triangulation 결과를 각 camera로 재투영한 `PRED` 점이다.

`ais_policy/final_policy/final_policy/FinalPolicy.py | _cache_detected_port()`는 최종 base-link 3D 값 `self._cached_port_base` 하나를 `ais_policy/final_policy/final_policy/FinalPolicy.py | _publish_triangulated_port_xyz()`와 `ais_policy/final_policy/final_policy/debug.py | _save_triangulation_debug_images()`에 동일하게 전달한다. evaluator의 `pred_x_m/pred_y_m/pred_z_m`은 이 publish 값을 저장하고, triangulation debug는 같은 값을 `ais_policy/final_policy/final_policy/geometry.py | project_3d_to_pixel()`로 표시한다. 따라서 결과 파일의 prediction과 triangulation debug의 `PRED` 계산 원점은 동일하다.

최종 결과와 좌표 cache 로그는 **볼드 시안**, RViz image publish 요약은 **볼드 초록**으로 표시된다.

Detection debug의 2D 중심점과 triangulation debug의 `PRED` 픽셀이 완전히 같을 필요는 없다. 여러 camera의 2D 관측으로 한 3D 점을 추정한 뒤 재투영하므로 calibration 오차, detection 오차, DLT 잔차만큼 차이가 날 수 있다. 성능 비교에는 raw detection debug가 아니라 triangulation debug의 `PRED`와 `GT`, 그리고 evaluator의 XYZ 오차를 사용한다.

## 5. test/test_triangulation_cases.py - Triangulation 회귀 검사

`test/test_triangulation_cases.py`는 case 수가 고정되지 않는지, seed 재현성, PortOffset 범위, Distrobox 인자, publish/debug의 동일 3D 값 사용을 확인한다.

```bash
cd /home/swlinux/Desktop/workspace/AIC_Sejong/ws_aic/src
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  pixi run pytest ais/ais_triangulation/test/test_triangulation_cases.py -q
```
