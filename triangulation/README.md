## Triangulation Case Map

`triangulation/triangulation_cases.yaml`은 YOLO keypoint + multiview triangulation 성능을 고정 scene에서 검증하기 위한 10개 case map입니다.

| 항목 | 내용 |
|------|------|
| 목적 | port entrance ground truth XYZ와 FinalPolicy의 multi-camera triangulation XYZ 비교 |
| Case 구성 | SFP 6개, SC 4개 |
| Map 기준 | `scene.task_board`, active rail, cable pose, `tasks.task_1` |
| Target frame | SFP: `task_board/nic_card_mount_N/sfp_port_{0,1}_link_entrance`, SC: `task_board/sc_port_N/sc_port_base_link_entrance` |

## Cases

| Case | Type | Target | Active rail | Scene note |
|------|------|--------|-------------|------------|
| `case_01_sfp_rail0_port0_left` | SFP | `nic_card_mount_0/sfp_port_0` | `nic_rail_0`, -21.5 mm, yaw -10.0° | left-side SFP |
| `case_02_sfp_rail1_port1_center` | SFP | `nic_card_mount_1/sfp_port_1` | `nic_rail_1`, 0.0 mm, yaw 0.0° | center SFP |
| `case_03_sfp_rail2_port0_far_yaw` | SFP | `nic_card_mount_2/sfp_port_0` | `nic_rail_2`, +23.4 mm, yaw +10.0° | far/yaw SFP |
| `case_04_sfp_rail3_port1_near` | SFP | `nic_card_mount_3/sfp_port_1` | `nic_rail_3`, -10.0 mm, yaw +5.0° | near SFP |
| `case_05_sfp_rail4_port0_offset` | SFP | `nic_card_mount_4/sfp_port_0` | `nic_rail_4`, +6.0 mm, yaw -2.5° | mild offset SFP |
| `case_06_sfp_rail2_port1_wide_yaw` | SFP | `nic_card_mount_2/sfp_port_1` | `nic_rail_2`, -18.0 mm, yaw +6.9° | wide/yaw SFP |
| `case_07_sc_rail0_left` | SC | `sc_port_0/sc_port_base` | `sc_rail_0`, -30.0 mm, yaw 0.0° | mild left SC |
| `case_08_sc_rail1_right` | SC | `sc_port_1/sc_port_base` | `sc_rail_1`, +55.0 mm, yaw 0.0° | right SC |
| `case_09_sc_rail0_center` | SC | `sc_port_0/sc_port_base` | `sc_rail_0`, 0.0 mm, yaw 0.0° | center SC |
| `case_10_sc_rail1_offset_yaw` | SC | `sc_port_1/sc_port_base` | `sc_rail_1`, -30.0 mm, yaw 0.0° | offset SC |

## Map 생성

`trials` 아래에 case를 정의하면 AIC engine이 해당 YAML을 읽어 task board와 cable scene을 생성합니다.

| YAML 필드 | 역할 |
|-----------|------|
| `scene.task_board.pose` | board의 world pose |
| `nic_rail_N`, `sc_rail_N` | 활성 port module과 rail translation/yaw |
| `cables.cable_0.pose` | plug/cable 초기 pose |
| `tasks.task_1` | plug type, port type, target module, time limit |

```bash
export DBX_CONTAINER_MANAGER=docker

distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  gazebo_gui:=true \
  launch_rviz:=false \
  spawn_task_board:=false \
  spawn_cable:=false \
  aic_engine_config_file:=/home/swlinux/Desktop/workspace/AIC_Sejong/triangulation/triangulation_cases.yaml
```

이후 `FinalPolicy`를 실행하면 YOLO keypoint를 각 camera view에서 추론하고, camera projection을 이용해 base_link 기준 port 3-DoF를 triangulation합니다.

```bash
cd ~/AIC_Sejong/ws_aic/src
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=final_policy.FinalPolicy
```

## GT/Prediction 비교

`evaluate_triangulation_xyz.py`는 case YAML에서 target frame을 찾고, `base_link` 기준 GT XYZ와 FinalPolicy의 추론 XYZ를 비교해 결과를 저장합니다.

아래 명령을 별도 터미널에서 먼저 실행한 뒤 FinalPolicy를 실행하면, detection stage에서 publish된 추론 XYZ를 받아 한 번 비교합니다.

```bash
python3 /home/swlinux/Desktop/workspace/AIC_Sejong/triangulation/evaluate_triangulation_xyz.py \
  --case-name case_01_sfp_rail0_port0_left \
  --prediction-topic /final_policy/triangulated_port_xyz \
  --once
```

저장 파일은 `triangulation/results/triangulation_xyz_results.csv`, `triangulation_xyz_results.jsonl`, `triangulation_xyz_summary.json`입니다.

| 입력 | 내용 |
|------|------|
| `--case-name` | `triangulation_cases.yaml`의 case 이름 |
| `--prediction-topic` | FinalPolicy가 publish하는 `geometry_msgs/PointStamped` 추론 XYZ |
| `--target-frame` | YAML 밖의 frame을 직접 비교할 때 사용 |
| `--predictions` | GT/추론 XYZ가 들어 있는 CSV/JSONL/JSON을 오프라인 비교 |
| `--overwrite` | 기존 결과를 덮어쓰고 새로 저장 |

10개 case는 engine case를 바꿔 실행한 뒤 같은 평가 명령을 반복하면 결과 파일에 누적됩니다.
