# AIC Sejong

[한국어](README.ko.md) | [English](README.en.md)

Solution code for the AI for Industry Challenge hosted by Intrinsic and Open Robotics (70th/166 Teams) <br>

[![Hugging Face Hub](https://img.shields.io/badge/Hugging%20Face-aic--sejong--team-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/aic-sejong-team)

## Competition Overview
The AI for Industry Challenge evaluates perception accuracy and insertion success rate for simulation-based policies that make a UR5e robot arm insert a cable into a target port.

<details>
<summary><strong>[1] Cable-Insertion Task and Policy Design</strong></summary>

Participants must estimate the port position and execute cable insertion policies using camera observations, robot state, and force/torque sensor information.

| Component | Role |
|-----------|------|
| UR5e robot arm | Execute cable-insertion motions |
| YOLO-pose | Detect port pose and key points |
| Multiview triangulation | Estimate the 3D port position from multiple camera views |
| Vision and F/T sensors | Detect insertion failures and decide retry behavior |
| Gazebo/AIC Simulator | Support repeated experiments, policy validation, and data collection |

This solution implements yaw/XYZ alignment logic to reduce port-position estimation error and builds a sensor-based retry flow for insertion-failure scenarios.

</details>

<details>
<summary><strong>[2] Data Collection and Collaboration Management</strong></summary>

To reduce bottlenecks in repeated experimental data collection, the project implements a Gazebo-based automatic data collection node and manages YOLO training data and LeRobot-format datasets.

| Item | Purpose |
|------|---------|
| Gazebo auto-collection node | Generate repeated experimental data and reduce collection time |
| YOLO training data | Train port detection and pose-estimation models |
| LeRobot dataset | Manage policy training and reproducible experiments |
| GitHub | Manage code, experiment artifacts, and collaboration issues |
| Hugging Face Hub | Share and load models and datasets |
| Notion | Document schedules, roles, and meeting notes |

Project artifacts are organized for use through the [Hugging Face Hub](https://huggingface.co/aic-sejong-team).

</details>

## Key Contributions

```
1. Implemented YOLO-pose based port-pose estimation and multiview triangulation based port-position estimation, with yaw/XYZ alignment logic to reduce position-estimation error (XX% performance improvement)
2. Implemented a Vision and F/T sensor based retry logic for insertion-failure scenarios (YY% performance improvement)
3. Built a Gazebo-based automatic data collection node to reduce repeated-experiment data collection bottlenecks
4. Managed distributed experiment artifacts and collaboration flow across GitHub, Hugging Face Hub, and Notion
5. Recruited teammates through the OROCA Naver Cafe and MODULABS, coordinated schedules and roles, and documented the project
```
<br>

## Models and Data

Project model and dataset artifacts are managed on the [Hugging Face Hub](https://huggingface.co/aic-sejong-team).

| Resource | Link |
|----------|------|
| Organization | [aic-sejong-team](https://huggingface.co/aic-sejong-team) |
| LeRobot Dataset | [aic-sejong-team/aic-dataset](https://huggingface.co/datasets/aic-sejong-team/aic-dataset) |
| Entrance Dataset | [aic-sejong-team/aic-entrance-dataset](https://huggingface.co/datasets/aic-sejong-team/aic-entrance-dataset) |
| ACT Policy | [aic-sejong-team/act_AIC](https://huggingface.co/aic-sejong-team/act_AIC) |

## Getting Started

### 1. Install Dependencies
```bash
git clone https://github.com/JungSeong/AIC_Sejong.git ~/AIC_Sejong
cd ~/AIC_Sejong/ws_aic/src
pixi install
```

### 2. Prepare the Eval Container
```bash
export DBX_CONTAINER_MANAGER=docker
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
```

### 3. Run a Policy
```bash
# Terminal 1
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true

# Terminal 2
cd ~/AIC_Sejong/ws_aic/src
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=final_policy.FinalPolicy
```

## Repository Map

| Path | Role |
|------|------|
| `data/` | Competition train/dev/test metadata and submission file |
| `ws_aic/src/aic/` | Official AIC repository and ROS 2 based evaluation environment |
| `ws_aic/src/ais/` | Team-developed packages |
| `ws_aic/src/ais/ais_motion_planning/` | YOLO + multiview based port detection and motion planning |
| `ws_aic/src/ais/ais_auto_capture/` | Gazebo-based automatic data collection |
| `ws_aic/src/ais/ais_yolo_train/` | YOLO training-data collection and evaluation |
| `ws_aic/src/ais/ais_retry_classifier/` | Insertion-failure detection and retry-decision experiments |
| `ws_aic/src/ais/ais_load_model_from_hf/` | Hugging Face Hub model/dataset upload and load utilities |
| `ws_aic/src/ais/ais_eda/` | Multiview bias and position-estimation error analysis notebooks |
| `ws_aic/src/docs/` | Experiment documents and session summaries |
| `readme/` | Korean and English README documents |

## Links

- [Hugging Face Hub](https://huggingface.co/aic-sejong-team)
- [Motion Planning Package](../ws_aic/src/ais/ais_motion_planning/README.md)
- [Automatic Data Collection Package](../ws_aic/src/ais/ais_auto_capture/README.md)
- [Retry Classifier Package](../ws_aic/src/ais/ais_retry_classifier/README.md)
- [Experiment Pseudocode Document](<../ws_aic/src/docs/psuedo code/pseudo_code.md>)
