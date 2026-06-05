# AIC Sejong

[![Documentation](https://img.shields.io/badge/Documentation-GitHub%20Pages-0A66C2)](https://jungseong.github.io/contests/aic-sejong/)
[![Staged Policy](https://img.shields.io/badge/Staged%20Policy-Cable%20Insertion-0A66C2)](https://jungseong.github.io/contests/aic-sejong/#core-workflows)
[![Vision Pipeline](https://img.shields.io/badge/Vision%20Pipeline-YOLO%20%2B%20Stereo-5B5FC7)](https://jungseong.github.io/contests/aic-sejong/#core-workflows)
[![Data Workflow](https://img.shields.io/badge/Data%20Workflow-Recording%20%26%20Training-FFB000)](https://jungseong.github.io/contests/aic-sejong/#core-workflows)

[한국어](README.ko.md) | [English](README.en.md)

A local workspace for the AI for Industry Challenge. The project develops, trains, and evaluates UR5e cable-insertion policies.

## Documentation

- [Getting Started](https://jungseong.github.io/contests/aic-sejong/#getting-started)
- [Core Workflows](https://jungseong.github.io/contests/aic-sejong/#core-workflows)

## Repository Structure

```
AIC_Sejong/
├── README.md
│
└── ws_aic/                  ← workspace root
    ├── model/               ← trained model weights
    │   ├── ais_yolo/        ← YOLO model weights
    │   │   └── weights/
    │   └── ais_distance_prediction/
    │
    └── src/                 ← source root (pixi.toml location)
        ├── pixi.toml        ← workspace environment definition
        ├── pixi.lock        ← pinned dependencies
        │
        ├── aic/             ← official AIC repository (git submodule)
        │   ├── aic_model/                  ← aic_model node (policy loader)
        │   ├── aic_adapter/                ← model node adapter (C++)
        │   ├── aic_example_policies/       ← example policies (WaveArm, CheatCode, RunACT, etc.)
        │   ├── aic_bringup/                ← simulation launch files
        │   ├── aic_engine/                 ← task orchestration + scoring
        │   ├── aic_controller/             ← robot arm impedance controller
        │   ├── aic_interfaces/             ← ROS 2 message/service interfaces
        │   │   ├── aic_control_interfaces
        │   │   ├── aic_engine_interfaces
        │   │   ├── aic_model_interfaces
        │   │   └── aic_task_interfaces
        │   └── aic_utils/
        │       ├── lerobot_robot_aic/      ← LeRobot ↔ AIC bridge driver
        │       ├── aic_teleoperation/      ← teleoperation utilities
        │       └── aic_mujoco/             ← MuJoCo simulator support
        │
        ├── ais/             ← ★ team-developed packages
        │   ├── ais_auto_capture/       ← automated data collection for YOLO training
        │   ├── ais_early_prediction/   ← early failure prediction (Transformer-based)
        │   ├── ais_eda/                ← multi-view bias exploratory data analysis
        │   ├── ais_encoder/            ← multimodal representation learning (Vision + Touch)
        │   ├── ais_load_model_from_hf/ ← HuggingFace model load/upload utility
        │   ├── ais_motion_planning/    ← YOLO + stereo-based port detection and motion planning
        │   ├── docker/                 ← Dockerfile, docker-compose.yaml
        │   └── ais_ours_policy/        ← ROS 2 node wrapper
        │       ├── data_gen_node/      ← data generation node
        │       └── motion_planning_node/ ← motion planning node
        │
        ├── data/            ← datasets
        │   ├── lerobot/     ← LeRobot format datasets (master branch)
        │   └── yolo/        ← YOLO training data (by date, e.g. 20260426/)
        │
        └── docs/            ← documentation
            └── summaries/   ← Claude session summaries (0405, 0409, ... 0423)
```

## Getting Started

### Requirements

- Ubuntu 24.04
- NVIDIA GPU recommended
- Docker, NVIDIA Container Toolkit, Distrobox, Pixi

### Install Dependencies

```bash
git clone https://github.com/JungSeong/AIC_Sejong.git ~/AIC_Sejong
cd ~/AIC_Sejong/ws_aic/src
pixi install
```

### Prepare Eval Container

```bash
export DBX_CONTAINER_MANAGER=docker
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
```

## Main Workflows

### Simulation and Policy Execution

```bash
# Terminal 1
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true

# Terminal 2
cd ~/AIC_Sejong/ws_aic/src
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=my_policy_node.StagedPolicy
```

### YOLO Data Collection

```bash
cd ~/AIC_Sejong/ws_aic/src
pixi run python ais/ais_motion_planning/collect_dataset.py
```

### ACT Training

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

## References

- Official guide: `ws_aic/src/aic/docs/getting_started.md`
- Scoring rules: `ws_aic/src/aic/docs/scoring.md`
- Policy guide: `ws_aic/src/aic/docs/policy.md`
- Motion planning package: `ws_aic/src/ais/ais_motion_planning/`
