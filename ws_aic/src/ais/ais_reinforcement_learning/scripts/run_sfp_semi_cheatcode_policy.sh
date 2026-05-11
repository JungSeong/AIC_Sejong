#!/usr/bin/env bash
set -euo pipefail

export AIC_YOLO_MODEL_PATH="${AIC_YOLO_MODEL_PATH:-/home/whyz/aic_sejong/ws_aic/model/ais_yolo/approach/SFP/weights/best.pt}"
export AIC_DISTANCE_MODEL_PATH="${AIC_DISTANCE_MODEL_PATH:-/home/whyz/aic_sejong/ws_aic/model/ais_distance_prediction/sfp_distance_resnet50_left_center_right_concat/best.pt}"

pixi run ros2 run aic_model aic_model \
  --ros-args \
  -p use_sim_time:=true \
  -p policy:=ais_reinforcement_learning.SfpSemiCheatcodePolicy
