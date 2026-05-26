#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${AIC_WS_SRC:-/home/whyz/aic_sejong/ws_aic/src}"
POLICY="distance_prediction_policy.DebugSfpDistancePolicy"
RUNS="${AIC_DEBUG_POLICY_RUNS:-5}"
TIMEOUT_S="${AIC_DEBUG_POLICY_TIMEOUT_S:-240}"
LOG_DIR="${AIC_DEBUG_POLICY_LOG_DIR:-/tmp/aic_debug_sfp_policy_$(date +%Y%m%d_%H%M%S)}"

export AIC_YOLO_MODEL_PATH="${AIC_YOLO_MODEL_PATH:-/home/whyz/aic_sejong/ws_aic/model/ais_yolo/approach/SFP/weights/best.pt}"
export AIC_DISTANCE_MODEL_PATH="${AIC_DISTANCE_MODEL_PATH:-/home/whyz/aic_sejong/ws_aic/model/ais_distance_prediction/sfp_distance_resnet50_left_center_right_concat/best.pt}"

run_loop() {
  mkdir -p "$LOG_DIR"
  cd "$SRC_ROOT"
  echo "policy=$POLICY" | tee "$LOG_DIR/config.txt"
  echo "yolo_model=$AIC_YOLO_MODEL_PATH" | tee -a "$LOG_DIR/config.txt"
  echo "distance_model=$AIC_DISTANCE_MODEL_PATH" | tee -a "$LOG_DIR/config.txt"
  echo "runs=$RUNS timeout_s=$TIMEOUT_S" | tee -a "$LOG_DIR/config.txt"

  for index in $(seq 1 "$RUNS"); do
    log_file="$LOG_DIR/run_${index}.log"
    echo "[$(date --iso-8601=seconds)] start run $index/$RUNS" | tee -a "$LOG_DIR/runner.log"
    set +e
    timeout "$TIMEOUT_S" pixi run ros2 run aic_model aic_model \
      --ros-args -p use_sim_time:=true -p policy:="$POLICY" \
      >"$log_file" 2>&1
    status=$?
    set -e
    echo "[$(date --iso-8601=seconds)] end run $index/$RUNS status=$status log=$log_file" \
      | tee -a "$LOG_DIR/runner.log"
    sleep 2
  done
}

if [[ "${1:-}" == "--foreground" ]]; then
  run_loop
else
  mkdir -p "$LOG_DIR"
  nohup "$0" --foreground >"$LOG_DIR/nohup.log" 2>&1 &
  echo "started background debug policy loop"
  echo "pid=$!"
  echo "log_dir=$LOG_DIR"
  echo "tail -f $LOG_DIR/runner.log"
fi
