#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="${AIC_WS_SRC:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
WS_AIC_ROOT="${AIC_WS_ROOT:-$(cd "$SRC_ROOT/.." && pwd)}"

OUTPUT_DIR="${SFP_RPY_DISTANCE_OUTPUT_DIR:-$WS_AIC_ROOT/data/distance_prediction/SFP_RPY}"
LOG_DIR="${SFP_RPY_DISTANCE_WATCHDOG_LOG_DIR:-/tmp/sfp_rpy_distance_dataset_watchdog_$(date +%Y%m%d_%H%M%S)}"
RESTART_DELAY_S="${SFP_RPY_DISTANCE_RESTART_DELAY_S:-10}"
MAX_RESTARTS="${SFP_RPY_DISTANCE_MAX_RESTARTS:-0}"

TARGET_MODULE_NAME="${SFP_RPY_DISTANCE_TARGET_MODULE_NAME:-nic_card_mount_0}"
CABLE_TIP_FRAME="${SFP_RPY_DISTANCE_CABLE_TIP_FRAME:-cable_0/sfp_tip_link}"
PORT_NAMES="${SFP_RPY_DISTANCE_PORT_NAMES:-sfp_port_0,sfp_port_1}"
ALL_PORT_NAMES="${SFP_RPY_DISTANCE_ALL_PORT_NAMES:-$PORT_NAMES}"
SAMPLING_MODE="${SFP_RPY_DISTANCE_SAMPLING_MODE:-uniform}"
SAMPLES_PER_PORT="${SFP_RPY_DISTANCE_SAMPLES_PER_PORT:-5000}"
GRID_STEP_MM="${SFP_RPY_DISTANCE_GRID_STEP_MM:-1.0}"
GRID_PER_AXIS="${SFP_RPY_DISTANCE_GRID_PER_AXIS:-5}"
RANDOM_SAMPLES="${SFP_RPY_DISTANCE_RANDOM_SAMPLES:-0}"
SEED="${SFP_RPY_DISTANCE_SEED:-42}"
PORT_FRAME_MODE="${SFP_RPY_DISTANCE_PORT_FRAME_MODE:-entrance}"
TCP_POSE_SOURCE="${SFP_RPY_DISTANCE_TCP_POSE_SOURCE:-controller_state}"
STIFFNESS="${SFP_RPY_DISTANCE_STIFFNESS:-1000,1000,1000,80,80,80}"
DAMPING="${SFP_RPY_DISTANCE_DAMPING:-80,80,80,20,20,20}"
RPY_JITTER_RAD="${SFP_RPY_DISTANCE_RPY_JITTER_RAD:-0.04}"
RPY_JITTER_AXES="${SFP_RPY_DISTANCE_RPY_JITTER_AXES:-roll,pitch,yaw}"
NOMINAL_GRIPPER_OFFSET_X="${SFP_RPY_DISTANCE_NOMINAL_GRIPPER_OFFSET_X:-0.0}"
NOMINAL_GRIPPER_OFFSET_Y="${SFP_RPY_DISTANCE_NOMINAL_GRIPPER_OFFSET_Y:-0.015385}"
NOMINAL_GRIPPER_OFFSET_Z="${SFP_RPY_DISTANCE_NOMINAL_GRIPPER_OFFSET_Z:-0.04245}"
NOMINAL_GRIPPER_ROLL="${SFP_RPY_DISTANCE_NOMINAL_GRIPPER_ROLL:-0.4432}"
NOMINAL_GRIPPER_PITCH="${SFP_RPY_DISTANCE_NOMINAL_GRIPPER_PITCH:--0.4838}"
NOMINAL_GRIPPER_YAW="${SFP_RPY_DISTANCE_NOMINAL_GRIPPER_YAW:-1.3303}"
NIC_TRANSLATION_M="${SFP_RPY_DISTANCE_NIC_TRANSLATION_M:-}"
NIC_YAW_RAD="${SFP_RPY_DISTANCE_NIC_YAW_RAD:-}"
RUN_ID="${SFP_RPY_DISTANCE_RUN_ID:-}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export RCUTILS_LOGGING_BUFFERED_STREAM="${RCUTILS_LOGGING_BUFFERED_STREAM:-0}"
export RCUTILS_LOGGING_USE_STDOUT="${RCUTILS_LOGGING_USE_STDOUT:-1}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

SAMPLES_PATH="$OUTPUT_DIR/samples.jsonl"
if [[ -z "$RUN_ID" && ! -s "$SAMPLES_PATH" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

RUN_ID_FILE="$LOG_DIR/run_id"
if [[ -n "$RUN_ID" ]]; then
  printf '%s\n' "$RUN_ID" >"$RUN_ID_FILE"
elif [[ -f "$RUN_ID_FILE" ]]; then
  RUN_ID="$(<"$RUN_ID_FILE")"
fi

write_config() {
  {
    echo "src_root=$SRC_ROOT"
    echo "output_dir=$OUTPUT_DIR"
    echo "log_dir=$LOG_DIR"
    echo "run_id=${RUN_ID:-auto-from-samples-jsonl}"
    echo "target_module_name=$TARGET_MODULE_NAME"
    echo "cable_tip_frame=$CABLE_TIP_FRAME"
    echo "port_names=$PORT_NAMES"
    echo "all_port_names=$ALL_PORT_NAMES"
    echo "sampling_mode=$SAMPLING_MODE"
    echo "samples_per_port=$SAMPLES_PER_PORT"
    echo "grid_step_mm=$GRID_STEP_MM"
    echo "grid_per_axis=$GRID_PER_AXIS"
    echo "random_samples=$RANDOM_SAMPLES"
    echo "seed=$SEED"
    echo "port_frame_mode=$PORT_FRAME_MODE"
    echo "tcp_pose_source=$TCP_POSE_SOURCE"
    echo "stiffness=$STIFFNESS"
    echo "damping=$DAMPING"
    echo "rpy_jitter_rad=$RPY_JITTER_RAD"
    echo "rpy_jitter_axes=$RPY_JITTER_AXES"
    echo "nominal_gripper_offset_x=$NOMINAL_GRIPPER_OFFSET_X"
    echo "nominal_gripper_offset_y=$NOMINAL_GRIPPER_OFFSET_Y"
    echo "nominal_gripper_offset_z=$NOMINAL_GRIPPER_OFFSET_Z"
    echo "nominal_gripper_roll=$NOMINAL_GRIPPER_ROLL"
    echo "nominal_gripper_pitch=$NOMINAL_GRIPPER_PITCH"
    echo "nominal_gripper_yaw=$NOMINAL_GRIPPER_YAW"
    echo "nic_translation_m=${NIC_TRANSLATION_M:-not_provided}"
    echo "nic_yaw_rad=${NIC_YAW_RAD:-not_provided}"
    echo "restart_delay_s=$RESTART_DELAY_S"
    echo "max_restarts=$MAX_RESTARTS"
  } >"$LOG_DIR/config.txt"
}

build_command() {
  COLLECT_CMD=(
    stdbuf -oL -eL pixi run python -u ais/ais_distance_prediction/data_generator/collect_sfp_RPY_dataset.py
    --output "$OUTPUT_DIR"
    --target-module-name "$TARGET_MODULE_NAME"
    --port-names "$PORT_NAMES"
    --all-port-names "$ALL_PORT_NAMES"
    --cable-tip-frame "$CABLE_TIP_FRAME"
    --sampling-mode "$SAMPLING_MODE"
    --samples-per-port "$SAMPLES_PER_PORT"
    --grid-step-mm "$GRID_STEP_MM"
    --grid-per-axis "$GRID_PER_AXIS"
    --random-samples "$RANDOM_SAMPLES"
    --seed "$SEED"
    --port-frame-mode "$PORT_FRAME_MODE"
    --tcp-pose-source "$TCP_POSE_SOURCE"
    --stiffness "$STIFFNESS"
    --damping "$DAMPING"
    --rpy-jitter-rad "$RPY_JITTER_RAD"
    --rpy-jitter-axes "$RPY_JITTER_AXES"
    --nominal-gripper-offset-x "$NOMINAL_GRIPPER_OFFSET_X"
    --nominal-gripper-offset-y "$NOMINAL_GRIPPER_OFFSET_Y"
    --nominal-gripper-offset-z "$NOMINAL_GRIPPER_OFFSET_Z"
    --nominal-gripper-roll "$NOMINAL_GRIPPER_ROLL"
    --nominal-gripper-pitch "$NOMINAL_GRIPPER_PITCH"
    --nominal-gripper-yaw "$NOMINAL_GRIPPER_YAW"
  )

  if [[ -n "$RUN_ID" ]]; then
    COLLECT_CMD+=(--run-id "$RUN_ID")
  fi
  if [[ -n "$NIC_TRANSLATION_M" ]]; then
    COLLECT_CMD+=(--nic-translation-m "$NIC_TRANSLATION_M")
  fi
  if [[ -n "$NIC_YAW_RAD" ]]; then
    COLLECT_CMD+=(--nic-yaw-rad "$NIC_YAW_RAD")
  fi
  if [[ -s "$SAMPLES_PATH" ]]; then
    COLLECT_CMD+=(--resume)
  fi
}

write_config
{
  echo "[$(date --iso-8601=seconds)] watchdog config"
  cat "$LOG_DIR/config.txt"
} | tee -a "$LOG_DIR/watchdog.log"

cd "$SRC_ROOT" || exit 1

restart_count=0
attempt=1
while true; do
  build_command
  log_file="$LOG_DIR/collect_attempt_${attempt}.log"

  {
    echo "[$(date --iso-8601=seconds)] start attempt=$attempt restart_count=$restart_count"
    printf 'command='
    printf '%q ' "${COLLECT_CMD[@]}"
    echo
    echo "log_file=$log_file"
  } | tee -a "$LOG_DIR/watchdog.log"

  "${COLLECT_CMD[@]}" 2>&1 | tee "$log_file"
  status=${PIPESTATUS[0]}

  echo "[$(date --iso-8601=seconds)] end attempt=$attempt status=$status" \
    | tee -a "$LOG_DIR/watchdog.log"

  if [[ "$status" -eq 0 ]]; then
    echo "[$(date --iso-8601=seconds)] collection completed; watchdog exiting" \
      | tee -a "$LOG_DIR/watchdog.log"
    exit 0
  fi

  if [[ "$status" -eq 130 || "$status" -eq 143 ]]; then
    echo "[$(date --iso-8601=seconds)] interrupted; watchdog exiting without restart" \
      | tee -a "$LOG_DIR/watchdog.log"
    exit "$status"
  fi

  restart_count=$((restart_count + 1))
  if [[ "$MAX_RESTARTS" -gt 0 && "$restart_count" -gt "$MAX_RESTARTS" ]]; then
    echo "[$(date --iso-8601=seconds)] max restarts reached; watchdog exiting" \
      | tee -a "$LOG_DIR/watchdog.log"
    exit "$status"
  fi

  echo "[$(date --iso-8601=seconds)] restarting after ${RESTART_DELAY_S}s" \
    | tee -a "$LOG_DIR/watchdog.log"
  sleep "$RESTART_DELAY_S"
  attempt=$((attempt + 1))
done
