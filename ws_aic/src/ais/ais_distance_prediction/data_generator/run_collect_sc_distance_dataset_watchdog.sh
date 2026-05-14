#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="${AIC_WS_SRC:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
WS_AIC_ROOT="${AIC_WS_ROOT:-$(cd "$SRC_ROOT/.." && pwd)}"

OUTPUT_DIR="${SC_DISTANCE_OUTPUT_DIR:-$WS_AIC_ROOT/data/distance_prediction/SC}"
LOG_DIR="${SC_DISTANCE_WATCHDOG_LOG_DIR:-/tmp/sc_distance_dataset_watchdog_$(date +%Y%m%d_%H%M%S)}"
RESTART_DELAY_S="${SC_DISTANCE_RESTART_DELAY_S:-10}"
MAX_RESTARTS="${SC_DISTANCE_MAX_RESTARTS:-0}"

TARGET_MODULE_NAME="${SC_DISTANCE_TARGET_MODULE_NAME:-}"
CABLE_TIP_FRAME="${SC_DISTANCE_CABLE_TIP_FRAME:-}"
PORT_NAMES="${SC_DISTANCE_PORT_NAMES:-sc_port_base}"
ALL_PORT_NAMES="${SC_DISTANCE_ALL_PORT_NAMES:-$PORT_NAMES}"
SAMPLING_MODE="${SC_DISTANCE_SAMPLING_MODE:-uniform}"
SAMPLES_PER_PORT="${SC_DISTANCE_SAMPLES_PER_PORT:-5000}"
GRID_STEP_MM="${SC_DISTANCE_GRID_STEP_MM:-1.0}"
GRID_PER_AXIS="${SC_DISTANCE_GRID_PER_AXIS:-5}"
RANDOM_SAMPLES="${SC_DISTANCE_RANDOM_SAMPLES:-0}"
SEED="${SC_DISTANCE_SEED:-42}"
PORT_FRAME_MODE="${SC_DISTANCE_PORT_FRAME_MODE:-auto}"
TCP_POSE_SOURCE="${SC_DISTANCE_TCP_POSE_SOURCE:-controller_state}"
STIFFNESS="${SC_DISTANCE_STIFFNESS:-51.0,50.0,300,5.0,5.0,15.0}"
DAMPING="${SC_DISTANCE_DAMPING:-31.0,30.0,87.0,8.0,8.0,15.0}"
RUN_ID="${SC_DISTANCE_RUN_ID:-}"

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
    echo "target_module_name=${TARGET_MODULE_NAME:-auto-discover}"
    echo "cable_tip_frame=${CABLE_TIP_FRAME:-auto-discover}"
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
    echo "restart_delay_s=$RESTART_DELAY_S"
    echo "max_restarts=$MAX_RESTARTS"
  } >"$LOG_DIR/config.txt"
}

build_command() {
  COLLECT_CMD=(
    stdbuf -oL -eL pixi run python -u ais/ais_distance_prediction/data_generator/collect_sc_distance_dataset.py
    --output "$OUTPUT_DIR"
    --port-names "$PORT_NAMES"
    --all-port-names "$ALL_PORT_NAMES"
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
  )

  if [[ -n "$TARGET_MODULE_NAME" ]]; then
    COLLECT_CMD+=(--target-module-name "$TARGET_MODULE_NAME")
  fi
  if [[ -n "$CABLE_TIP_FRAME" ]]; then
    COLLECT_CMD+=(--cable-tip-frame "$CABLE_TIP_FRAME")
  fi
  if [[ -n "$RUN_ID" ]]; then
    COLLECT_CMD+=(--run-id "$RUN_ID")
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
