#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_AIC_ROOT="${AIC_WS_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
COLLECT_SCRIPT="${SFP_YOLO_COLLECT_SCRIPT:-$SCRIPT_DIR/collect_dataset.py}"
ZENOH_SESSION_CONFIG_HOST="${SFP_YOLO_ZENOH_SESSION_CONFIG_HOST:-$WS_AIC_ROOT/src/ais/docker/aic_eval/aic_zenoh_config.json5}"

DISTROBOX_NAME="${SFP_YOLO_DISTROBOX:-aic_eval_arm64_nvidia}"
MOUNTS="${SFP_YOLO_MOUNTS:-0 1 2 3 4}"
MOUNTS="${MOUNTS//,/ }"
EPISODES="${SFP_YOLO_EPISODES:-100}"
N_VIEWPOINTS="${SFP_YOLO_N_VIEWPOINTS:-15}"
VAL_RATIO="${SFP_YOLO_VAL_RATIO:-0.1}"
MOVE_SETTLE_S="${SFP_YOLO_MOVE_SETTLE_S:-2.5}"
INTERVAL_S="${SFP_YOLO_INTERVAL_S:-0.1}"
SIM_BOOT_WAIT_S="${SFP_YOLO_SIM_BOOT_WAIT_S:-45}"
SIM_STOP_WAIT_S="${SFP_YOLO_SIM_STOP_WAIT_S:-5}"
SIM_FORCE_STOP_WAIT_S="${SFP_YOLO_SIM_FORCE_STOP_WAIT_S:-2}"
SIM_CLEANUP_TIMEOUT_S="${SFP_YOLO_SIM_CLEANUP_TIMEOUT_S:-30}"
ZENOH_BOOT_WAIT_S="${SFP_YOLO_ZENOH_BOOT_WAIT_S:-3}"
RESET_ZENOH="${SFP_YOLO_RESET_ZENOH:-1}"
STOP_ZENOH_ON_EXIT="${SFP_YOLO_STOP_ZENOH_ON_EXIT:-1}"
KILL_EXISTING_SIM="${SFP_YOLO_KILL_EXISTING_SIM:-1}"
RUN_ID="${SFP_YOLO_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${SFP_YOLO_LOG_DIR:-/tmp/sfp_yolo_by_mount_$RUN_ID}"
OUTPUT_DIR="${SFP_YOLO_OUTPUT_DIR:-}"

mkdir -p "$LOG_DIR"

SIM_PID=""
ZENOH_PID=""
ZENOH_STARTED=0
EXITING=0

cleanup_existing_zenoh_router() {
  if [[ "$RESET_ZENOH" != "1" ]]; then
    return
  fi

  echo "[$(date --iso-8601=seconds)] cleanup existing Zenoh router" | tee -a "$LOG_DIR/run.log"
  distrobox enter --no-tty --name "$DISTROBOX_NAME" -- bash -s -- "$SIM_CLEANUP_TIMEOUT_S" \
    >>"$LOG_DIR/run.log" 2>&1 <<'AIC_ZENOH_CLEANUP'
    set +e
    cleanup_timeout_s="${1:-30}"

    port_7447_busy() {
      if command -v ss >/dev/null 2>&1; then
        ss -H -ltn "sport = :7447" 2>/dev/null | grep -q .
        return $?
      fi
      if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:7447 -sTCP:LISTEN >/dev/null 2>&1
        return $?
      fi
      return 1
    }

    pkill -INT -f "rmw_zenohd" 2>/dev/null || true
    sleep 1
    pkill -TERM -f "rmw_zenohd" 2>/dev/null || true
    sleep 1
    pkill -KILL -f "rmw_zenohd" 2>/dev/null || true

    deadline=$((SECONDS + cleanup_timeout_s))
    while pgrep -f "rmw_zenohd" >/dev/null 2>&1 || port_7447_busy; do
      if (( SECONDS >= deadline )); then
        echo "ERROR: Zenoh cleanup timed out; tcp/7447 or rmw_zenohd still remains"
        pgrep -af "rmw_zenohd" 2>/dev/null || true
        ss -H -ltnp "sport = :7447" 2>/dev/null || true
        lsof -nP -iTCP:7447 -sTCP:LISTEN 2>/dev/null || true
        exit 1
      fi
      sleep 1
    done

    echo "Zenoh cleanup complete: rmw_zenohd stopped and tcp/7447 is free"
AIC_ZENOH_CLEANUP
}

start_zenoh_router() {
  local zenoh_log="$LOG_DIR/zenoh_router.log"

  if distrobox enter --no-tty --name "$DISTROBOX_NAME" -- pgrep -f "rmw_zenohd" >/dev/null 2>&1; then
    echo "[$(date --iso-8601=seconds)] Zenoh router already running; reusing it" | tee -a "$LOG_DIR/run.log"
    return
  fi

  echo "[$(date --iso-8601=seconds)] starting Zenoh router log=$zenoh_log" | tee -a "$LOG_DIR/run.log"
  setsid distrobox enter --no-tty --name "$DISTROBOX_NAME" -- bash -lc '
    . /ws_aic/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_zenoh_cpp
    export ZENOH_ROUTER_CONFIG_URI=/aic_zenoh_config.json5
    ZENOH_CONFIG_OVERRIDE='\''mode="router"'\''
    ZENOH_CONFIG_OVERRIDE+=''\'';listen/endpoints=["tcp/[::]:7447"]'\''
    ZENOH_CONFIG_OVERRIDE+=''\'';connect/endpoints=[]'\''
    ZENOH_CONFIG_OVERRIDE+=''\'';routing/router/peers_failover_brokering=true'\''
    ZENOH_CONFIG_OVERRIDE+=''\'';transport/shared_memory/enabled=false'\''
    export ZENOH_CONFIG_OVERRIDE
    exec ros2 run rmw_zenoh_cpp rmw_zenohd
  ' >"$zenoh_log" 2>&1 &
  ZENOH_PID=$!
  ZENOH_STARTED=1

  sleep "$ZENOH_BOOT_WAIT_S"
  if ! distrobox enter --no-tty --name "$DISTROBOX_NAME" -- pgrep -f "rmw_zenohd" >/dev/null 2>&1; then
    echo "[$(date --iso-8601=seconds)] ERROR: Zenoh router did not start; see $zenoh_log" | tee -a "$LOG_DIR/run.log"
    return 1
  fi
}

cleanup_zenoh_router() {
  if [[ "$ZENOH_STARTED" != "1" || "$STOP_ZENOH_ON_EXIT" != "1" ]]; then
    return
  fi

  echo "[$(date --iso-8601=seconds)] stopping Zenoh router" | tee -a "$LOG_DIR/run.log"
  if [[ -n "${ZENOH_PID:-}" ]] && kill -0 "$ZENOH_PID" 2>/dev/null; then
    kill -INT "-$ZENOH_PID" 2>/dev/null || kill -INT "$ZENOH_PID" 2>/dev/null || true
    sleep 1
    kill -TERM "-$ZENOH_PID" 2>/dev/null || kill -TERM "$ZENOH_PID" 2>/dev/null || true
    sleep 1
    kill -KILL "-$ZENOH_PID" 2>/dev/null || kill -KILL "$ZENOH_PID" 2>/dev/null || true
    wait "$ZENOH_PID" 2>/dev/null || true
  fi
  distrobox enter --no-tty --name "$DISTROBOX_NAME" -- pkill -f "rmw_zenohd" >/dev/null 2>&1 || true
  ZENOH_PID=""
  ZENOH_STARTED=0
}

cleanup_existing_sim_processes() {
  if [[ "$KILL_EXISTING_SIM" != "1" ]]; then
    return
  fi

  echo "[$(date --iso-8601=seconds)] cleanup existing AIC/Gazebo processes" | tee -a "$LOG_DIR/run.log"
  distrobox enter --no-tty --name "$DISTROBOX_NAME" -- bash -s -- "$SIM_CLEANUP_TIMEOUT_S" \
    >>"$LOG_DIR/run.log" 2>&1 <<'AIC_SIM_CLEANUP'
    set +e
    cleanup_timeout_s="${1:-30}"

    patterns=(
      "gz sim"
      "gz_server"
      "gzserver"
      "gzclient"
      "gazebo"
      "ign gazebo"
      "ruby.*gz"
      "aic_gz_bringup"
      "aic_engine"
      "aic_model"
      "aic_adapter"
      "component_container"
      "robot_state_publisher"
      "ros2_control_node"
      "controller_manager"
      "ros_gz_bridge"
      "parameter_bridge"
      "ros2.*spawner"
      "spawn_task_board"
      "static_transform_publisher"
      "topic_tools"
      "tf_relay"
      "rviz2"
    )
    entrypoint_pattern="/entrypoint.sh"

    all_patterns=("${patterns[@]}" "$entrypoint_pattern")

    pkill_one() {
      local signal="$1"
      local pattern="$2"

      if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        sudo -n pkill -A "-$signal" -f "$pattern" 2>/dev/null && return 0
        sudo -n pkill "-$signal" -f "$pattern" 2>/dev/null && return 0
      fi

      pkill -A "-$signal" -f "$pattern" 2>/dev/null && return 0
      pkill "-$signal" -f "$pattern" 2>/dev/null || true
    }

    has_matches() {
      local pattern

      for pattern in "${all_patterns[@]}"; do
        if pgrep -f "$pattern" >/dev/null 2>&1; then
          return 0
        fi
      done

      return 1
    }

    list_matches() {
      local pattern

      for pattern in "${all_patterns[@]}"; do
        pgrep -af "$pattern" 2>/dev/null || true
      done
    }

    cleanup_pass() {
      local signal="$1"
      local pattern

      echo "cleanup pass signal=$signal"
      for pattern in "${patterns[@]}" "$entrypoint_pattern"; do
        pkill_one "$signal" "$pattern"
      done
    }

    cleanup_pass INT
    sleep 2
    cleanup_pass TERM
    sleep 2
    cleanup_pass KILL

    deadline=$((SECONDS + cleanup_timeout_s))
    while has_matches; do
      if (( SECONDS >= deadline )); then
        echo "ERROR: cleanup timed out; refusing to start a new Gazebo while these processes remain:"
        list_matches
        exit 1
      fi
      sleep 1
    done

    echo "cleanup complete: no matching AIC/Gazebo processes remain"
AIC_SIM_CLEANUP
  local status=$?
  if [[ "$status" -ne 0 ]]; then
    echo "[$(date --iso-8601=seconds)] ERROR: existing AIC/Gazebo processes still remain; aborting before new sim start" \
      | tee -a "$LOG_DIR/run.log"
    return "$status"
  fi

  sleep "$SIM_STOP_WAIT_S"
}

cleanup_sim() {
  local do_existing_cleanup=1
  if [[ "$EXITING" == "1" ]]; then
    do_existing_cleanup=0
  fi

  if [[ -n "${SIM_PID:-}" ]] && kill -0 "$SIM_PID" 2>/dev/null; then
    echo "[$(date --iso-8601=seconds)] stopping sim pid=$SIM_PID" | tee -a "$LOG_DIR/run.log"
    kill -INT "-$SIM_PID" 2>/dev/null || kill -INT "$SIM_PID" 2>/dev/null || true
    sleep "$SIM_STOP_WAIT_S"
    if kill -0 "$SIM_PID" 2>/dev/null; then
      kill -TERM "-$SIM_PID" 2>/dev/null || kill -TERM "$SIM_PID" 2>/dev/null || true
    fi
    sleep "$SIM_FORCE_STOP_WAIT_S"
    if kill -0 "$SIM_PID" 2>/dev/null; then
      kill -KILL "-$SIM_PID" 2>/dev/null || kill -KILL "$SIM_PID" 2>/dev/null || true
    fi
    wait "$SIM_PID" 2>/dev/null || true
  fi
  SIM_PID=""
  if [[ "$do_existing_cleanup" == "1" ]]; then
    cleanup_existing_sim_processes || return $?
  fi
}

trap 'EXITING=1; cleanup_sim; cleanup_existing_sim_processes; cleanup_zenoh_router; exit 130' INT TERM
trap 'EXITING=1; cleanup_sim; cleanup_existing_sim_processes; cleanup_zenoh_router' EXIT

write_config() {
  {
    echo "ws_aic_root=$WS_AIC_ROOT"
    echo "collect_script=$COLLECT_SCRIPT"
    echo "zenoh_session_config_host=$ZENOH_SESSION_CONFIG_HOST"
    echo "distrobox_name=$DISTROBOX_NAME"
    echo "mounts=$MOUNTS"
    echo "episodes=$EPISODES"
    echo "n_viewpoints=$N_VIEWPOINTS"
    echo "val_ratio=$VAL_RATIO"
    echo "move_settle_s=$MOVE_SETTLE_S"
    echo "interval_s=$INTERVAL_S"
    echo "sim_boot_wait_s=$SIM_BOOT_WAIT_S"
    echo "sim_stop_wait_s=$SIM_STOP_WAIT_S"
    echo "sim_force_stop_wait_s=$SIM_FORCE_STOP_WAIT_S"
    echo "sim_cleanup_timeout_s=$SIM_CLEANUP_TIMEOUT_S"
    echo "zenoh_boot_wait_s=$ZENOH_BOOT_WAIT_S"
    echo "reset_zenoh=$RESET_ZENOH"
    echo "stop_zenoh_on_exit=$STOP_ZENOH_ON_EXIT"
    echo "kill_existing_sim=$KILL_EXISTING_SIM"
    echo "run_id=$RUN_ID"
    echo "log_dir=$LOG_DIR"
    echo "output_dir=${OUTPUT_DIR:-default}"
  } >"$LOG_DIR/config.txt"
}

start_sim_for_mount() {
  local mount_idx="$1"
  local sim_log="$LOG_DIR/sim_mount_${mount_idx}.log"
  local mount_args=()
  local i

  if [[ ! "$mount_idx" =~ ^[0-4]$ ]]; then
    echo "[$(date --iso-8601=seconds)] ERROR: invalid mount index '$mount_idx'. Use 0..4." | tee -a "$LOG_DIR/run.log"
    exit 2
  fi

  for i in 0 1 2 3 4; do
    if [[ "$i" == "$mount_idx" ]]; then
      mount_args+=("nic_card_mount_${i}_present:=true")
    else
      mount_args+=("nic_card_mount_${i}_present:=false")
    fi
  done

  echo "[$(date --iso-8601=seconds)] starting sim mount=$mount_idx log=$sim_log" | tee -a "$LOG_DIR/run.log"
  setsid distrobox enter --no-tty --name "$DISTROBOX_NAME" -- bash -lc '
    . /ws_aic/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_zenoh_cpp
    export ZENOH_ROUTER_CHECK_ATTEMPTS=5
    export ZENOH_SESSION_CONFIG_URI=/aic_zenoh_config.json5
    export ZENOH_ROUTER_CONFIG_URI=/aic_zenoh_config.json5
    export ZENOH_CONFIG_OVERRIDE=";transport/shared_memory/enabled=false"
    exec ros2 launch aic_bringup aic_gz_bringup.launch.py "$@"
  ' aic_gazebo_launch \
    "${mount_args[@]}" \
    spawn_task_board:=true \
    spawn_cable:=true \
    cable_type:=sfp_sc_cable \
    attach_cable_to_gripper:=true \
    ground_truth:=true \
    start_aic_engine:=false \
    gazebo_gui:=true \
    launch_rviz:=false \
    >"$sim_log" 2>&1 &
  SIM_PID=$!
}

collect_mount() {
  local mount_idx="$1"
  local collect_log="$LOG_DIR/collect_mount_${mount_idx}.log"
  local stem_prefix="${RUN_ID}_nic_mount_${mount_idx}_"
  local collect_cmd=(
    pixi run python "$COLLECT_SCRIPT"
    --target SFP
    --episodes "$EPISODES"
    --n-viewpoints "$N_VIEWPOINTS"
    --val-ratio "$VAL_RATIO"
    --move-settle-s "$MOVE_SETTLE_S"
    --interval-s "$INTERVAL_S"
    --stem-prefix "$stem_prefix"
  )

  if [[ -n "$OUTPUT_DIR" ]]; then
    collect_cmd+=(--output "$OUTPUT_DIR")
  fi

  echo "[$(date --iso-8601=seconds)] collecting mount=$mount_idx prefix=$stem_prefix log=$collect_log" \
    | tee -a "$LOG_DIR/run.log"
  printf 'command=' | tee -a "$LOG_DIR/run.log"
  printf '%q ' "${collect_cmd[@]}" | tee -a "$LOG_DIR/run.log"
  echo | tee -a "$LOG_DIR/run.log"

  RMW_IMPLEMENTATION=rmw_zenoh_cpp \
    ZENOH_ROUTER_CHECK_ATTEMPTS=5 \
    ZENOH_SESSION_CONFIG_URI="$ZENOH_SESSION_CONFIG_HOST" \
    ZENOH_CONFIG_OVERRIDE=";transport/shared_memory/enabled=false" \
    "${collect_cmd[@]}" 2>&1 | tee "$collect_log"
  return "${PIPESTATUS[0]}"
}

write_config
{
  echo "[$(date --iso-8601=seconds)] config"
  cat "$LOG_DIR/config.txt"
} | tee -a "$LOG_DIR/run.log"

cleanup_existing_sim_processes || exit $?
cleanup_existing_zenoh_router || exit $?
start_zenoh_router || exit $?

for mount_idx in $MOUNTS; do
  cleanup_sim || exit $?
  start_sim_for_mount "$mount_idx" || {
    cleanup_sim
    exit 1
  }
  echo "[$(date --iso-8601=seconds)] waiting ${SIM_BOOT_WAIT_S}s for sim boot" | tee -a "$LOG_DIR/run.log"
  sleep "$SIM_BOOT_WAIT_S"

  collect_mount "$mount_idx"
  status=$?
  if [[ "$status" -ne 0 ]]; then
    echo "[$(date --iso-8601=seconds)] collect failed mount=$mount_idx status=$status" | tee -a "$LOG_DIR/run.log"
    cleanup_sim
    exit "$status"
  fi

  cleanup_sim || exit $?
done

echo "[$(date --iso-8601=seconds)] all mounts completed" | tee -a "$LOG_DIR/run.log"
