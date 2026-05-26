#!/usr/bin/env bash
set -uo pipefail

NIC_TRANSLATION_LOW="${SFP_RPY_NIC_TRANSLATION_LOW:--0.0215}"
NIC_TRANSLATION_HIGH="${SFP_RPY_NIC_TRANSLATION_HIGH:-0.0234}"
NIC_YAW_LOW="${SFP_RPY_NIC_YAW_LOW:--0.17453292519943295}"
NIC_YAW_HIGH="${SFP_RPY_NIC_YAW_HIGH:-0.17453292519943295}"
NIC_TRANSLATION_M="${SFP_RPY_NIC_TRANSLATION_M:-}"
NIC_YAW_RAD="${SFP_RPY_NIC_YAW_RAD:-}"
DISTROBOX_NAME="${AIC_DISTROBOX_NAME:-aic_eval_arm64_nvidia}"
GAZEBO_GUI="${SFP_RPY_GAZEBO_GUI:-true}"
LAUNCH_RVIZ="${SFP_RPY_LAUNCH_RVIZ:-true}"
RUN_ID="${SFP_RPY_DISTANCE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ENV_FILE="${SFP_RPY_SCENE_ENV_FILE:-/tmp/sfp_rpy_distance_scene_${RUN_ID}.env}"

uniform_random() {
  local low="$1"
  local high="$2"
  local r="$RANDOM"
  awk -v low="$low" -v high="$high" -v r="$r" 'BEGIN { printf "%.12f", low + (r / 32767.0) * (high - low) }'
}

if [[ -z "$NIC_TRANSLATION_M" ]]; then
  NIC_TRANSLATION_M="$(uniform_random "$NIC_TRANSLATION_LOW" "$NIC_TRANSLATION_HIGH")"
fi
if [[ -z "$NIC_YAW_RAD" ]]; then
  NIC_YAW_RAD="$(uniform_random "$NIC_YAW_LOW" "$NIC_YAW_HIGH")"
fi

mkdir -p "$(dirname "$ENV_FILE")"
{
  printf 'export SFP_RPY_DISTANCE_RUN_ID=%q\n' "$RUN_ID"
  printf 'export SFP_RPY_DISTANCE_NIC_TRANSLATION_M=%q\n' "$NIC_TRANSLATION_M"
  printf 'export SFP_RPY_DISTANCE_NIC_YAW_RAD=%q\n' "$NIC_YAW_RAD"
} >"$ENV_FILE"

echo "SFP RPY randomized scene"
echo "run_id=$RUN_ID"
echo "nic_card_mount_0_translation=$NIC_TRANSLATION_M"
echo "nic_card_mount_0_yaw=$NIC_YAW_RAD"
echo "collector_env_file=$ENV_FILE"
echo
echo "In the collector terminal, run:"
echo "  source $ENV_FILE"
echo "  ./run_collect_sfp_RPY_dataset_watchdog.sh"
echo

exec distrobox enter -r "$DISTROBOX_NAME" -- /entrypoint.sh \
  nic_card_mount_0_present:=true \
  nic_card_mount_0_translation:="$NIC_TRANSLATION_M" \
  nic_card_mount_0_yaw:="$NIC_YAW_RAD" \
  spawn_task_board:=true \
  spawn_cable:=true \
  cable_type:=sfp_sc_cable \
  attach_cable_to_gripper:=true \
  ground_truth:=true \
  start_aic_engine:=false \
  gazebo_gui:="$GAZEBO_GUI" \
  launch_rviz:="$LAUNCH_RVIZ"
