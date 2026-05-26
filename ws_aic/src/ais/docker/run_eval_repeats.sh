#!/usr/bin/env bash

set -uo pipefail

RUNS="${1:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-${SCRIPT_DIR}/docker-compose.yaml}"
RESULTS_ROOT="${AIC_RESULTS_ROOT:-/home/swlinux/aic_results}"
SESSION_TAG="${AIC_REPEAT_SESSION_TAG:-repeat_$(date +%Y%m%d_%H%M%S)}"
RUN_TAG_PREFIX="${AIC_REPEAT_RUN_TAG_PREFIX:-${SESSION_TAG}}"
LOG_DIR="${AIC_REPEAT_LOG_DIR:-${RESULTS_ROOT}/repeat_logs/${SESSION_TAG}}"
SUMMARY_CSV="${LOG_DIR}/summary.csv"
CURRENT_RUN_TAG=""

if ! [[ "${RUNS}" =~ ^[0-9]+$ ]] || [[ "${RUNS}" -lt 1 ]]; then
  echo "Usage: $0 [run_count]" >&2
  echo "Example: $0 10" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

csv_header() {
  local header="timestamp,run_index,run_tag,exit_code,total"
  local trial category

  for trial in 1 2 3; do
    header+=",trial_${trial}_tier_1_score"
    header+=",trial_${trial}_tier_2_score"
    header+=",trial_${trial}_tier_3_score"
    for category in contacts duration insertion_force trajectory_efficiency trajectory_smoothness; do
      header+=",trial_${trial}_${category}_score"
    done
  done

  printf '%s\n' "${header}"
}

yaml_total() {
  local scoring_file="$1"
  awk -F': ' '$1 == "total" {print $2; exit}' "${scoring_file}"
}

yaml_score() {
  local scoring_file="$1"
  local trial="trial_${2}:"
  local tier="tier_${3}:"
  local category="${4:-}"

  awk -v trial="${trial}" -v tier="${tier}" -v category="${category}" '
    $0 == trial {
      in_trial = 1
      in_tier = 0
      in_category = (category == "")
      next
    }
    in_trial && /^trial_[0-9]+:/ && $0 != trial {
      exit
    }
    in_trial && $0 == "  " tier {
      in_tier = 1
      in_category = (category == "")
      next
    }
    in_trial && in_tier && /^  tier_[0-9]+:/ && $0 != "  " tier {
      exit
    }
    category != "" && in_trial && in_tier && $0 == "      " category ":" {
      in_category = 1
      next
    }
    category != "" && in_trial && in_tier && in_category && /^      [^ ].*:/ && $0 != "      " category ":" {
      exit
    }
    in_trial && in_tier && in_category && /^[[:space:]]+score:/ {
      sub(/^[[:space:]]*score:[[:space:]]*/, "")
      print
      exit
    }
  ' "${scoring_file}"
}

compose() {
  RUN_TAG="$1" docker compose -f "${COMPOSE_FILE}" "${@:2}"
}

cleanup_current_run() {
  if [[ -n "${CURRENT_RUN_TAG}" ]]; then
    echo "[$(date --iso-8601=seconds)] interrupted; cleaning up RUN_TAG=${CURRENT_RUN_TAG}" >&2
    compose "${CURRENT_RUN_TAG}" down --remove-orphans >/dev/null 2>&1 || true
  fi
}

trap 'cleanup_current_run; exit 130' INT TERM

append_summary_row() {
  local run_index="$1"
  local run_tag="$2"
  local exit_code="$3"
  local scoring_file="${RESULTS_ROOT}/${run_tag}/scoring.yaml"
  local timestamp total row trial score

  timestamp="$(date --iso-8601=seconds)"

  if [[ ! -f "${scoring_file}" ]]; then
    printf '%s,%s,%s,%s,\n' "${timestamp}" "${run_index}" "${run_tag}" "${exit_code}" >> "${SUMMARY_CSV}"
    return
  fi

  cp "${scoring_file}" "${LOG_DIR}/scoring_run_${run_index}.yaml"

  total="$(yaml_total "${scoring_file}")"
  row="${timestamp},${run_index},${run_tag},${exit_code},${total}"

  for trial in 1 2 3; do
    row+=",$(yaml_score "${scoring_file}" "${trial}" 1)"
    row+=",$(yaml_score "${scoring_file}" "${trial}" 2)"
    row+=",$(yaml_score "${scoring_file}" "${trial}" 3)"
    row+=",$(yaml_score "${scoring_file}" "${trial}" 2 "contacts")"
    row+=",$(yaml_score "${scoring_file}" "${trial}" 2 "duration")"
    row+=",$(yaml_score "${scoring_file}" "${trial}" 2 "insertion force")"
    row+=",$(yaml_score "${scoring_file}" "${trial}" 2 "trajectory efficiency")"
    row+=",$(yaml_score "${scoring_file}" "${trial}" 2 "trajectory smoothness")"
  done

  printf '%s\n' "${row}" >> "${SUMMARY_CSV}"
}

if [[ ! -f "${SUMMARY_CSV}" ]]; then
  csv_header > "${SUMMARY_CSV}"
fi

cd "${AIS_DIR}" || exit 1

echo "runs=${RUNS}"
echo "compose_file=${COMPOSE_FILE}"
echo "results_root=${RESULTS_ROOT}"
echo "log_dir=${LOG_DIR}"
echo "summary_csv=${SUMMARY_CSV}"

for run_index in $(seq 1 "${RUNS}"); do
  run_tag="${RUN_TAG_PREFIX}_$(printf '%02d' "${run_index}")"
  CURRENT_RUN_TAG="${run_tag}"
  compose_log="${LOG_DIR}/compose_run_${run_index}.log"

  echo "[$(date --iso-8601=seconds)] start run ${run_index}/${RUNS}: RUN_TAG=${run_tag}"

  compose "${run_tag}" down --remove-orphans >> "${compose_log}" 2>&1

  compose "${run_tag}" up --abort-on-container-exit --exit-code-from eval 2>&1 | tee "${compose_log}"
  exit_code="${PIPESTATUS[0]}"

  append_summary_row "${run_index}" "${run_tag}" "${exit_code}"

  compose "${run_tag}" down --remove-orphans >> "${compose_log}" 2>&1

  echo "[$(date --iso-8601=seconds)] end run ${run_index}/${RUNS}: exit_code=${exit_code}"
  echo "  scoring=${RESULTS_ROOT}/${run_tag}/scoring.yaml"
  echo "  summary=${SUMMARY_CSV}"
  CURRENT_RUN_TAG=""
done
