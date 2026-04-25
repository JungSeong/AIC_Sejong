#!/usr/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE="${ZENOH_CONFIG_OVERRIDE:-transport/shared_memory/enabled=false}"

# data_gen_policy 패키지 경로 추가 (setup.py editable install의 .pth가 비어있는 문제 우회)
PIXI_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PIXI_PROJECT_ROOT}:${PYTHONPATH:-}"
