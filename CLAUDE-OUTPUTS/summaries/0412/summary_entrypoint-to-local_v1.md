# entrypoint.sh → 로컬 실행 변환 요약

> **요약 (2-3줄):** 기존 `aic()` / `custom()` 함수는 distrobox + Docker로 `/entrypoint.sh`를 실행하는 방식이었다. `/entrypoint.sh` 내부는 Zenoh 라우터를 백그라운드로 띄운 뒤 `ros2 launch aic_bringup aic_gz_bringup.launch.py`를 실행하는 구조다. 소스 빌드 완료 후에는 Docker 없이 동일한 동작을 아래 함수로 대체할 수 있다.

---

## 1. entrypoint.sh 동작 분석

```
/entrypoint.sh [launch_args...]
  ├── source /ws_aic/install/setup.bash
  ├── export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  ├── ros2 run rmw_zenoh_cpp rmw_zenohd &   ← Zenoh 라우터 (백그라운드)
  └── ros2 launch aic_bringup aic_gz_bringup.launch.py [launch_args...]
```

> Docker에서는 컨테이너 간 공유메모리가 안 되므로 `transport/shared_memory/enabled=false`. **로컬 실행 시에는 shared memory 활성화 가능** → 성능 향상.

---

## 2. 기존 함수 → 로컬 함수 변환

### 기존 `aic()` (distrobox 방식)

```bash
aic() {
  aic_ws || return 1
  distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest --name aic_eval 2>/dev/null || true
  distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=true start_aic_engine:=true
}
```

### 변환 후 `aic()` (로컬 방식)

```bash
aic() {
  aic_ws || return 1

  source ~/ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true;transport/shared_memory/transport_optimization/pool_size=536870912'

  # Zenoh 라우터 백그라운드 시작
  ros2 run rmw_zenoh_cpp rmw_zenohd &
  ZENOH_PID=$!
  trap "kill $ZENOH_PID 2>/dev/null" EXIT

  # 평가 환경 실행
  ros2 launch aic_bringup aic_gz_bringup.launch.py \
    ground_truth:=true \
    start_aic_engine:=true
}
```

---

### 기존 `custom()` (distrobox 방식)

```bash
custom() {
  aic_ws || return 1
  distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest --name aic_eval 2>/dev/null || true
  distrobox enter -r aic_eval -- /entrypoint.sh \
    spawn_task_board:=true \
    task_board_x:=0.3 task_board_y:=-0.1 task_board_z:=1.2 \
    task_board_yaw:=0.785 \
    nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 \
    sc_port_0_present:=true sc_port_0_translation:=-0.04 \
    spawn_cable:=true cable_type:=sfp_sc_cable \
    attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
}
```

### 변환 후 `custom()` (로컬 방식)

```bash
custom() {
  aic_ws || return 1

  source ~/ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true;transport/shared_memory/transport_optimization/pool_size=536870912'

  # Zenoh 라우터 백그라운드 시작
  ros2 run rmw_zenoh_cpp rmw_zenohd &
  ZENOH_PID=$!
  trap "kill $ZENOH_PID 2>/dev/null" EXIT

  # 커스텀 씬 실행 (aic_engine 없이, 수동 task board 배치)
  ros2 launch aic_bringup aic_gz_bringup.launch.py \
    spawn_task_board:=true \
    task_board_x:=0.3 task_board_y:=-0.1 task_board_z:=1.2 \
    task_board_yaw:=0.785 \
    nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 \
    sc_port_0_present:=true sc_port_0_translation:=-0.04 \
    spawn_cable:=true cable_type:=sfp_sc_cable \
    attach_cable_to_gripper:=true \
    ground_truth:=true \
    start_aic_engine:=false
}
```

---

## 3. 차이점 비교

| 항목 | 기존 (distrobox) | 변환 후 (로컬) |
|------|-----------------|--------------|
| 실행 방식 | `distrobox enter -- /entrypoint.sh` | 직접 `ros2 run` + `ros2 launch` |
| Zenoh 라우터 | entrypoint 내부에서 자동 시작 | 함수 내 `&` 백그라운드로 직접 시작 |
| Shared Memory | `false` (컨테이너 간 불가) | `true` (네이티브, 성능 향상) |
| 워크스페이스 | `/ws_aic/install/setup.bash` (컨테이너 내부) | `~/ws_aic/install/setup.bash` (호스트) |
| GPU 플래그 | `--nvidia` distrobox 옵션 필요 | 자동 인식 (네이티브 CUDA) |

---

## 4. ~/.bashrc에 추가할 전체 코드

```bash
# AIC 환경 설정
export DBX_CONTAINER_MANAGER=docker

aic_ws() {
  cd ~/ws_aic/src/aic || return 1
}

# 로컬 공통 초기화
_aic_local_init() {
  source ~/ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true;transport/shared_memory/transport_optimization/pool_size=536870912'
  ros2 run rmw_zenoh_cpp rmw_zenohd &
  ZENOH_PID=$!
  trap "kill $ZENOH_PID 2>/dev/null" EXIT
}

# 기본 평가 실행
aic() {
  aic_ws || return 1
  _aic_local_init
  ros2 launch aic_bringup aic_gz_bringup.launch.py \
    ground_truth:=true \
    start_aic_engine:=true
}

# 커스텀 씬 실행
custom() {
  aic_ws || return 1
  _aic_local_init
  ros2 launch aic_bringup aic_gz_bringup.launch.py \
    spawn_task_board:=true \
    task_board_x:=0.3 task_board_y:=-0.1 task_board_z:=1.2 \
    task_board_yaw:=0.785 \
    nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 \
    sc_port_0_present:=true sc_port_0_translation:=-0.04 \
    spawn_cable:=true cable_type:=sfp_sc_cable \
    attach_cable_to_gripper:=true \
    ground_truth:=true \
    start_aic_engine:=false
}
```

---

## 5. 폴리시 실행 (별도 터미널)

`aic()` 또는 `custom()` 실행 후 **새 터미널**에서:

```bash
source ~/ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true;transport/shared_memory/transport_optimization/pool_size=536870912'

# pixi 환경으로 폴리시 실행
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm
```

> **주의:** `aic_engine`이 `aic_model` 노드를 **30초 내**에 찾지 못하면 타임아웃. `start_aic_engine:=true`인 경우 반드시 30초 이내에 폴리시 노드를 실행해야 함.
