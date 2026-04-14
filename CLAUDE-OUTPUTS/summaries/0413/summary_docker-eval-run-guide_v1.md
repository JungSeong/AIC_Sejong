# Docker Eval 컨테이너로 정책 실행하기

> Docker eval 컨테이너(distrobox) + 호스트 pixi 환경 조합으로 정책을 실행하는 방법 정리.
> `source ~/ws_aic/install/setup.bash`는 **필요 없다** — pixi가 자체 환경을 제공하기 때문.

---

## 핵심 구조

```
┌──────────────────────────────────┐     ┌──────────────────────────────────┐
│  Terminal 1: Docker eval 컨테이너   │     │  Terminal 2: 호스트 (pixi 환경)    │
│                                  │     │                                  │
│  • Zenoh 라우터                   │◄───►│  • aic_model (내 정책)             │
│  • Gazebo 시뮬레이션               │ ROS │  • pixi run 으로 실행              │
│  • aic_engine (스코어링)           │     │  • source 불필요                  │
│  • aic_controller                │     │                                  │
└──────────────────────────────────┘     └──────────────────────────────────┘
```

**`source ~/ws_aic/install/setup.bash`가 불필요한 이유:** `pixi run`이 실행 시 자체적으로 ROS 2 + 의존성 환경을 구성한다. 호스트에 별도로 ROS 2를 설치하거나 워크스페이스를 빌드할 필요가 없다.

---

## 실행 순서

### Terminal 1 — Eval 컨테이너 (먼저 실행)

```bash
# 최초 1회: 컨테이너 생성
export DBX_CONTAINER_MANAGER=docker
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval

# 매번 실행
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true
```

> `/entrypoint.sh`가 내부적으로 **Zenoh 라우터 + Gazebo + aic_engine**을 모두 올린다.
> Zenoh 라우터를 별도로 실행할 필요 없음.

**확인:** Gazebo/RViz 창이 뜨고 터미널에 `No node with name 'aic_model' found. Retrying...`이 나오면 정상.

---

### Terminal 2 — 내 정책 실행 (30초 이내)

```bash
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=my_policy_node.Baseline
```

이게 전부다. **`source`, `ros2 run rmw_zenoh_cpp rmw_zenohd` 모두 불필요.**

---

## v1 문서(trained-model-to-policy_v1.md)의 "8. 실행 순서"와의 차이

v1 문서는 **소스 빌드(source build) 워크플로우**를 기준으로 작성되어 있다.

| 항목 | 소스 빌드 (v1) | Docker eval (본 문서) |
|------|---------------|----------------------|
| Zenoh 라우터 | `ros2 run rmw_zenoh_cpp rmw_zenohd` 별도 실행 | **불필요** — `/entrypoint.sh`에 포함 |
| 시뮬레이션 | `ros2 launch aic_bringup ...` 수동 실행 | **불필요** — `/entrypoint.sh`에 포함 |
| ROS 환경 설정 | `source ~/ws_aic/install/setup.bash` 필수 | **불필요** — `pixi run`이 자동 처리 |
| 정책 실행 | `ros2 run aic_model aic_model ...` | `pixi run ros2 run aic_model aic_model ...` |
| 실행 순서 | Zenoh → 정책 → 시뮬레이션 | **eval 컨테이너 → 정책** (2단계) |

---

## ground_truth 옵션

| 용도 | 명령 |
|------|------|
| **개발/디버깅** (CheatCode 등) | `/entrypoint.sh ground_truth:=true start_aic_engine:=true` |
| **실제 평가 시뮬레이션** | `/entrypoint.sh ground_truth:=false start_aic_engine:=true` |
| **환경 탐색 전용** (엔진 없이) | `/entrypoint.sh ground_truth:=true start_aic_engine:=false` |

---

## 자주 하는 실수

| 실수 | 해결 |
|------|------|
| eval 컨테이너보다 정책을 먼저 실행 | eval 컨테이너가 **Zenoh 라우터를 제공**하므로 반드시 먼저 실행 |
| 정책 실행이 30초 넘게 지연 | aic_engine이 타임아웃 → eval 컨테이너 재시작 후 재실행 |
| `source setup.bash` 시도 | 호스트에 ROS가 없으면 실패 → `pixi run`만 쓰면 됨 |
| `pixi run` 없이 `ros2 run` 직접 실행 | 호스트에 ROS가 설치되어 있지 않으면 `ros2 command not found` |
| 코드 수정 후 바로 실행 | `pixi reinstall ros-kilted-my-policy-node` 먼저 실행 |

---

## wandb 설정

`lerobot-train`에서 `--wandb.enable=true` 사용 시 wandb 로그인 필요:

```bash
pixi run wandb login
# API key 입력 (https://wandb.ai/authorize 에서 확인)
```

> wandb는 **훈련(lerobot-train) 전용**. 정책 실행(추론)에는 불필요.
