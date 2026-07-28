# ais_auto_capture — 자동 데이터 수집 스크립트

## 스크립트 구성

| 파일 | 환경 | 역할 |
|---|---|---|
| `collect_lerobot_data.py` | distrobox (x86) | LeRobot 에피소드 자동 수집 |
| `collect_lerobot_data_aarch.py` | 소스 빌드 (aarch64) | LeRobot 에피소드 자동 수집 |
| `collect_yolo_data_aarch.py` | 소스 빌드 (aarch64) | YOLO 데이터셋 자동 수집 |
| `collect_portoffset_randomization_data.py` | distrobox/eval engine | PortOffsetCollect vision-offset 정렬 데이터 자동 수집 |
| `convert_to_lerobot.py` | 공통 | raw 에피소드 → LeRobot 포맷 변환 |

---

## 1. collect_lerobot_data_aarch.py — LeRobot 에피소드 수집 (aarch64)

세트당 7개 trial(NIC×5 + SC×2)을 Gazebo에서 자동 실행하며 LeRobot 포맷으로 저장한다.

**흐름**
1. trial별 랜덤 파라미터로 aic_engine config YAML 생성
2. Zenoh 라우터 → Gazebo(`aic_gz_bringup`) → `LeRobot` policy 순으로 시작
3. `episode_summary.json` 파일 수로 완료 감지
4. Gazebo 종료 → 다음 세트 반복

**사용법**
```bash
# 기본: 10 세트 × 7 에피소드
python3 collect_lerobot_data_aarch.py

# 50 세트, 보드 위치/yaw 랜덤화
python3 collect_lerobot_data_aarch.py --sets 50 --diversify

# 명령어만 출력 (실제 실행 X)
python3 collect_lerobot_data_aarch.py --sets 5 --dry-run

# Gazebo GUI·RViz 없이 실행
python3 collect_lerobot_data_aarch.py --headless

# LeRobot 로컬 저장 + HuggingFace 업로드
python3 collect_lerobot_data_aarch.py \
  --lerobot-out-dir ~/data \
  --lerobot-repo-id aic-sejong-team/aic-dataset
```

---

## 2. collect_yolo_data_aarch.py — YOLO 데이터셋 수집 (aarch64)

시나리오별로 Gazebo를 별도 세션으로 실행하며 3대 카메라 스냅샷 + TF 기반 bbox 라벨을 자동 생성한다.

**흐름 (시나리오당)**
1. 랜덤 파라미터로 aic_engine config YAML 생성 + scenario_params JSON 저장
2. Zenoh 라우터 → Gazebo → `LeRobot` policy 시작
   - `LeRobot` policy를 사용해야 Task Board가 실제로 spawn됨
   - `autocapture`는 lifecycle만 수행하므로 entity spawn이 보장되지 않음
3. 카메라 데이터 및 포트 TF 확인 후 스냅샷 N장 수집
4. YOLO 라벨 자동 생성 (TF 기반 핀홀 투영)
5. Gazebo 종료 → 다음 시나리오

**시나리오 구성 (세트당 7개)**
- NIC rail 0~4: SFP 포트 레이블 (`sfp_port`, class 0)
- SC rail 0~1: SC 포트 레이블 (`sc_port`, class 1)

**출력 구조**
```
<output>/<YYYYMMDD>/
├── images/
│   ├── train/  s00001_nic0_snap0000_left.jpg, ...
│   └── val/
├── labels/
│   ├── train/  s00001_nic0_snap0000_left.txt
│   └── val/
└── data.yaml
```

**사용법**
```bash
# 기본: 10 세트 × 7 시나리오, 스냅샷 20장
python3 collect_yolo_data_aarch.py --sets 10

# 스냅샷 수 / 보드 위치 랜덤화
python3 collect_yolo_data_aarch.py --sets 20 --snapshots 30 --diversify

# Gazebo GUI 없이, 명령어만 출력 테스트
python3 collect_yolo_data_aarch.py --sets 5 --headless --dry-run

# 출력 경로 지정
python3 collect_yolo_data_aarch.py --sets 10 --output ~/data/yolo
```

**주요 옵션**

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--sets` | 10 | 수집 세트 수 |
| `--snapshots` | 20 | 시나리오당 스냅샷 수 |
| `--diversify` | off | 보드 x/y 위치 랜덤화 |
| `--headless` | off | Gazebo GUI·RViz 비활성 |
| `--gazebo-wait` | 60 | Gazebo 초기화 대기(초) |
| `--val-ratio` | 0.3 | 검증 세트 비율 |
| `--output` | `src/data/yolo` | YOLO 데이터셋 출력 경로 |
| `--dry-run` | off | 명령어만 출력 |

---

## 3. collect_portoffset_randomization_data.py — PortOffsetCollect 정렬 데이터 수집

랜덤화된 task board/cable/lighting 조건에서 `data_gen_node.PortOffsetCollect` policy를 실행해 vision-offset 정렬 학습 데이터를 수집한다. Distrobox에는 항상 일반 사용자 권한으로 진입한다.

```bash
pixi run python ais/ais_auto_capture/collect_portoffset_randomization_data.py \
  --trials 20 \
  --samples-per-trial 24 \
  --port-types sfp,sc \
  --dataset-version 0726-001 \
  --push-to-hub true \
  --vision-offset-repo-id aic-sejong-team/aic-vision-offset-dataset \
  --vision-offset-hf-revision 0726-001 \
  --upload-on-port-type sc \
  --record-rosbag true \
  --launch-rviz false \
  --cleanup
```

### 파라미터 정리

`collect_portoffset_randomization_data.py`는 Gazebo를 trial마다 일반 사용자 권한의 Distrobox로 자동 실행하고, SFP/SC target과 simulator 조명을 함께 랜덤화합니다. 별도 옵션 없이 항상 `distrobox enter aic_eval` 형태로 진입하며 `-r`은 사용하지 않습니다.

`--push-to-hub`, `--color-log`, `--randomize-lighting`, `--launch-rviz`, `--record-rosbag`은 positive/negative 옵션 쌍을 사용하지 않고 하나의 옵션에 `true` 또는 `false`를 전달합니다.

```bash
--push-to-hub false
--color-log true
--randomize-lighting true
--launch-rviz false
--record-rosbag true
```

CLI 기본값과 choices는 `portoffset_randomization/constants.py`에서 관리합니다. 분포 그래프도 실제 시나리오 상수와 CLI 기본값을 읽어 생성하므로 값을 변경한 후 아래 명령으로 갱신합니다.

`--cleanup`은 이전 수집이 비정상 종료되어 남은 collector 소유 프로세스를 정리한 뒤 새 수집을 계속합니다. `--cleanup-only`는 같은 정리만 수행하고 trial을 시작하지 않습니다. 두 옵션 모두 데이터셋이나 생성된 결과 파일을 삭제하지 않습니다.

### rosbag 자동 녹화

`--record-rosbag true`는 각 trial을 독립 MCAP으로 기록합니다.

```text
Gazebo + Zenoh 시작
  → rosbag 준비 확인
  → policy 실행 및 종료
  → rosbag SIGINT finalize 및 검증
  → Gazebo 종료
  → 다음 trial
```

| 옵션 | 기본값 | 역할 |
|---|---:|---|
| `--record-rosbag` | `false` | `true`이면 자동 녹화를 활성화합니다. |
| `--rosbag-output-dir` | `AIC_Sejong/rosbags/portoffset` | 최상위 출력 경로입니다. |
| `--rosbag-topics` | `/clock`, TF, controller, 카메라 토픽 | 녹화할 토픽 목록입니다. |
| `--rosbag-start-timeout-s` | `20` | recorder 시작 대기시간입니다. |
| `--rosbag-stop-grace-s` | `30` | SIGINT finalize 대기시간입니다. |

기본 출력은 `AIC_Sejong/rosbags/portoffset/{dataset-version}/{run-id}/{trial}`입니다. `metadata.yaml`, 0보다 큰 message count, 모든 MCAP의 시작·종료 magic이 유효해야 green bold `RECORDING COMPLETED` 로그가 출력되고 다음 trial로 진행합니다. 실패 시 해당 run을 중단합니다.

| 함수 | 핵심 책임 |
|---|---|
| `_run_trial()` | `Gazebo → rosbag → policy` 시작 순서와 `policy → rosbag → Gazebo` 종료 순서를 보장합니다. |
| `start_rosbag()` / `wait_for_rosbag_start()` | recorder를 시작하고 녹화 준비를 확인합니다. |
| `stop_rosbag()` / `validate_rosbag()` | SIGINT 종료와 MCAP 완결성 검증을 수행합니다. |
| `cleanup_stale_processes()` | 비정상 종료 후 남은 recorder를 SIGINT 우선으로 정리합니다. |

```bash
# 잔존 프로세스 정리 후 수집
pixi run python ais/ais_auto_capture/collect_portoffset_randomization_data.py \
  --cleanup \
  --trials 20

# 잔존 프로세스 정리 후 즉시 종료
pixi run python ais/ais_auto_capture/collect_portoffset_randomization_data.py \
  --cleanup-only
```

```bash
cd ~/AIC_Sejong/ws_aic/src
pixi run python ais/ais_auto_capture/plot_scenario_randomization.py
```

| 파라미터 종류 | 범위 | 역할 |
|---|---|---|
| Target port type | `--port-types sfp`, `sc`, `sfp,sc` 기본 `sfp,sc` | 수집할 포트 계열을 선택합니다. 기본은 SFP와 SC를 모두 수집합니다. |
| Target port order | `--port-order random` 또는 `round_robin`, 기본 `random` | SFP/SC trial 배치 순서를 결정합니다. |
| Trial count | `--trials`, 기본 `20` | 생성 및 수집할 Gazebo trial 개수입니다. |
| Samples per trial | `--samples-per-trial`, 기본 `24` | 한 trial의 수집 시도 수입니다. timestamp/visibility gating에서 거부되면 실제 저장 수는 더 적을 수 있습니다. |
| SFP task board X | `0.13 ~ 0.17 m` | SFP/NIC trial에서 task board의 world X 위치를 랜덤화합니다. |
| SFP task board Y | `-0.25 ~ -0.20 m` | SFP/NIC trial에서 task board의 world Y 위치를 랜덤화합니다. |
| SFP task board yaw | `3.10 ~ 3.1415 rad` | SFP/NIC trial에서 task board 회전각을 랜덤화합니다. |
| SC task board X | `0.15 ~ 0.19 m` | SC trial에서 task board의 world X 위치를 랜덤화합니다. |
| SC task board Y | `-0.05 ~ 0.05 m` | SC trial에서 task board의 world Y 위치를 랜덤화합니다. |
| SC task board yaw | `3.10 ~ 3.1415 rad` | SC trial에서 task board 회전각을 랜덤화합니다. |
| Task board Z / roll / pitch | `z=1.14 m`, `roll=0`, `pitch=0` | 보드 높이와 기울기는 고정해 xy/yaw 중심의 scene variation만 적용합니다. |
| SFP NIC rail | `0 ~ 4` 중 1개 | SFP target이 들어갈 NIC rail을 선택합니다. |
| SFP port index | `sfp_port_0` 또는 `sfp_port_1` | SFP target port를 선택합니다. |
| SFP NIC translation | `-0.0215 ~ 0.0234 m` | 선택된 NIC card mount의 rail 방향 위치를 랜덤화합니다. |
| SFP NIC yaw | `-10 ~ +10 deg` | 선택된 NIC card mount의 yaw를 랜덤화합니다. |
| SC rail | `0 ~ 1` 중 1개 | SC target이 들어갈 SC rail을 선택합니다. |
| SC translation | `-0.06 ~ 0.055 m` | 선택된 SC mount의 rail 방향 위치를 랜덤화합니다. |
| Gripper offset noise | 각 축 `-0.002 ~ +0.002 m` | cable gripper offset 기준값에 미세 오차를 더합니다. SFP 기준값은 `[0, 0.015385, 0.04245] m`, SC 기준값은 `[0, 0.015385, 0.04045] m`입니다. |
| Cable RPY noise | `--cable-rpy-noise-deg`, 기본 `±20 deg` | cable 초기 roll/pitch/yaw 기준값에 회전 오차를 더합니다. |
| Robot home joint noise | `--robot-joint-noise-deg`, 기본 `±4 deg` | robot home joint 초기값에 관절별 오차를 더합니다. |
| Collect XYZ range | X/Y 기본 `-50 ~ +50 mm`, Z 기본 `0 ~ 100 mm` | PortOffsetCollect가 포트 기준 위치 offset sample을 생성하는 범위입니다. `--dx-*`, `--dy-*`, `--dz-*`로 축별 override할 수 있습니다. |
| Collect RPY range | roll/pitch 기본 `±25 deg`, yaw 기본 `±35 deg` | PortOffsetCollect가 포트 기준 자세 offset sample을 생성하는 범위입니다. `--roll-*`, `--pitch-*`, `--yaw-*`로 축별 override할 수 있습니다. |
| RPY norm cap | `--rpy-norm-max-rad`, 기본 미사용 | sampling된 RPY vector magnitude를 제한합니다. |
| Actual RPY norm filter | `--actual-rpy-norm-max-rad`, 기본 `--rpy-norm-max-rad` 정책값 사용 | 저장 직전 실제 plug-port quaternion angle이 큰 sample을 제외합니다. |
| Timestamp sync tolerance | `--sync-tolerance-ms`, 기본 `30 ms` | 좌·중·우 Image, ControllerState, 동적 TF의 ROS source timestamp skew가 이 값을 넘으면 sample을 저장하지 않습니다. |
| Summary grace / trial interval | summary 후 `3 s`, trial 간 `3 s` | AIC engine의 scoring/reset 완료를 기다린 뒤 종료하고, 종료 검증 후 다음 trial을 시작합니다. |
| Simulator teardown | 정상 종료: SIGINT `5 s` → SIGTERM `2 s` → SIGKILL `1 s`; Ctrl+C: SIGTERM `2 s` → SIGKILL `1 s` | 외부 Distrobox wrapper PGID와 config marker로 찾은 내부 ROS 2/Gazebo PGID를 각각 등록합니다. Ctrl+C 시 추가 SIGINT는 cleanup이 끝날 때까지 무시하며, 내부 simulator와 wrapper 종료 및 registry 갱신을 완료합니다. |
| Pre-run cleanup | `--cleanup`, 기본 off | run marker와 registry로 소유권이 확인된 이전 policy/Gazebo PGID를 정리한 후 새 수집을 시작합니다. |
| Cleanup only | `--cleanup-only`, 기본 off | 이전 수집의 잔존 PGID만 정리하고 trial을 시작하지 않은 채 종료합니다. |
| Visibility filter | `--min-visible-cameras` 기본 `1`, `--visibility-margin-px` 기본 `8 px` | 포트가 충분히 보이는 sample만 저장하도록 카메라 visibility 기준을 정합니다. |
| Lighting randomization | 기본 `true`, `--randomize-lighting false`로 비활성화 | trial마다 Gazebo world SDF를 생성해 조명/배경을 truncated Gaussian 분포로 랜덤화합니다. |
| Light intensity scale | `N(μ=1.0, σ=0.1167)`, `[0.65, 1.35]` | `enclosure_light`, `ceiling_01`, `ceiling_02` intensity에 곱할 scale입니다. |
| Light color jitter | `N(μ=0, σ=0.04)`, `[-0.12, 0.12]` | 각 light diffuse RGB를 channel별로 변화시킵니다. |
| Light pose jitter | XY `N(0, 0.0833) m`, `[-0.25, 0.25] m`; Z `N(0, 0.0667) m`, `[-0.20, 0.20] m` | 각 light 위치를 변화시켜 highlight/shadow 위치를 바꿉니다. |
| Ambient / background | ambient `N(0.04, 0.0133)`, `[0, 0.08]`; background `N(0.14, 0.02)`, `[0.08, 0.20]` | scene ambient와 background 밝기를 변화시킵니다. |
| Gazebo mode | `--headless` on/off | 활성화하면 Gazebo GUI와 RViz를 모두 비활성화합니다. |
| RViz | `--launch-rviz true\|false`, 기본 `true` | `false`이면 Gazebo GUI 상태와 관계없이 RViz만 비활성화합니다. `--headless` 사용 시에는 항상 비활성화됩니다. |
| Color log | 기본 `true`, `--color-log false` 또는 `NO_COLOR`로 비활성화 | trial별 랜덤화 로그를 색상/볼드로 구분해 출력합니다. |
| Trial timeout | 기본 `time-limit-s + 180 s` | `episode_summary.json` 생성 대기 제한 시간입니다. |
| Dataset version | `--dataset-version`, 기본 빈 문자열 | `data/ais_portoffset_randomization/{version}` 하위에 저장할 버전을 지정합니다. |
| Hugging Face upload | 기본 `false`, `--push-to-hub true`로 활성화 | PortOffsetCollect policy에 `AIC_VISION_OFFSET_PUSH_TO_HUB`를 명시 전달합니다. 실험 중 의도치 않은 업로드를 막기 위해 runner 기본값은 `false`입니다. |
| Hugging Face target | `--vision-offset-repo-id`, `--vision-offset-hf-revision`, `--vision-offset-hf-path-in-repo`, `--hf-private` | 업로드할 HF dataset repo, revision, repo 내부 경로, private repo 생성 여부를 지정합니다. |
| Upload port filter | `--upload-on-port-type` 값 `sfp`, `sc`, 또는 빈 문자열 기본 빈 문자열 | 특정 포트 타입 trial에서만 업로드하도록 제한합니다. 빈 문자열이면 PortOffsetCollect가 포트 타입 제한 없이 판단합니다. |

### Timestamp gating

center image `header.stamp`가 sample 기준 시각입니다. 세 Image와 ControllerState를 먼저 검사하고, port/plug TF를 그 시각으로 조회한 뒤 모든 동적 source가 `--sync-tolerance-ms` 안에 있을 때만 저장합니다. static TF의 `0` stamp는 허용하며, settle 대기와 TCP 속도 기반 정지 판정은 사용하지 않습니다.

sidecar JSON과 `metadata.jsonl`의 `timestamps`에는 `capture_stamp_ns`, camera별 stamp, `controller_stamp_ns`, port/plug TF stamp, `skew_ns`, `sync_tolerance_ns`, `sync_valid`, `dataset_write_stamp_ns`가 기록됩니다.

| 함수 | 변경 내용 |
|---|---|
| `init_runtime()` | sync tolerance를 초기화하고 settle/속도 설정을 제거합니다. |
| `_observation_sync_metadata()` | camera/controller stamp 기록과 1차 gating을 수행합니다. |
| `_lookup_transform_at()` / `_tf_sync_metadata()` | center image 시각 TF 조회와 2차 gating을 수행합니다. |
| `_stage_collect()` | 정지 대기 없이 observation을 한 번 읽고 gating을 통과한 sample만 저장기로 전달합니다. |
| `_save_xyz_rpy_sample()` | 통과한 sample의 timestamp와 skew를 실제 dataset metadata에 기록합니다. |

기본 실행에서 허용 오차를 명시하려면 다음 옵션을 추가합니다.

```bash
--sync-tolerance-ms 30
```

---

## 5. 랜덤화 파라미터 범위 (`collect_lerobot_data*.py`)

LeRobot episode 자동 수집 스크립트는 세트당 SFP/NIC trial 5개와 SC trial 2개를 생성한다.

| 파라미터 종류 | 범위 | 역할 |
|---|---|---|
| NIC/SFP card translation | `-0.0215 ~ 0.0234 m` | SFP/NIC target card의 rail 방향 위치를 랜덤화합니다. |
| NIC/SFP card yaw | `-10 ~ +10 deg` | SFP/NIC target card의 yaw를 랜덤화합니다. |
| SC port translation | `-0.06 ~ 0.055 m` | SC target port의 rail 방향 위치를 랜덤화합니다. |
| SFP task board X | `0.13 ~ 0.17 m` | `--diversify` 사용 시 SFP/NIC trial의 task board X 위치를 랜덤화합니다. |
| SFP task board Y | `-0.25 ~ -0.15 m` | `--diversify` 사용 시 SFP/NIC trial의 task board Y 위치를 랜덤화합니다. |
| SC task board X | `0.15 ~ 0.19 m` | `--diversify` 사용 시 SC trial의 task board X 위치를 랜덤화합니다. |
| SC task board Y | `-0.05 ~ 0.05 m` | `--diversify` 사용 시 SC trial의 task board Y 위치를 랜덤화합니다. |
| Task board yaw | `0.0 ~ 3.1415 rad` | 모든 trial에서 task board yaw를 랜덤화합니다. |
| Gripper offset noise | 각 축 `-0.002 ~ +0.002 m` | cable gripper offset 기준값에 미세 오차를 추가합니다. |
| SFP gripper offset base | `[0, 0.015385, 0.04245] m` | SFP cable grasp offset 기준값입니다. |
| SC gripper offset base | `[0, 0.015385, 0.04045] m` | SC cable grasp offset 기준값입니다. |

---

## 6. collect_lerobot_data.py — LeRobot 에피소드 수집 (distrobox / x86)

`collect_lerobot_data_aarch.py`와 동일한 역할이지만 distrobox 컨테이너 환경에서 동작한다.
aarch64 환경에서는 `collect_lerobot_data_aarch.py`를 사용할 것.
