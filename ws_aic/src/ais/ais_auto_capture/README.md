# ais_auto_capture — 자동 데이터 수집 스크립트

## 스크립트 구성

| 파일 | 환경 | 역할 |
|---|---|---|
| `collect_lerobot_data.py` | distrobox (x86) | LeRobot 에피소드 자동 수집 |
| `collect_lerobot_data_aarch.py` | 소스 빌드 (aarch64) | LeRobot 에피소드 자동 수집 |
| `collect_yolo_data_aarch.py` | 소스 빌드 (aarch64) | YOLO 데이터셋 자동 수집 |
| `collect_portoffset_randomization.py` | distrobox/eval engine | PortOffsetCollect vision-offset 정렬 데이터 자동 수집 |
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

## 3. collect_portoffset_randomization.py — PortOffsetCollect 정렬 데이터 수집

랜덤화된 task board/cable/lighting 조건에서 `data_gen_node.PortOffsetCollect` policy를 실행해 vision-offset 정렬 학습 데이터를 수집한다. 이 runner는 기본적으로 rootless distrobox 사용을 권장한다.

```bash
  pixi run python ais/ais_auto_capture/collect_portoffset_randomization.py \
    --trials 20 \
    --samples-per-trial 24 \
    --port-types sfp,sc \
    --dataset-version v1.0 \
    --push-to-hub \
    --vision-offset-repo-id aic-sejong-team/aic-vision-offset-dataset \
    --vision-offset-hf-revision main \
    --upload-on-port-type sc \
    --headless \
    --rootless-distrobox \
    --cleanup
```

### 파라미터 정리

`collect_portoffset_randomization.py`는 Gazebo를 trial마다 rootless distrobox로 자동 실행하고, SFP/SC target과 simulator 조명을 함께 랜덤화합니다.

| 파라미터 종류 | 범위 | 역할 |
|---|---|---|
| Target port type | `--port-types sfp`, `sc`, `sfp,sc` 기본 `sfp,sc` | 수집할 포트 계열을 선택합니다. 기본은 SFP와 SC를 모두 수집합니다. |
| Target port order | `--port-order round_robin` 또는 `random`, 기본 `round_robin` | SFP/SC trial 배치 순서를 결정합니다. |
| Trial count | `--trials`, 기본 `20` | 생성 및 수집할 Gazebo trial 개수입니다. |
| Samples per trial | `--samples-per-trial`, 기본 `24` | 한 trial에서 저장할 vision-offset sample 수입니다. |
| SFP task board X | `0.13 ~ 0.17 m` | SFP/NIC trial에서 task board의 world X 위치를 랜덤화합니다. |
| SFP task board Y | `-0.25 ~ -0.20 m` | SFP/NIC trial에서 task board의 world Y 위치를 랜덤화합니다. |
| SFP task board yaw | `0.55 ~ 0.80 rad` | SFP/NIC trial에서 task board 회전각을 랜덤화합니다. |
| SC task board X | `0.15 ~ 0.19 m` | SC trial에서 task board의 world X 위치를 랜덤화합니다. |
| SC task board Y | `-0.05 ~ 0.05 m` | SC trial에서 task board의 world Y 위치를 랜덤화합니다. |
| SC task board yaw | `0.0 ~ 3.1415 rad` | SC trial에서 task board 회전각을 랜덤화합니다. |
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
| Capture settle time | `--capture-settle-s`, 기본 `0.25 s` | offset 적용 후 이미지/metadata를 저장하기 전 안정화 대기 시간입니다. |
| Visibility filter | `--min-visible-cameras` 기본 `1`, `--visibility-margin-px` 기본 `8 px` | 포트가 충분히 보이는 sample만 저장하도록 카메라 visibility 기준을 정합니다. |
| Lighting randomization | 기본 on, `--no-randomize-lighting`으로 off | trial마다 Gazebo world SDF를 생성해 조명/배경을 랜덤화합니다. |
| Light intensity scale | `0.65 ~ 1.35` | `enclosure_light`, `ceiling_01`, `ceiling_02` intensity에 곱할 scale 범위입니다. |
| Light color jitter | 기본 `±0.12` | 각 light diffuse RGB를 channel별로 흔듭니다. |
| Light pose jitter | XY 기본 `±0.25 m`, Z 기본 `±0.20 m` | 각 light 위치를 약간 이동시켜 highlight/shadow 위치를 바꿉니다. |
| Ambient / background | ambient `0.0 ~ 0.08`, background `0.08 ~ 0.20` | scene ambient와 background 밝기를 랜덤화합니다. |
| Gazebo mode | `--headless` on/off | Gazebo GUI/RViz 실행 여부를 결정합니다. |
| Color log | 기본 on, `--no-color-log` 또는 `NO_COLOR`로 off | trial별 랜덤화 로그를 색상/볼드로 구분해 출력합니다. |
| Trial timeout | 기본 `time-limit-s + 180 s` | `episode_summary.json` 생성 대기 제한 시간입니다. |
| Dataset version | `--dataset-version`, 기본 빈 문자열 | `data/ais_portoffset_randomization/{version}` 하위에 저장할 버전을 지정합니다. |
| Hugging Face upload | 기본 off, `--push-to-hub`로 on, `--no-push-to-hub`로 off | PortOffsetCollect policy에 `AIC_VISION_OFFSET_PUSH_TO_HUB`를 명시 전달합니다. 실험 중 의도치 않은 업로드를 막기 위해 runner 기본값은 off입니다. |
| Hugging Face target | `--vision-offset-repo-id`, `--vision-offset-hf-revision`, `--vision-offset-hf-path-in-repo`, `--hf-private` | 업로드할 HF dataset repo, revision, repo 내부 경로, private repo 생성 여부를 지정합니다. |
| Upload port filter | `--upload-on-port-type` 값 `sfp`, `sc`, 또는 빈 문자열 기본 빈 문자열 | 특정 포트 타입 trial에서만 업로드하도록 제한합니다. 빈 문자열이면 PortOffsetCollect가 포트 타입 제한 없이 판단합니다. |

---

## 4. convert_to_lerobot.py — raw → LeRobot 변환

`collect_lerobot_data*.py`가 raw 포맷으로 저장한 에피소드를 LeRobot 데이터셋 포맷으로 일괄 변환한다.

```bash
python3 convert_to_lerobot.py \
  --capture-dir /tmp/aic_episodes \
  --out-dir ~/data/lerobot \
  --repo-id aic-sejong-team/aic-dataset \
  --fps 10
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
