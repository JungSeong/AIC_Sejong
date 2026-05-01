# ais_auto_capture — 자동 데이터 수집 스크립트

## 스크립트 구성

| 파일 | 환경 | 역할 |
|---|---|---|
| `collect_data.py` | distrobox (x86) | LeRobot 에피소드 자동 수집 |
| `collect_data_aarch.py` | 소스 빌드 (aarch64) | LeRobot 에피소드 자동 수집 |
| `collect_yolo_data_aarch.py` | 소스 빌드 (aarch64) | YOLO 데이터셋 자동 수집 |
| `convert_to_lerobot.py` | 공통 | raw 에피소드 → LeRobot 포맷 변환 |

---

## 1. collect_data_aarch.py — LeRobot 에피소드 수집 (aarch64)

세트당 7개 trial(NIC×5 + SC×2)을 Gazebo에서 자동 실행하며 LeRobot 포맷으로 저장한다.

**흐름**
1. trial별 랜덤 파라미터로 aic_engine config YAML 생성
2. Zenoh 라우터 → Gazebo(`aic_gz_bringup`) → `DataCollect` policy 순으로 시작
3. `episode_summary.json` 파일 수로 완료 감지
4. Gazebo 종료 → 다음 세트 반복

**사용법**
```bash
# 기본: 10 세트 × 7 에피소드
python3 collect_data_aarch.py

# 50 세트, 보드 위치/yaw 랜덤화
python3 collect_data_aarch.py --sets 50 --diversify

# 명령어만 출력 (실제 실행 X)
python3 collect_data_aarch.py --sets 5 --dry-run

# Gazebo GUI·RViz 없이 실행
python3 collect_data_aarch.py --headless

# LeRobot 로컬 저장 + HuggingFace 업로드
python3 collect_data_aarch.py \
  --lerobot-out-dir ~/data \
  --lerobot-repo-id aic-sejong-team/aic-dataset
```

---

## 2. collect_yolo_data_aarch.py — YOLO 데이터셋 수집 (aarch64)

시나리오별로 Gazebo를 별도 세션으로 실행하며 3대 카메라 스냅샷 + TF 기반 bbox 라벨을 자동 생성한다.

**흐름 (시나리오당)**
1. 랜덤 파라미터로 aic_engine config YAML 생성 + scenario_params JSON 저장
2. Zenoh 라우터 → Gazebo → `DataCollect` policy 시작
   - `DataCollect` policy를 사용해야 Task Board가 실제로 spawn됨
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

## 3. convert_to_lerobot.py — raw → LeRobot 변환

`collect_data*.py`가 raw 포맷으로 저장한 에피소드를 LeRobot 데이터셋 포맷으로 일괄 변환한다.

```bash
python3 convert_to_lerobot.py \
  --capture-dir /tmp/aic_episodes \
  --out-dir ~/data/lerobot \
  --repo-id aic-sejong-team/aic-dataset \
  --fps 10
```

---

## 4. 랜덤화 파라미터 범위 (`LIMITS`)

| 파라미터 | 범위 | 단위 | 항상 랜덤? |
|---|---|---|---|
| `nic_translation` | -0.0215 ~ 0.0234 | m | 항상 |
| `nic_yaw` | ±10° | rad | 항상 |
| `sc_translation` | -0.06 ~ 0.055 | m | 항상 |
| `mount_translation` | ±0.09425 | m | 항상 |
| `board_yaw` | 0 ~ π | rad | 항상 |
| `board_x` (NIC) | 0.13 ~ 0.17 | m | `--diversify` 시만 |
| `board_y` (NIC) | -0.25 ~ -0.15 | m | `--diversify` 시만 |
| `board_x` (SC) | 0.15 ~ 0.19 | m | `--diversify` 시만 |
| `board_y` (SC) | -0.05 ~ 0.05 | m | `--diversify` 시만 |

---

## 5. collect_data.py — LeRobot 에피소드 수집 (distrobox / x86)

`collect_data_aarch.py`와 동일한 역할이지만 distrobox 컨테이너 환경에서 동작한다.
aarch64 환경에서는 `collect_data_aarch.py`를 사용할 것.
