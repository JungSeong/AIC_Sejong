# Task Board 랜덤화 구조 및 사용 가이드

## 1. 랜덤화 파라미터 범위 (`LIMITS`)

`generate_scenario.py`의 `LIMITS` 딕셔너리에 모든 범위가 정의되어 있다.

| 파라미터 | 범위 | 단위 | 항상 랜덤? |
|---|---|---|---|
| `nic_translation` | -0.0215 ~ 0.0234 | m | ✅ 항상 |
| `nic_yaw` | ±10° (±0.1745 rad) | rad | ✅ 항상 |
| `sc_translation` | -0.06 ~ 0.055 | m | ✅ 항상 |
| `mount_translation` | ±0.09425 | m | ✅ 항상 |
| `board_yaw` | 0 ~ π (0 ~ 180°) | rad | ✅ 항상 |
| `board_x` (Trial 1/2) | 0.13 ~ 0.17 | m | `--diversify` 시만 |
| `board_y` (Trial 1/2) | -0.25 ~ -0.15 | m | `--diversify` 시만 |
| `board_x` (Trial 3) | 0.15 ~ 0.19 | m | `--diversify` 시만 |
| `board_y` (Trial 3) | -0.05 ~ 0.05 | m | `--diversify` 시만 |

> `mount_yaw` 는 LIMITS에 정의되어 있지만 현재 코드에서 사용되지 않는다 (항상 0.0).

---

## 2. Trial별 구성

### Trial 1/2 — NIC 카드 삽입 (SFP 플러그)
- NIC rail: trial 1 → rail 0, trial 2 → rail 1 에 NIC 카드 배치 (나머지 absent)
- SC port 0: 배경용으로 배치 (삽입 대상 아님)
- mount rail 0: lc/sfp/sc 모두 present / mount rail 1: lc만 present
- 케이블: `sfp_sc_cable`
- 보드 기본 위치: x=0.15, y=-0.2

### Trial 3 — SC 케이블 삽입
- NIC rail: 전부 absent
- SC rail 0/1 중 active rail에 SC 마운트 배치
- mount rail 0: sfp/sc present / mount rail 1: lc present
- 케이블: `sfp_sc_cable_reversed`
- 보드 기본 위치: x=0.17, y=0.0

---

## 3. 두 스크립트의 역할

| 스크립트 | 역할 |
|---|---|
| `generate_scenario.py` | 단일 trial 1개 생성. 명령어 출력 또는 직접 실행. 개발/디버그용 |
| `collect_data_native.py` | 세트(7 trial)×N 세트 자동 반복 수집. NIC×5 + SC×2 |

---

## 4. generate_scenario.py 사용법

```bash
# Trial 1 파라미터 출력 (실행 X)
python3 generate_scenario.py 1

# Trial 2 실제 실행
python3 generate_scenario.py 2 --run

# Trial 3, 시드 고정 (재현 가능)
python3 generate_scenario.py 3 --seed 42

# 보드 위치/yaw 모두 랜덤화 (--diversify)
python3 generate_scenario.py 1 --diversify

# pixi 소스 빌드 환경 사용
python3 generate_scenario.py 1 --mode pixi

# 특정 파라미터만 고정 (나머지는 랜덤)
python3 generate_scenario.py 1 --set nic_card_mount_0_yaw=0.0 task_board_x=0.15
```

---

## 5. collect_data_native.py 사용법

```bash
# 기본: 10 세트 × 7 에피소드
python3 collect_data_native.py

# 50 세트, 보드 위치/yaw 모두 랜덤화
python3 collect_data_native.py --sets 50 --diversify

# 명령어만 출력 (실제 실행 X)
python3 collect_data_native.py --sets 5 --dry-run

# 시드 고정 (단일, 재현용)
python3 collect_data_native.py --seed 42

# 협업 모드: 팀원별로 겹치지 않는 시드 범위 할당
python3 collect_data_native.py --seed-start 0   --seed-end 99    # A: 100 세트
python3 collect_data_native.py --seed-start 100 --seed-end 199   # B: 100 세트
python3 collect_data_native.py --seed-start 200 --seed-end 299   # C: 100 세트

# Gazebo GUI 없이 백그라운드 실행 (서버 환경)
python3 collect_data_native.py --headless

# Gazebo 초기화 대기 시간 조정 (기본 60초)
python3 collect_data_native.py --gazebo-wait 90

# HuggingFace Hub 업로드 (세트 완료 후 즉시)
python3 collect_data_native.py --hub-repo-id aic-sejong-team/aic-dataset
```

---

## 6. --diversify 플래그 정리

| | `--diversify` 없음 (기본) | `--diversify` 있음 |
|---|---|---|
| 보드 x, y | 고정값 (trial별 기본값) | 범위 내 랜덤 |
| 보드 yaw | 0 ~ π 랜덤 | 0 ~ π 랜덤 |
| NIC translation | 항상 랜덤 | 항상 랜덤 |
| NIC yaw | 항상 랜덤 | 항상 랜덤 |
| SC translation | 항상 랜덤 | 항상 랜덤 |
| mount translation | 항상 랜덤 | 항상 랜덤 |

평가 환경과 유사한 조건으로 수집할 때는 `--diversify` 없이,
도메인 다양화(domain randomization) 목적이라면 `--diversify`를 붙인다.

---

## 7. 파라미터 범위 수정

`generate_scenario.py` 상단의 `LIMITS` 딕셔너리를 직접 수정하면 된다.
`collect_data_native.py`는 이 딕셔너리를 import해서 사용하므로 한 곳만 수정하면 두 스크립트 모두 반영된다.

```python
LIMITS = {
    "nic_translation":   (-0.0215, 0.0234),   # 이 값을 수정
    "nic_yaw":           (-math.radians(10), math.radians(10)),
    ...
}
```
