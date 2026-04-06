# 채점 테스트 & 평가 가이드

> 날짜: 2026-04-05
> 원본: `ws_aic/src/aic/docs/scoring_tests.md`

---

## 사전 준비

모든 터미널에서 공통 설정:
```bash
source ~/ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_ROUTER_CHECK_ATTEMPTS=-1
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true;...'
```

결과 저장 위치: `$AIC_RESULTS_DIR/scoring.yaml` (기본: `~/aic_results/`, 실행마다 덮어쓰므로 경로 구분 권장)

---

## Tier별 점수 범위 요약

| Tier | 항목 | 범위 |
|------|------|------|
| 1 | 모델 유효성 | 0–1 |
| 2 | 궤적 부드러움 | 0–5 |
| 2 | 태스크 소요시간 | 0–10 |
| 2 | 궤적 효율 | 0–5 |
| 2 | 삽입 힘 패널티 | 0 ~ −12 |
| 2 | 금지구역 접촉 패널티 | 0 ~ −24 |
| 3 | 케이블 삽입 | −10 / 0~40 / 60 |

---

## 7가지 테스트 예시

| 예시 | 정책 | Tier 1 | Tier 2 주요 결과 | Tier 3 |
|------|------|--------|-----------------|--------|
| 1 | 모델 없음 | **실패** | — | — |
| 2 | CheatCode (ground_truth:=true) | 통과 | 고 부드러움, 패널티 없음 | **60점** (삽입 성공) |
| 3 | WaveArm | 통과 | 부드러움 ○, 시간/효율 미부여 | **0점** |
| 4 | WallToucher | 통과 | 접촉 패널티 −24점 | **0점** |
| 5 | WallPresser | 통과 | 힘 패널티 −12점 (접촉 패널티도 발생 가능) | **0점** |
| 6 | GentleGiant (저속) | 통과 | Tier 2 미부여 (근접 미달) | **0점** |
| 7 | SpeedDemon (고속·고강성) | 통과 | 힘 패널티 −12점 | **0점** |

---

## 실행 명령 패턴

3개 터미널을 사용하는 공통 패턴:
```bash
# Terminal 0 — Zenoh 라우터
ros2 run rmw_zenoh_cpp rmw_zenohd

# Terminal 1 — 정책 노드
ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.<PolicyName>

# Terminal 2 — 시뮬레이션 + 엔진
AIC_RESULTS_DIR=~/aic_results/<name> \
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=<true/false> start_aic_engine:=true
```

---

## 금지구역 접촉 대상 모델

| 모델 | 포함 내용 |
|------|-----------|
| `enclosure` | 바닥, 코너 포스트, 천장 |
| `enclosure walls` | 투명 아크릴 패널 |
| `task_board` | 보드 및 장착된 모든 컴포넌트 |

> 케이블 모델은 별도 엔티티로 패널티 미적용.
