# Perturbation 및 Gripper Offset 기록 로직 수정 요약

> 날짜: 2026-04-23
> 작성자: Gemini CLI
> 관련 코드: `perturbcollect/policy.py`, `cheatcode.py`

---

## 1. 개요
데이터 수집 시 모델의 강건성(Robustness) 분석을 위해 **교란(Perturbation)** 값과 **파지 편차(Gripper Offset)** 정보를 `meta.json`, `episode_summary.json`, `steps.jsonl`에 기록하도록 코드를 수정함.

---

## 2. 주요 수정 사항

| 파일 경로 | 수정 내용 | 기록 위치 |
|-----------|-----------|-----------|
| `perturbcollect/policy.py` | 에피소드 시작 시 초기 파지 편차(`initial_gripper_offset`) 계산 및 기록 | `meta.json`, `episode_summary.json` |
| `perturbcollect/policy.py` | 샘플링된 XY 교란 값(`perturb_dx_m`, `perturb_dy_m`) 명시적 기록 | `meta.json`, `episode_summary.json` |
| `cheatcode.py` | 매 스텝마다 그리퍼-플러그 간 상대 포즈(`gripper_offset`) 계산 및 기록 | `steps.jsonl` (extras) |

---

## 3. Perturbation 정의 및 목적
- **정의**: `insert` 페이즈 진입 시 목표 좌표에 의도적으로 주입하는 무작위 XY 오프셋 (최대 10mm).
- **목적**: 
    - 파지 편차(~2mm) 및 제어 오차에 대응하는 로버스트한 정책 학습.
    - F/T 센서 피드백을 통한 오차 극복 데이터 확보.

---

## 4. 분석 인사이트 (예상)
- 에피소드별 초기 `gripper_offset` 분포 확인을 통해 시뮬레이션의 도메인 랜덤화 수준 파악 가능.
- `perturbation` 크기에 따른 삽입 성공률 상관관계 분석을 통해 정책의 한계 성능 도출 가능.
