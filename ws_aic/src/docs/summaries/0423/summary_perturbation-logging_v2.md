# Perturbation 및 Ground Truth Offset 기록 로직 개선 요약

> 날짜: 2026-04-23
> 작성자: Gemini CLI
> 관련 코드: `code/collect_data.py`, `perturbcollect/policy.py`

---

## 1. 개요
시나리오 생성 시점에 랜덤하게 결정되는 **그라운드 트루스 파지 오프셋(Ground Truth Gripper Offset)** 값을 데이터셋에 직접 포함시키기 위해, 생성 스크립트와 정책 노드 간의 데이터 전달 로직을 구축함.

---

## 2. 상세 수정 사항

### A. 시나리오 생성부 (`code/collect_data.py`)
- `_make_nic_trial`, `_make_sc_trial` 함수에서 파지 오차($\pm 2mm$)를 랜덤 샘플링하도록 로직 복구 및 강화.
- 샘플링된 값을 `os.environ` 환경변수로 내보내 정책 노드에 전달.

### B. 정책 기록부 (`perturbcollect/policy.py`)
- 환경변수에서 `ground_truth_gripper_offset`을 읽어와 `meta.json` 및 `episode_summary.json`에 기록.
- TF 기반의 측정값(`initial_gripper_offset`)과 대조 분석이 가능하도록 구조화.

---

## 3. 분석 가치
- **오차 대조**: 시뮬레이션에 주입된 실제 오프셋과 로봇이 TF를 통해 인식하는 오프셋 사이의 일치 여부 확인.
- **강건성 지표**: $2mm$ 이상의 파지 오차가 발생했을 때, $10mm$ 범위의 제어 교란(Perturbation)이 삽입 성공률에 미치는 복합적 영향 분석.

---

## 4. 최종 확인 사항
- 모든 수정 사항에 대해 주석을 추가하여 수정 이유(Robustness 분석) 명시.
- `git diff`를 통해 생성-전달-기록 프로세스 연결 확인 완료.
