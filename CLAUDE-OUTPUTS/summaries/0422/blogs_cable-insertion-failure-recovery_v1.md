# 기술 블로그 조사: 케이블/커넥터 삽입 실패 복구

> 조사일: 2026-04-22

---

## 1. Multi-Stage Cable Routing — 공식 프로젝트 페이지

> 출처: [cablerouting project site](https://sites.google.com/view/cablerouting/home) — Luo et al. (UC Berkeley / Google), 2023

**요약**: IEEE TRO 2024에 게재된 논문의 공식 페이지. 데모 영상, 데이터셋, 코드 링크 포함.
Franka Panda + Basler 카메라(end-effector 장착) + RealSense(워크셀 고정) 조합으로
케이블 클립 삽입 및 실패 복구를 시연.

### 핵심 인사이트
- 고수준 정책이 **실패 감지 후 동일 primitive를 자동 재시도**하거나 "케이블 당김(tighten)" primitive를 선택
- 단순 시퀀싱 대비: 각 단계 실패 확률이 곱해지므로 5단계 태스크에서 성공률 급락 → 계층 구조 필수
- 카메라 배치 전략이 중요: end-effector 카메라(근접 뷰) + 고정 카메라(전체 뷰)를 병행

### AIC 적용 가능성
> 커넥터 삽입 실패 시 end-effector 카메라로 케이블 끝 위치를 재확인 후 재정렬 primitive를 실행하는 방식에 직접 참고 가능

---

## 2. Transformers for Early Failure Prediction (Correll Lab, 2023)

> 출처: [Early failure prediction using Transformers](https://www.colorado.edu/lab/correll/2023/07/10/early-failure-prediction-during-robotic-assembly-using-transformers) — University of Colorado Boulder

**요약**: dilated CNN으로 실패를 예측하는 Watson & Correll (2023) 논문의 후속 연구.
Transformer가 dilated CNN보다 **3배 더 짧은 시계열**로 실패를 예측할 수 있으며,
preemptive restart를 적용하면 makespan을 **최대 40% 단축** 가능.

### 핵심 인사이트
- Transformer의 장점: 더 이른 시점에 실패를 예측 → 불필요한 삽입 시도 시간 단축
- Transformer의 단점: 정확도가 dilated CNN보다 소폭 낮음 (정밀도 vs 속도 트레이드오프)
- 데이터셋: 241회 Peg-in-hole 시도에서 수집한 F/T 시계열

| 모델 | 예측 속도 | 정확도 | Makespan 감소 |
|------|---------|--------|-------------|
| Dilated CNN | 기준 | 높음 | ~20% |
| **Transformer** | **3배 빠름** | 소폭 낮음 | **~40%** |

### AIC 적용 가능성
> F/T 센서 데이터를 Transformer로 실시간 처리하면, 삽입 시도 초반에 실패를 예측하고 빠르게 재시도할 수 있어 경쟁 환경에서 전체 작업 시간을 줄이는 데 직접 활용 가능

---

## 3. Dual-Arm Peg-in-Hole with DNN & 자동 복구 (MDPI, 2021)

> 출처: [Dual-Arm Peg-in-Hole Using DNN with Double F/T Sensor](https://www.mdpi.com/2076-3417/11/15/6970) — MDPI Applied Sciences

**요약**: 삽입 힘이 임계값을 초과하면 두 번째 DNN이 peg를 중앙으로 되돌리는 조정 사이클 실행.
각 단계를 독립적으로 수정 가능한 이산 이벤트 시스템 설계로 유지보수 편의성 확보.

### 핵심 인사이트
- 실패 감지를 **힘 임계값 기반**으로 단순화 가능 (학습 불필요)
- 복구 DNN은 별도 학습: "현재 위치 오차 → 보정 동작" 매핑
- 이산 이벤트 시스템: 복구 로직만 따로 교체 가능 → 모듈화에 유리

### AIC 적용 가능성
> 삽입 실패 시 힘 임계값 트리거로 복구 모듈을 활성화하는 단순하고 신뢰성 높은 baseline으로 활용 가능

---

## 종합 정리

| 출처 | 핵심 아이디어 | AIC 적용 가능성 |
|------|------------|--------------|
| Cable Routing 프로젝트 페이지 | 계층 구조 + 자동 재시도 primitive | 삽입 실패 → 재정렬 → 재시도 설계 |
| Correll Lab Transformer | F/T 시계열 + Transformer → 40% makespan 감소 | 실시간 실패 예측 모듈 |
| MDPI Dual-Arm | 힘 임계값 + 복구 DNN | 간단한 baseline 복구 로직 |
