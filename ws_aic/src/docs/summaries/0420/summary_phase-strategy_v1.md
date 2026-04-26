# AIC 단계별 전략 & 모델 설계 방향

> **요약**: Qual은 좁은 랜덤화 범위 내 Specialization 우선, Phase 1은 Flowstate 기반 완성도, Phase 2는 Sim2Real 일반화가 핵심. 각 단계별 요구 모델과 전략이 다르므로 단계적 접근이 필요.

---

## 1. 전체 타임라인

| 단계 | 기간 | 핵심 목표 | 평가 방식 |
|------|------|-----------|-----------|
| **Qualification** | ~2026-05-15 | Sim 내 삽입 성공 (Top 30) | Gazebo 자동 평가 |
| **Phase 1** | 2026-05-28 ~ 07-14 | Flowstate 기반 완성도 (Top 10) | Gazebo 자동 평가 |
| **Phase 2** | 2026-07-27 ~ 08-25 | 실제 로봇 배포 (Top 5) | 물리 로봇 평가 |

참고: [`overview.md`](../../ws_aic/src/aic/docs/overview.md)

---

## 2. Qualification — Specialization이 우선인가?

### 랜덤화 범위 (고정)

| 요소 | 랜덤화 범위 |
|------|------------|
| Task board pose | position + yaw 랜덤 |
| NIC rail 선택 | 5개 중 랜덤 (rail_0 ~ rail_4) |
| NIC 위치 (translation) | [0, 0.062] m |
| NIC 회전 (yaw) | [-10°, +10°] |
| SC port translation | [0, 0.115] m |
| 그리퍼 파지 편차 | ~2mm, ~0.04 rad |

참고: [`qualification_phase.md`](../../ws_aic/src/aic/docs/qualification_phase.md), [`glossary.md`](../../ws_aic/src/aic/docs/glossary.md)

### 결론: **Specialization 우선이 맞다**

Qual은 위 범위를 벗어나는 케이스가 없음. 즉:
- **"이 범위 내에서 항상 성공"** 을 목표로 최적화하는 것이 Top 30 진입 전략
- 지나친 일반화(unseen domain 대응)는 Qual 내에서 오히려 성능을 희석시킬 수 있음
- 단, 랜덤화 범위 내 **경계값(극단적 rail 위치, 최대 yaw 편차)** 에서도 성공해야 하므로 범위 내 커버리지는 필요

### Qual 채점 기준 요약 (최대 100점/trial × 3 trial)

| Tier | 항목 | 배점 |
|------|------|------|
| T1 | 모델 유효성 | 1 |
| T2 | Trajectory smoothness | 0~6 |
| T2 | Task duration | 0~12 (**5초 이내 = 만점**) |
| T2 | Trajectory efficiency | 0~6 |
| T2 | 과도한 삽입력 패널티 | 0~-12 (**20N 초과 1초 이상**) |
| T2 | 충돌 패널티 | 0~-24 |
| T3 | 삽입 성공 | 75 / 오삽입 -12 |
| T3 | 부분 삽입/근접도 | 0~50 |

참고: [`scoring.md`](../../ws_aic/src/aic/docs/scoring.md)

**핵심**: T3(삽입 성공 75점)이 압도적으로 큼 → **일단 삽입 성공이 최우선**, T2는 보조 지표.

---

## 3. Qual 모델 설계 방향

### 입력 관찰값
- 3개 손목 카메라 (left / center / right, 20 FPS, 1152×1024)
- 손목 F/T 센서 (ATI AXIA80-M20): force xyz, torque xyz
- `/tf`: 포트·플러그·그리퍼 pose (ground truth, 학습 시만 사용 가능)

### 권장 아키텍처

```
[이미지 × 3뷰] ──► [Backbone Encoder] ──┐
[F/T 센서]     ──────────────────────────┼──► [Policy Head] ──► ΔPose (Cartesian)
[TCP pose]     ──────────────────────────┘
```

| 선택지 | 장점 | 단점 |
|--------|------|------|
| **공유 인코더 (3뷰 동일 가중치)** | 파라미터 절약, 학습 데이터 3배 효과 | 뷰별 특성 차이 무시 |
| 뷰별 독립 인코더 | 뷰별 최적화 가능 | 파라미터 3배, 데이터 부족 위험 |
| 사전학습 백본 (DINOv2, R3M 등) | 적은 데이터로 강한 특징 추출 | fine-tuning 필요, 도메인 갭 |

→ EDA(Section 5-2) 결과에서 cross-view correlation이 높으면 **공유 인코더** 채택

### 출력 (Action)
- Cartesian-space `MotionUpdate`: Δx, Δy, Δz + orientation
- 또는 ACT (Action Chunking with Transformers): 제공된 baseline 참고

참고: [`aic_interfaces.md`](../../ws_aic/src/aic/docs/aic_interfaces.md), [`policy.md`](../../ws_aic/src/aic/docs/policy.md)

---

## 4. Phase 1 — Flowstate 기반 완성도

> *Coming Soon (문서 미공개)* — 현재 알려진 것만 정리

- **Intrinsic Flowstate** 접근 권한 부여 → 개발 환경 확장
- Qual 정책 기반으로 **완전한 케이블 핸들링 솔루션** 개발 (pick → route → insert)
- 평가: Gazebo 자동 평가 유지, 더 복잡한 시나리오 예상

### 예상 추가 요구사항
- 현재 Qual은 "이미 플러그를 쥔 상태"에서 시작 → Phase 1은 **pick & place까지 포함** 가능
- 다중 케이블 핸들링, 케이블 라우팅 등 추가 subtask 예상
- **Generalization 비중 증가**: 더 넓은 랜덤화 범위 또는 unseen 구성

### 모델 전략 전환 포인트
- Qual 모델의 인코더를 **재사용**, policy head만 확장
- 사전학습 백본의 중요성 증가 (데이터가 부족해질 경우)

---

## 5. Phase 2 — Sim2Real

> *Coming Soon (문서 미공개)* — 현재 알려진 것만 정리

- **물리 로봇(Intrinsic HQ)** 에서 실제 평가
- **Sim2Real gap**이 결정적 변수가 됨
- 동일 ROS 2 인터페이스 사용 → 코드 변경 최소화 목표

### Sim2Real 핵심 이슈

| 갭 요소 | Qual에서의 준비 방법 |
|---------|---------------------|
| 조명/색상 차이 | ColorJitter 증강 |
| 카메라 노이즈 | GaussianNoise 증강 |
| 케이블 물리 차이 | MuJoCo/IsaacLab 멀티 시뮬레이터 학습 |
| 그리퍼 편차 | PerturbCollect로 ±2mm 데이터 수집 |
| 실제 F/T 노이즈 | 시뮬 F/T에 noise augmentation |

공식 권장사항 (`qualification_phase.md`):
> "We actually **encourage you to train across different simulators**. These physical variations offer an excellent opportunity for domain randomization."

멀티 시뮬레이터: **Gazebo + IsaacLab (NVIDIA) + MuJoCo (Google DeepMind)**

---

## 6. 단계별 전략 요약

```
Qual (현재)
├── 목표: 범위 내 삽입 성공률 극대화 (Specialization)
├── 모델: 공유 인코더 + Cartesian policy head
├── 데이터: Gazebo 수집 + PerturbCollect (±10mm XY 다양화)
└── 증강: ColorJitter, RandomAffine (NIC/SC 갭 & 위치 편향 해소)

Phase 1 (예상)
├── 목표: 완전한 케이블 핸들링 솔루션
├── 모델: Qual 인코더 재사용 + 확장된 policy head
└── 데이터: 더 다양한 시나리오 + Flowstate 활용

Phase 2 (예상)
├── 목표: Sim2Real 성공
├── 모델: 멀티 시뮬레이터 학습 + domain randomization
└── 핵심: 실제 조명/물리 갭 극복
```

---

## 7. 지금 당장 해야 할 것 (Qual 기준)

| 우선순위 | 작업 |
|----------|------|
| 🔴 | 삽입 성공률 확보 (T3 75점) |
| 🔴 | NIC/SC 배치 1:1, approach 오버샘플링 |
| 🟡 | ColorJitter로 NIC↔SC 도메인 갭 줄이기 |
| 🟡 | PerturbCollect로 XY 정렬 오차 다양화 |
| 🟢 | T2 점수 최적화 (속도/스무스니스) — 삽입 성공 후 |
| 🟢 | 멀티 시뮬레이터 학습 준비 (Phase 2 대비) |
