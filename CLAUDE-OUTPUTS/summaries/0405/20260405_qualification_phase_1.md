# Qualification Phase 기술 개요

> 날짜: 2026-04-05
> 원본: `ws_aic/src/aic/docs/qualification_phase.md`

---

## 개요

완전히 시뮬레이션(Gazebo)에서 진행. NVIDIA(IsaacLab)·Google DeepMind(MuJoCo) 미러 환경도 훈련에 활용 가능. 완벽한 물리 재현보다 **기능적 검증**에 초점.

| 접근법 | 내용 |
|--------|------|
| Signal over Precision | 태스크 올바른 수행 여부에 집중 |
| 튜닝된 환경 | 케이블 물리 최대한 근사한 Gazebo 환경 제공 |
| 도메인 랜덤화 | 여러 시뮬레이터 간 훈련으로 Sim-to-Sim-to-Real 준비 |

---

## Phase 제약 조건

| 항목 | 내용 |
|------|------|
| 태스크 | Trial당 케이블 1회 삽입 (플러그 한쪽만) |
| 커넥터 타입 | `SFP_MODULE`→`SFP_PORT` / `SC_PLUG`→`SC_PORT` (미공개 타입 없음) |
| 환경 | Gazebo (Flowstate 없음) |
| 시작 상태 | 로봇이 플러그 파지 상태로 시작, 타겟에서 수 cm 이내 |
| 파지 편차 | ~2mm, ~0.04 rad 편차 가능 → 로버스트 정책 권장 |
| 랜덤화 | 태스크보드 위치·방향, 컴포넌트 레일 위치 매 Trial마다 랜덤 |

---

## 3개 Trial 구성

| | Trial 1 & 2 | Trial 3 |
|--|-------------|---------|
| **목적** | 정책 수렴 + NIC 랜덤 위치 대응 | 플러그/포트 타입 일반화 |
| **파지 플러그** | SFP_MODULE | SC_PLUG |
| **타겟 포트** | SFP_PORT_0 또는 SFP_PORT_1 (NIC 카드) | SC_PORT_0 또는 SC_PORT_1 (SC 레일) |
| **랜덤 요소** | 보드 포즈, NIC 레일 배정, NIC 오프셋 | 보드 포즈, SC 포트 레일 이동량 |
| **두 Trial 차이** | 랜덤값만 다름 | — |

---

*점수 체계: `scoring.md` / 다음 단계: `phases.md`*
