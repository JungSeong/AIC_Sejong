# 논문 요약: Vision-force-fused Curriculum Learning for Robotic Contact-rich Assembly Tasks

**저자:** Piaopiao Jin, Yinjie Lin, Yaoxian Song, Tiefeng Li, Wei Yang (Zhejiang University)  
**게재지:** Frontiers in Neurobotics, 2023. DOI: 10.3389/fnbot.2023.1280773  
**키워드:** contact-rich manipulation, multimodal perception, sensor fusion, curriculum learning, robotic assembly task

---

## 1. Introduction (서론)

### 핵심 문제
로봇 조립 작업(assembly)은 정밀한 지각, 정교한 제어, 지능적 의사결정을 동시에 필요로 하는 **contact-rich** 작업이다. 기존 연구는 비전(vision)이나 힘(force) 중 하나만 활용하는 단일 모달 방식에 치중해 있었으며, 두 센서를 효과적으로 융합하는 통합 메커니즘이 부재했다.

### 기존 방식의 한계
- **센서 기반 컨트롤러 통합 방식**: 비전 서보 제어와 힘 제어를 분리 설계 후 결합 → 접촉 중 발생하는 힘이 목표 포즈 추정에 도움이 된다는 사실을 무시함
- **데이터 기반 방식**: 학습 기반 접근이 부상하고 있으나 단일 모달리티의 한계를 벗어나지 못함

### 제안 방식 요약
- 비전+힘 특징을 동시에 추출·융합하는 **멀티모달 인식 프레임워크** 제안
- 작업 난이도에 따라 단계적으로 멀티모달 특징을 활용하는 **Curriculum Learning (CL)** 도입
- 시뮬레이션 → 실물 로봇 **제로샷(zero-shot) 전이** 달성
- 주요 성과: 0.1 mm 클리어런스 조건에서 **95.2% 삽입 성공률**, 미학습 형상에 대한 일반화

---

## 2. Related Work (관련 연구)

### 2.1 조립 작업에서의 힘·비전 인식
- **힘만 사용**: 힘/토크 과도 응답으로 peg-hole 상대 위치 추정 가능하나, 사전 형상 지식 필요 및 미학습 형상에 취약
- **비전만 사용**: RGB-D 카메라로 삽입 위치 탐색 가능하나, 접촉 중 발생하는 힘 정보를 활용하지 못함
- **비전+힘 결합**: Visual servoing + 임피던스 제어 결합 시도들이 있으나, 각 센서를 개별 컨트롤러에 분리 적용하는 방식이라 상보성(complementarity)을 충분히 활용하지 못함

### 2.2 강화학습 기반 조작
- RL은 환경 변화에 대한 적응성 면에서 유망하나, **샘플 비효율성**과 **실물 배포 어려움**이 단점
- **Curriculum Learning (CL)**: 난이도를 점진적으로 높여가며 학습 → 데이터 효율성 향상 및 학습 가능성 확대
- 모델 기반(model-based) 방법이 광범위 상호작용 없이 학습을 가능하게 하는 대안으로 연구됨

---

## 3. Problem Statement (문제 정의)

- **태스크**: 직사각형 peg를 대응하는 hole에 삽입. 클리어런스 최대 0.1 mm, 삽입 깊이 10 mm
- **입력**: 비전 관측값 $x_v$ (RGB 이미지), 힘 관측값 $x_f$ (F/T 센서)
- **출력**: 증분 모션 벡터 $\Delta X \in \mathbb{R}^4$ = $[\Delta x, \Delta y, \Delta z, \Delta\theta]$
- **수식**:
  - $X_{target} = X_{cur} + \Delta X$
  - $\Delta X = f(x_v, x_f)$ (비전·힘 원시 데이터 → 모션 벡터 매핑)
- **핵심 어려움**: 접촉 전·후 비전과 힘 데이터의 특성이 크게 달라 단일 정책 함수로 통합하기가 어려움 → 모달리티별 인코더로 해결

---

## 4. Method (제안 방법)

### 전체 파이프라인
```
[RGB Images] → [Visual Encoder E_vision] → φ_v (128-dim)
[F/T Data]   → [Force Encoder E_force]  → φ_f (30-dim)
                                            ↓
                              φ_v ⊕ φ_f → [MLP Policy π_mlp] → ΔX
```

### 4.1 Vision-force Feature Fusion (비전-힘 특징 융합)

**힘 인코더 (E_force)**
- 최근 5 프레임의 F/T 데이터를 Experience Replay + Sliding Window로 집계
- 집계된 신호를 평탄화(flatten) → **30차원 힘 특징** $\phi_f$
- 정규화: 평균($\mu_f$)과 분산($\sigma^2$)으로 정규화 후 tanh 적용 → [-1, 1] 범위

**비전 인코더 (E_vision) — Self-supervised 방식**
- 목표: peg와 hole의 공간적 관계(Ex, Ey, Ez, Eθ)를 이미지에서 추출
- 구조: Front/Rear RGB 이미지 각각 → ResNet50 backbone → 128-dim 특징 → 3-layer MLP Predictor
- 출력: Ex, Ey, Eθ의 부호(양/음) 예측 (Ez는 깊이 정보 손실로 관측 불가)
- 학습 데이터: MuJoCo 시뮬레이션에서 합성 이미지 6만장 자동 생성 (라벨 자동 부여)
- Domain Randomization: Gaussian blur, white noise, random shadow, random crop + 색상 랜덤화

### 4.2 Curriculum Policy Learning (커리큘럼 정책 학습)

**2단계 학습 구조**

| 단계 | 클리어런스 | 관측 공간 | 역할 |
|---|---|---|---|
| Stage 1 (Easy) | 0.5 mm (쉬움) | 128-dim ($\phi_v$ only) | 비전 기반 대략적 정렬 학습 |
| Stage 2 (Hard) | 0.1 mm (어려움) | 158-dim ($\phi_v \oplus \phi_f$) | 힘 피드백으로 정밀 삽입 파인튜닝 |

- Stage 1에서 학습한 비전 정책 $\phi_{init\_mlp}$를 Stage 2의 초기값으로 활용
- 강화학습 알고리즘: **PPO (Proximal Policy Optimization)**
- **보상 설계 (Sparse)**:
  - peg가 hole에 반쯤 삽입되면 +0.5
  - peg가 hole에 완전 삽입되면 +0.5 (총 +1.0)
  - peg가 그리퍼에서 떨어지면 -0.2 (패널티)

**모션 실행**
- 출력 $\Delta X$는 Cartesian Motion/Force Controller (Lin et al., 2022)가 실행
- z축 방향 접촉력은 명령으로 지정, x·y·roll은 모션 벡터로 제어

### 4.3 Implementation Details (구현 세부사항)

- **비전 인코더 학습**: Adam optimizer, 20 epochs, batch size 32, lr=1e-4, PyTorch 1.11
- **정책 학습 환경**: MuJoCo 시뮬레이터
- **초기 포즈 랜덤화**: x·y 각 ±10 mm, z 5~20 mm, 회전 ±10°
- **PPO 하이퍼파라미터**: n_steps=64, batch_size=32, gae_lambda=0.998 (stable-baselines3)

---

## 5. Experiment (실험)

### 5.1 평가 기준
- **성공 기준**: peg를 hole에 10 mm 이상 삽입 완료
- **실패 기준**: 삽입 도중 그리퍼에서 peg가 떨어지는 경우

### 5.2 시뮬레이션 결과 분석 (RQ1, RQ2)

**RQ1: 기존 방법 대비 성능 비교 (Table 1)**

| 모델 | 클리어런스 | 성공률 | 형상 일반화 | 인간 시연 필요 |
|---|---|---|---|---|
| Gao & Tedrake (2021) | 0.2 mm | 74% | ✗ | ✗ |
| Lee et al. (2020b) | 2 mm | 78% | ✓ | ✗ |
| Spector et al. (2022) | – | 97.5% | ✗ | ✓ |
| **Ours** | **0.1 mm** | **95.2%** | **✓** | **✗** |

- 클리어런스 0.1 mm는 기존 최고 난이도(0.2 mm) 대비 2배 더 정밀
- 인간 시연 없이 97.5%에 근접한 성능 달성

**RQ2: 미학습 형상 및 초기 위치에 대한 일반화**
- 초기 위치 오차 ±3 cm 범위에서 전체 평균 **95.2%** 성공
- 위치 오차 1.5 cm 이내에서는 거의 100% 달성
- 미학습 형상 성공률:
  - Square (학습): 95.3%
  - Pentagon: 86%
  - Triangle: 68%
  - Circle: 60% (선 접촉으로 슬립 발생 → 상대적으로 낮음)

### 5.3 모듈 기여도 Ablation Study (RQ3)

비교 대상 3가지 모델:
1. **Vision-only CL**: 비전만 사용, CL 학습
2. **Vision-force CL (Ours)**: 비전+힘 융합, CL 학습
3. **Naive RL**: 비전+힘 사용, CL 없이 naive RL 학습

결과 (Figure 6):
- **Naive RL**: 성공률 0% — 0.1 mm 클리어런스에서 naive RL은 학습 자체가 불가능
- **Vision-only CL**: 성공률 70% — CL만으로도 어느 정도 작동 가능
- **Vision-force CL**: 성공률 95.2% — 힘 융합으로 **25%p 향상**

핵심 인사이트:
- CL이 없으면 (Naive RL) 극도로 정밀한 작업에서 수렴 실패
- 힘 정보는 접촉 발생 시 보완적 역할을 수행 (비전으로 방향 탐색, 힘으로 접촉 정밀 제어)

### 5.4 실물 로봇 실험 (RQ4)

- **시뮬레이션 → 실물 제로샷 전이** 성공
- 실물 실험 클리어런스: Square 0.37 mm, Pentagon 0.44 mm, Triangle 1 mm, Circle 0.41 mm

**Table 2: 실물 성공 횟수 (10회 시도)**

| 모델 | Square | Pentagon | Triangle | Circle |
|---|---|---|---|---|
| Vision-only CL | 3/10 | 8/10 | 3/10 | 2/10 |
| **Vision-force CL** | **6/10** | **9/10** | **5/10** | **4/10** |

- 전 형상에서 Vision-force CL이 Vision-only CL 대비 일관되게 우수
- 실물에서도 시뮬레이션과 동일한 성능 격차 유지 → 도메인 랜덤화 효과 검증

---

## 6. Conclusion (결론)

- **비전-힘 융합 Curriculum Learning** 프레임워크를 제안하여 contact-rich 조립 작업에서 높은 정밀도와 일반화 능력을 동시에 달성
- 0.1 mm 클리어런스에서 95.2% 성공, 인간 시연 불필요, 미학습 형상 일반화, 실물 제로샷 전이 모두 달성
- **비전**: 전역적 방향 탐색 (hole 위치 파악)
- **힘**: 접촉 발생 시 보완적 정밀 제어
- **CL**: 쉬운 태스크 → 어려운 태스크 순차 학습으로 극정밀 작업 학습 가능하게 함

---

## 우리 프로젝트와의 연관성

| 이 논문 | AIC 프로젝트 (우리) |
|---|---|
| Peg-in-hole, 0.1 mm clearance | 케이블(SFP/SC) 삽입, 포트에 정밀 삽입 |
| 비전(RGB) + F/T 센서 융합 | 3-view 카메라 + controller_state (tcp_error 포함) |
| Curriculum Learning (쉬운 환경 → 어려운 환경) | 현재 단일 난이도 ACT 모델 사용 |
| Self-supervised visual encoder | YOLO 기반 포트 검출 |
| PPO 기반 RL 정책 | Behavior Cloning (ACT) 기반 모방 학습 |
| 시뮬레이션 → 실물 제로샷 전이 | Docker 컨테이너 → AIC 평가 환경 제출 |

**참고할 핵심 아이디어**: 현재 우리 방식(BC/ACT)에 force feature를 명시적으로 분리 인코딩하고, 쉬운 시나리오(큰 클리어런스 or 더 많은 perturbation 여유)부터 학습하는 Curriculum 방식을 도입하면 성능 향상 여지가 있음.
