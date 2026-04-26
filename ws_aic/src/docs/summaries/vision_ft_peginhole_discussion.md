# Vision + F/T 센서 기반 Peg-in-Hole 논문 탐색 및 적용 방향 논의
**날짜:** 2026-04-24

---

## 1. 탐색한 논문 목록

### 핵심 논문
| 논문 | 저자 | 학회 | 링크 |
|---|---|---|---|
| Making Sense of Vision and Touch | Lee et al. (Stanford) | ICRA 2019 Best Paper | https://hf.co/papers/1810.10191 |
| Imitating Human Search Strategies for Assembly | Ehlers et al. | IROS 2018 | https://hf.co/papers/1809.04860 |
| Factory: Fast Contact for Robotic Assembly | Narang et al. (NVIDIA) | IROS 2022 | https://hf.co/papers/2205.03532 |

### 최신 논문 (2024~2025)
| 논문 | 특징 | 링크 |
|---|---|---|
| FORGE | Sim-to-real, force threshold, snap-fit/gear 조립 | https://hf.co/papers/2408.04587 |
| ImplicitRDP | Vision+Force 융합 diffusion policy, slow-fast learning | https://hf.co/papers/2512.10946 |
| ForceVLA | VLA + F/T 센서 + MoE | https://hf.co/papers/2505.22159 |
| ATG-MoE | RGB-D + proprioception + 언어 명령 | https://hf.co/papers/2603.19029 |
| VTAM | Visual + Tactile 융합, contact-rich 특화 | https://hf.co/papers/2603.23481 |

---

## 2. "Making Sense of Vision and Touch" 요약

### 개요
- **학회:** ICRA 2019, Best Paper Award
- **저자:** Michelle A. Lee, Yuke Zhu et al. (Fei-Fei Li, Animesh Garg, Jeannette Bohg 그룹)

### 센서 구성
| 센서 | 종류 |
|---|---|
| 시각 | RGB 카메라 (wrist-mounted) |
| 힘 | 6-DOF Wrist F/T 센서 (fx, fy, fz, tx, ty, tz) |
| 고유감각 | Robot joint states (position, velocity) |

> Tactile 센서가 아닌 **Force/Torque 센서** 사용

### 핵심 아이디어: Self-supervised Multimodal 표현 학습
레이블 없이 auxiliary prediction task들을 동시 학습하여 물리적으로 의미있는 표현(representation) 학습

**Auxiliary Tasks:**
| Task | 입력 | 예측 대상 |
|---|---|---|
| Contact prediction | Vision + Force | 접촉 여부 |
| End-effector pose prediction | Vision + Force | TCP 위치/자세 |
| Optical flow prediction | Vision | 다음 프레임 픽셀 이동 |
| Force prediction | Vision + proprioception | 다음 timestep force |

### 학습 구조
```
RGB Image  → Visual Encoder ─┐
                              ├→ Fused z → 여러 prediction heads
F/T Signal → Force Encoder  ─┘

→ encoder 학습 완료 후 freeze
→ 그 위에 SAC (RL) policy 학습
```

### 2단계 파이프라인
1. **Phase 1:** Self-supervised pretraining (auxiliary tasks로 encoder 학습)
2. **Phase 2:** SAC (Soft Actor-Critic) RL로 peg insertion policy 학습

---

## 3. Proprioceptive Data 설명

**Proprioception** = 자기 몸 상태를 느끼는 내부 감각 (시각 없이도 팔 위치를 아는 것)

로봇에서: 각 관절 encoder → joint position(θ), velocity(θ̇), effort(τ)

`sensor_msgs/JointState`로 제공되며, Forward Kinematics를 통해 end-effector 위치/자세 계산 가능

### 현재 환경에서의 상태
`Observation.msg` 기준:
- `joint_states.position` (7 joints) → **현재 recording 중** ✅
- `joint_states.velocity` → **미저장** ❌
- `joint_states.effort` → **미저장** ❌
- `wrist_wrench` (6-DOF F/T) → **현재 recording 중** ✅

---

## 4. 현재 환경 vs 논문 적용 가능성

### 환경 비교
| 항목 | 논문 | 현재 환경 |
|---|---|---|
| RGB 카메라 | 1개 | 3개 (left/center/right) ✅ |
| F/T 센서 | 6-DOF wrist | 6-DOF wrist_wrench ✅ |
| Joint states | position | position ✅ |
| 학습 방식 | Self-supervised → RL (SAC) | Imitation Learning (LeRobot) |
| 태스크 | peg insertion | cable insertion ✅ |

### 적용 가능성
- ✅ **Self-supervised Encoder 학습**: 현재 수집 데이터로 auxiliary task 구성 가능
- ❌ **RL (SAC) downstream**: 리얼 로봇에서 수천 번 탐색 필요, 현실적으로 어려움

### 현실적인 적용 방안
```
[Phase 1] Self-supervised Encoder 학습 (논문 방식)
  steps.jsonl 데이터로 Vision + Force encoder offline 학습
        ↓
[Phase 2] LeRobot IL 파이프라인 유지
  pretrain된 encoder를 ACT/Diffusion Policy backbone으로 교체
```

---

## 5. Self-supervised Learning 개념 정리

### Autoencoder와의 차이
| | Autoencoder | 이 논문 |
|---|---|---|
| 목표 | 입력을 재구성 | 다른 값을 예측 |
| Supervision | 입력 자체 | 다른 모달리티 or 미래 상태 |
| 철학 | "잘 압축하면 잘 복원" | "예측을 잘 하려면 물리를 이해해야 함" |

더 가까운 개념: **Multi-task Self-supervised Learning** (CLIP의 철학과 유사)

### 핵심 정의
```
일반 supervised:   데이터 + 사람이 만든 라벨 → 학습
Self-supervised:   데이터 → 데이터 자체에서 라벨 자동 생성 → 학습
```

**이 논문에서 라벨 자동 생성 예시:**
| 예측 task | 라벨 출처 |
|---|---|
| 접촉 여부 | F/T 센서값 (Fz > threshold) |
| TCP 위치 | FK로 계산된 xyz |
| 다음 Force | 실제 다음 프레임 F/T값 |

→ 라벨은 수단, **특징 벡터(encoder)가 목적**

---

## 6. Baseline 방향 논의

| 방법 | 데이터 요구량 | LeRobot 지원 | 특징 |
|---|---|---|---|
| BC (Behavior Cloning) | 적음 | ✅ | 가장 단순, ablation 용이 |
| Diffusion Policy | ~100 에피소드 | ✅ | 요즘 manipulation baseline으로 많이 씀 |
| ACT | 500+ 에피소드 | ✅ | 강력하지만 데이터 많이 필요 |
| 현재 스크립트 정책 | 불필요 | - | 학습 없는 baseline |

**미결 사항:** 대회 vs 연구 목적에 따라 baseline 전략 달라짐
