# 논문 요약: MV2MAE — Multi-View Video Masked Autoencoders

**저자:** Ketul Shah, Robert Crandall, Jie Xu, Peng Zhou, Marian George, Mayank Bansal, Rama Chellappa  
**소속:** Johns Hopkins University & Amazon  
**arXiv:** 2401.15900v1 (2024.01.29)  
**분야:** Self-supervised Video Representation Learning

---

## 1. 핵심 아이디어 (한 줄 요약)

> 멀티뷰 동기화 비디오에서 **다른 시점(viewpoint)의 영상으로 마스킹된 패치를 복원**하도록 학습시키면, 3D 기하 정보를 내재화한 시점 불변(view-invariant) 표현을 자기지도(self-supervised) 방식으로 얻을 수 있다.

---

## 2. 배경 및 문제 의식

- 멀티뷰 비디오(스포츠, 자율주행, 노인돌봄, 로봇조작 등)는 3D 구조 이해에 유리하지만, 기존 MAE 기반 방법들은 **단일 시점(single-view)** 위주로 설계되어 시점 변화에 취약
- 비디오의 배경처럼 **정적인(static) 영역**은 인접 프레임에서 단순 복사(copy-paste)로 복원 가능 → 의미 있는 표현 학습을 방해
- 기존 멀티뷰 자기지도 학습(ViewCLR 등)은 대조학습(contrastive) 방식으로 메모리 집약적이고 다단계 학습 필요

---

## 3. 제안 방법

### 3.1 전체 구조 (Figure 1)

```
[Source View 비디오] ──┐
                       ├─► Shared Encoder (ViT) ──► Self-View Decoder ──► Source View 복원
[Target View 비디오] ──┘                        │
                                                └─► Cross-View Decoder ──► Target View 복원
                                                    (Cross-Attention: source token이 K,V
                                                     target masked token이 Q)
```

- **Shared Encoder**: source/target view 각각의 visible 토큰을 동일한 ViT로 인코딩
- **Self-View Decoder**: 각 뷰를 스스로 복원 (기존 VideoMAE 방식)
- **Cross-View Decoder (핵심 기여)**: source view의 visible 토큰을 K, V로 삼고, target view의 masked 토큰을 Q로 삼아 cross-attention으로 target view 복원
  - 이 과정에서 모델이 두 시점 간 기하 관계를 학습하도록 강제

### 3.2 Motion-Weighted Reconstruction Loss (핵심 기여 2)

- 배경 같은 정적 영역은 시간적으로 인접한 프레임에서 복사만으로 복원 가능 → 학습에 무의미
- **프레임 차이(frame difference)**로 각 패치의 움직임 정도를 계산하여 **가중치** 부여:
  ```
  L = (1/ρN) × Σ w_i × |I_i - Î_i|²
  ```
  - w_i: i번째 패치의 모션 가중치 (소프트맥스로 정규화)
  - temperature 파라미터 τ로 정적/동적 영역 강조 정도 조절 (τ=60 최적)
- 추가 학습 파라미터 없이 단순 프레임 차이로 계산 → 가볍고 효과적

### 3.3 구현 세부사항

- **인코더**: ViT-S/16 (기본), ViT-T / ViT-B 로도 스케일업 가능
- **입력**: 16 RGB 프레임, stride=4, 해상도 128×128
- **토크나이징**: 시간 patch size 2, 공간 patch size 16×16 → 512 토큰
- **마스킹 비율**: ρ=0.7 (멀티뷰이므로 단일뷰 0.9보다 낮게 설정 — 뷰 간 추론을 위해 더 많은 정보 필요)
- **학습**: AdamW, 1600 epochs, 사전학습 후 분류 헤드 붙여 파인튜닝

---

## 4. 실험 결과

### 4.1 사전학습/파인튜닝 데이터셋
- **NTU RGB+D 60/120**: 대규모 멀티뷰 행동 인식 (3개 카메라, Kinect-v2)
- **ETRI**: 노인 일상생활 행동 인식 (8개 동기화 시점)

### 4.2 SOTA 비교 (파인튜닝)

| 데이터셋 | 벤치마크 | MV2MAE(Ours) | 기존 최고 비지도 |
|---|---|---|---|
| NTU-60 | xview | **95.9%** | 94.1% (ViewCLR) |
| NTU-60 | xsub | **90.0%** | 89.7% (ViewCLR) |
| NTU-120 | xsub | **85.3%** | 84.5% (ViewCLR) |
| ETRI | xsub | **96.5%** | 95.1% (ConViViT, 지도학습) |

→ 비지도 방법 중 전 벤치마크 SOTA, ETRI에서는 지도학습까지 능가

### 4.3 전이학습 (Transfer Learning)

NTU-60으로 사전학습 후 소규모 데이터셋에 파인튜닝:

| 데이터셋 | MV2MAE | 기존 최고 비지도 |
|---|---|---|
| NUCLA | **97.6%** | 89.1% (ViewCLR, +8.5%p) |
| PKU-MMD-II | **60.1%** | 57.3% (HaLP) |
| ROCOG-v2 | **89.0%** | 87.0% (Reddy et al.) |

→ 전이학습에서 특히 두드러진 성능 — 표현의 범용성 확인

### 4.4 Ablation Study 핵심 결과

- **Temperature τ=60**: 정적 영역 너무 강조(τ↑) 또는 동적 영역만 강조(τ↓)하면 모두 성능 하락, τ=60 최적
- **마스킹 비율 ρ=0.7**: 멀티뷰 특성상 단일뷰(0.9)보다 낮아야 뷰 간 기하 추론 가능
- **소스 뷰 수**: 소스 뷰가 많을수록 복원이 쉬워져 오히려 성능 하락 (1개가 최적)
- **뷰 간 거리**: 소스-타겟 뷰가 너무 멀면(View1↔View4) 성능 저하 → 인접 시점이 적합
- **모델 크기**: ViT-T(82.0) → ViT-S(83.4) → ViT-B(85.1)로 스케일업 시 일관된 향상

---

## 5. Conclusion

- **Cross-view reconstruction + Motion-weighted loss** 두 가지 심플한 기여로 멀티뷰 비디오 자기지도 학습 SOTA 달성
- 추가 모달리티(depth, pose) 없이 RGB만으로 지도학습 수준 또는 그 이상의 성능
- ViewCLR 대비 메모리 효율적이고 단일 단계 학습

---

## AIC 프로젝트와의 연관성

이 논문은 행동 인식(action recognition) 도메인이지만, AIC 프로젝트에서 참고할 수 있는 개념:

| 이 논문 | AIC 프로젝트 적용 가능성 |
|---|---|
| 멀티뷰 이미지 간 Cross-Attention으로 기하 정보 학습 | 3-view 카메라(left/center/right) 간 관계 학습에 활용 가능 |
| Motion-weighted loss (움직이는 영역 강조) | 로봇 그리퍼/케이블 끝 부분에 가중치 부여한 학습 손실 설계 |
| Self-supervised 사전학습 후 downstream 파인튜닝 | 대량 데이터로 시각 인코더 사전학습 후 삽입 정책 학습에 활용 |
