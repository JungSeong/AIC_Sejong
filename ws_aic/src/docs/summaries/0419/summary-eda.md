# EDA Multiview Bias3 — 섹션별 인사이트 정리

> **요약**: `eda_multiview_bias3.ipynb`의 각 섹션에서 얻을 수 있는 인사이트와, 이를 데이터 정제 및 증강 전략으로 어떻게 연결하는지 정리한 문서.  
> 목적: 백본 인코더 학습 전 데이터의 편향·다양성을 파악하고, 적절한 전처리/증강 근거를 확보.

---

## Section 1-1 — 데이터 파싱

**얻는 것**: 전체 데이터셋의 기본 건강 상태.

| 확인 항목 | 정상 | 이상 신호 |
|-----------|------|-----------|
| 핵심 컬럼 결측(NaN) | 없음 | `tip_x_error`, `plug_x` 등 NaN 다수 → 세션 필터링 |
| 세션당 스텝 수 | 예상 범위 이내 | 너무 적음 → 도중 종료 에피소드 → 제거 대상 |

**증강/정제 연결**: 결측 세션을 먼저 제거하지 않으면 이후 모든 분석이 오염됨.

---

## Section 1-2 — Phase & Task 분포 불균형

**얻는 것**: insert/approach/stabilize 비율, NIC/SC 비율.

| 상황 | 의미 | 대응 |
|------|------|------|
| insert 비율 > 60% | 인코더가 삽입 외관에만 특화 | approach 오버샘플링 |
| NIC : SC > 70:30 | 특정 태스크가 배치 지배 | 배치 내 **1:1 샘플링** 필수 |
| stabilize 비율 매우 낮음 | 무시해도 됨 | 별도 처리 불필요 |

**핵심 메시지**: 불균형한 phase/task 분포는 인코더가 "삽입 직전 자세"에만 최적화되어 approach 구간을 학습하지 못하는 원인이 됨.

---

## Section 2-1 — Tip Error 분포 (Output Space Bias)

**얻는 것**: approach/insert 각 위상의 오차 분포 및 수렴 궤적.

| 확인 항목 | 정상 | 이상 신호 |
|-----------|------|-----------|
| `|mean| vs std` | `|mean| < std × 0.3` (대칭 분포) | `|mean| > std × 0.3` → 특정 방향 치우침 → 삽입 시작 시 Gaussian XY offset 추가 |
| approach scatter | 퍼진 구름 형태 | 지나치게 좁음 → 시작 자세 다양성 부족 → 인코더 위치 암기 위험 |
| insert scatter | 원점 주변 집중 | 원점에서 멀리 퍼짐 → 삽입 실패 에피소드 포함 |

**비수렴 케이스(insert에서 tip_error가 줄지 않는 궤적)는 제거 대상이 아님.**  
어려운 시작 조건·접촉 실패 복구 상황 정보를 담고 있어 인코더 강건성에 기여.  
단, **완전 실패(time_limit까지 발산)** 는 학습 비중을 낮추거나 별도 플래그 처리.

---

## Section 2-2 — z_offset 궤적 & Force XYZ 분포

**얻는 것**: 삽입 깊이 다양성 + 3축 접촉력 분포.

| 축 | approach 기대 | insert 기대 | 이상 신호 |
|----|-------------|------------|----------|
| force_z | ≈ 0 | 넓은 분포 (접촉) | 항상 0 → F/T 센서 Tare 오류 |
| force_x/y | ≈ 0 | 소폭 분포 | 큰 값 → 심한 측면 충돌 |

**측면력 비율 판단**:
- `std(force_x/y) / std(force_z) > 0.5` → 정렬 실패 에피소드 다수 → 필터링 또는 증강 검토
- approach에서 force_x/y 크면 → 케이블 장력 문제

---

## Section 3-2 — 픽셀 분산 히트맵 (Input Spatial Bias)

**얻는 것**: 이미지 내 어느 픽셀이 실제로 "움직이는 정보"를 담고 있는가.

| spatial_conc 값 | 의미 | 대응 |
|-----------------|------|------|
| < 0.15 (BIASED) | 분산의 50%가 전체 픽셀의 15% 미만에 집중 | **RandomCrop + AffineTransform 필수** |
| 0.15 ~ 0.30 (caution) | 중간 수준 집중 | 약한 augmentation 권장 |
| > 0.30 (OK) | 분산이 이미지 전반에 고루 분포 | augmentation 선택적 |

**추가 관찰**:
- NIC vs SC의 **peak 좌표가 다름** → 인코더가 "이 위치면 NIC" shortcut 학습 가능 → Affine translate로 희석
- 하단 검은 영역(로봇 몸체) = 분산 ≈ 0 → **공통 crop margin 설정**으로 제거 검토

---

## Section 3-3 — 8×8 Grid 분산 (공간 활성 영역 정량화)

**얻는 것**: 어느 이미지 구역이 실제 정보를 담는가를 블록 단위로 요약.

| 활성 셀 수 (값 > 0.5, 64개 중) | 의미 |
|-------------------------------|------|
| 10개 미만 | 이미지의 85% 이상이 정적 배경 → **crop 또는 ROI masking 검토** |

- center 뷰가 가장 활성 셀이 많아야 정상
- left/right가 더 집중되어 있으면 카메라 배치 이슈 가능성

---

## Section 4-3 — 노이즈 / 텍스처 / 주파수 (Input Feature Bias)

**얻는 것**: 이미지의 선명도(laplacian), 노이즈(noise_std), 고주파 비율(fft_high), 엣지 밀도(edge_density).

| 발견 | 의미 | 증강 대응 |
|------|------|-----------|
| insert > approach (fft_high, edge_density) | 커넥터 근접 시 고주파 텍스처 증가 → **phase label leak 위험** | phase별 augmentation 강도 분리 |
| right/nic approach noise_std 높음 | 특정 뷰/태스크에 노이즈 집중 | 다른 뷰에 GaussianNoise 추가해 균일화 |
| laplacian_var 전반 600~800 (충분히 선명) | 이미지 품질 양호 | **blur 증강은 약하게** (kernel ≤ 5, p ≤ 0.2) |

---

## Section 4-4 — HSV 스칼라 특징 (Input Appearance Bias)

**얻는 것**: brightness, saturation, hue_entropy의 NIC/SC 간 도메인 갭.

| 피처 | NIC | SC | 위험 |
|------|-----|----|------|
| brightness | ~0.57 | ~0.47 | 인코더가 밝기로 태스크 구분 → shortcut |
| saturation | 낮음 | 2~3배 높음 | 채도로 태스크 구분 → shortcut |
| hue_entropy | ~0.80 | ~0.60 | 색 다양성으로 태스크 구분 → shortcut |

**증강 대응**:
- `ColorJitter(brightness, saturation, hue)` — 세 피처 모두 NIC/SC 갭 줄이기
- Phase-domain shift(approach vs insert brightness 분포 차이)는 상대적으로 작음 → 무시 가능

---

## Section 4-5 — 전경 Blob 중심 분포 (Input Spatial Bias)

**얻는 것**: 로봇/플러그가 이미지 내 어느 위치에 주로 등장하는가.

| cx_IQR & cy_IQR | 의미 | 대응 |
|-----------------|------|------|
| 둘 다 < 0.10 (BIASED) | 물체가 항상 같은 위치 | **RandomAffine / RandomCrop 필수** |
| > 0.20 | 물체가 다양한 위치에 등장 | augmentation 선택적 |

- NIC vs SC의 blob 중심이 **분리**되어 있으면 → 인코더가 위치로 태스크 구분 가능 → shortcut 위험

---

## Section 5-1 / 5-2 — Cross-view 분석 (View Diversity & Shared Encoder Validity)

**얻는 것**: 세 뷰(left/center/right)를 공유 인코더로 처리해도 되는가.

| mean diff (0~255 scale) | 결론 |
|-------------------------|------|
| < 10 | 단일 공유 인코더 충분 |
| 10 ~ 20 | 공유 인코더 + 뷰별 projection head 고려 |
| > 20 | 뷰별 독립 인코더 또는 뷰 ID 임베딩 필요 |

| Pearson r (cross-view feature) | 결론 |
|-------------------------------|------|
| > 0.80 (모든 특징) | 단일 공유 인코더 사용 가능 ✅ |
| < 0.50 (일부 특징) | 뷰별 독립 인코더 권장 |

- **laplacian r이 낮으면** 뷰마다 초점 거리가 달라 선명도 차이 존재 → blur augmentation 강도를 뷰별로 다르게 설정

---

## Section 6 — Bias Summary (설계 결론)

**우선순위 체계**:

| 편향 종류 | 영향 범위 | 긴급도 |
|-----------|----------|--------|
| Task 불균형 (input) | 모든 배치 | 🔴 필수 — 배치 내 1:1 샘플링 |
| Phase 불균형 (input) | 모든 배치 | 🔴 필수 — 오버샘플링 또는 가중치 샘플링 |
| Tip error bias (output) | 정책 일반화 | 🟡 중요 — 삽입 시작점 다양화 |
| 공간 편향 BIASED (input) | 인코더 ROI | 🟡 중요 — RandomCrop / Affine |
| 공간 편향 caution (input) | 인코더 ROI | 🟢 권장 — 약한 augmentation |

---

## Section 7 — Temporal Redundancy (프레임 간 시간적 중복성)

**얻는 것**: 연속 프레임이 얼마나 중복되는가 → 배치 샘플링 전략 결정.

| MAD 평균 (0~255 scale) | 의미 | 대응 |
|------------------------|------|------|
| < 1.5 | 연속 프레임이 사실상 동일 | **temporal stride ≥ 3~5 권장** |
| 1.5 ~ 5 | 적당한 변화량 | stride 2 정도 |
| > 5 | 모션이 빠름 | stride 1, 모든 프레임 활용 |

- SSIM > 0.95인 연속 프레임 비율 > 30% → random interval sampling 전략 필요
- approach(빠른 이동)와 insert(미세 조정)의 MAD 차이가 크면 → **위상별 stride 따로 설정** 검토

---

## Section 8 — PCA / t-SNE Separability (시각적 분리 가능성)

**얻는 것**: NIC/SC 및 approach/insert가 이미지 특징 공간에서 분리되는가 → 인코더 설계 결론.

| 패턴 | 의미 | 대응 |
|------|------|------|
| NIC/SC 클러스터 명확히 분리 | 태스크 구분 정보 충분 | 인코더가 task-type feature 학습 가능 |
| NIC/SC가 완전히 섞임 | 시각적으로 구별 불가 | 추가 모달리티 필요 |
| phase 라벨 분리 | approach↔insert가 시각적으로 다름 | phase-conditional 인코더 고려 |
| phase 라벨이 섞임 | approach↔insert 시각적으로 비슷 | phase를 visual signal로 쓰기 어려움 |

**PCA explained variance 기준**:
- 상위 10 PC → 분산 70% 이상 설명: 정보가 저차원에 집중 (좋음, 작은 인코더로 충분)
- 50 PC를 써도 50% 미만: 이미지 복잡 or 노이즈 많음 → **더 큰 인코더 필요**

---

## Section 9 — Inter-session Diversity (세션 간 시각적 다양성)

**얻는 것**: 서로 다른 수집 세션이 얼마나 다른 장면을 보여주는가 → 배경 암기(overfitting) 위험 평가.

| 지표 | 임계값 | 의미 |
|------|--------|------|
| 세션 간 mean-image SSIM | > 0.90 | 모든 세션이 거의 같은 장면 → 배경 암기 위험 ⚠️ |
| 세션 간 hue histogram L1 | < 0.05 | 색상 분포 고정 → **ColorJitter 필수** |
| 세션 간 brightness std | < 0.02 | 조명 변화 없음 → **RandomBrightness 필수** |

- 세션 간 SSIM이 낮으면(< 0.70) → 수집 조건에 이미 충분한 다양성 → augmentation 강도 낮춰도 됨
- Rail별로 따로 계산: 같은 rail은 카메라 앵글 유사 → 더 취약

---

## Section 10 — Gripper Offset Deviation (그리퍼 오프셋 편차 분석)

**얻는 것**: 에피소드 시작 시 실제 gripper_offset 편차가 명목값 대비 얼마나 분포하는가.

명목 gripper_offset:
- **SFP (NIC)**: translation z = 0.04245m
- **SC**: translation z = 0.04045m (2mm 차이)

| 위치 편차 std | 의미 | 대응 |
|--------------|------|------|
| > 2mm | 실제 편차 범위 커버됨 → OK | 별도 augmentation 불필요 |
| < 1mm | 너무 이상적 | **±2mm 위치 perturbation augmentation 필수** |

| 회전 편차 std | 의미 | 대응 |
|--------------|------|------|
| > 0.04 rad | 실제 편차 범위 커버됨 → OK | 별도 augmentation 불필요 |
| < 0.02 rad | 너무 이상적 | **±0.04 rad rotation jitter 필수** |

**초기 편차 크기 ↔ 수렴 난이도 상관관계**:
- 양의 상관 (편차 클수록 수렴 어려움): 정상 — 편차가 실제 난이도를 유발
- 무상관: 현재 policy가 이미 편차에 robust하거나, 편차 범위가 너무 좁아 영향 없음
- 음의 상관: 데이터 품질 이슈 확인 필요

---

## 종합 증강 전략 체크리스트

| 증강 | 근거 섹션 | 적용 대상 |
|------|-----------|-----------|
| **배치 내 NIC:SC = 1:1** | 1-2 | 전체 학습 |
| **approach 오버샘플링** | 1-2 | 전체 학습 |
| `RandomAffine` (translate + rotate) | 3-2, 4-5 | 공간 편향 해소 |
| `ColorJitter` (brightness, saturation, hue) | 4-4, 9 | NIC/SC 도메인 갭 희석 |
| `GaussianNoise` (뷰별 강도 다르게) | 4-3 | right/nic 노이즈 균일화 |
| Temporal stride ≥ 3 | 7 | 배치 내 중복 프레임 제거 |
| `GaussianBlur` (약하게, p ≤ 0.2) | 4-3 | 선명도 다양화 (과하면 안 됨) |
| 하단 crop (20~30px) | 3-2 | 정적 배경 제거 |
| Gripper offset perturbation | 10 | 편차 std < 2mm인 경우만 |
