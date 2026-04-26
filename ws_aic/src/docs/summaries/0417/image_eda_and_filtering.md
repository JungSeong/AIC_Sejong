# AIC Capture 이미지 EDA & 가우시안 기반 정제 방법

> 작성일: 2026-04-17  
> 데이터: `aic_data/captures/` (trial_1~3, 각 531 프레임, 3-cam: left / center / right)  
> 해상도: **1024 × 1152 px, uint8 BGR**

---

## 1. 데이터셋 개요

| 에피소드 | Trial | 프레임 수 | 카메라 |
|---|---|---|---|
| 20260417_133201_trial_1 | SFP → NIC rail 0 | 531 × 3 | left / center / right |
| 20260417_133453_trial_2 | SFP → NIC rail 1 | 531 × 3 | left / center / right |
| 20260417_133658_trial_3 | SC → SC rail 1   | 531 × 3 | left / center (right 없음) |

---

## 2. 이미지 EDA 결과

### 2-1. 밝기 (Mean Grayscale Intensity, 0–255)

| 에피소드 | left | center | right |
|---|---|---|---|
| trial_1 | μ=147.3, σ=2.5 | μ=143.3, σ=7.2 | μ=150.3, σ=2.8 |
| trial_2 | μ=145.0, σ=10.6 | μ=145.3, σ=12.6 | μ=150.7, σ=6.6 |
| trial_3 | μ=105.2, σ=12.4 | μ=90.1, σ=17.4 | μ=89.7, σ=18.0 |

**주요 관찰:**
- trial_1·2는 밝기 μ ≈ 143–151, σ < 13 으로 안정적
- **trial_3는 μ ≈ 90–105, 약 40 포인트 어두움.** SC 삽입 시 보드 위치가 x=0.17, y=0.0 으로 이동하면서 카메라 앵글 대비 조명 조건이 다름
- trial_3 center/right의 σ=17–18 → 로봇 팔이 광원을 가리는 순간 밝기 급변

### 2-2. 대비 (Grayscale Standard Deviation)

| 에피소드 | left | center | right |
|---|---|---|---|
| trial_1 | μ=79.5, σ=1.4 | μ=80.5, σ=1.3 | μ=79.1, σ=1.2 |
| trial_2 | μ=80.5, σ=1.0 | μ=82.2, σ=1.1 | μ=82.2, σ=1.1 |
| trial_3 | μ=68.9, σ=6.0 | μ=55.5, σ=15.5 | μ=47.3, σ=20.1 |

**주요 관찰:**
- trial_1·2: 대비 σ < 2 → 매우 안정적, 씬이 일관됨
- trial_3 center/right: σ = 15–20 → **프레임마다 대비 편차 매우 큼**
  - 케이블·그리퍼가 카메라 화각을 많이 가리는 구간 발생 추정

### 2-3. 선명도 (Laplacian Variance — 높을수록 선명)

| 에피소드 | left | center | right |
|---|---|---|---|
| trial_1 | μ=2513, σ=49 | μ=2420, σ=277 | μ=2269, σ=66 |
| trial_2 | μ=2390, σ=228 | μ=2347, σ=346 | μ=2492, σ=235 |
| trial_3 | μ=2073, σ=213 | μ=1529, σ=310 | μ=1941, σ=411 |

**주요 관찰:**
- trial_3 center: μ=1529 으로 가장 낮음 → 보드 위치 차이로 초점 거리 변화 가능성
- center 카메라가 전반적으로 σ가 가장 큼 → 로봇 팔 진입·후퇴 시 아웃-포커스 프레임 포함
- trial_1 left: σ=49 → 가장 안정적인 시퀀스

### 2-4. 주파수 도메인 분석 (FFT Low-Frequency Ratio)

2D FFT 후 중심 1/8 영역 에너지 비율 (저주파 = 큰 구조물·배경):

| 에피소드 | left | center | right |
|---|---|---|---|
| trial_1 | 46.5% | 47.7% | 47.3% |
| trial_2 | 47.2% | 47.9% | 48.4% |
| trial_3 | 46.1% | 47.6% | 46.1% |

**주요 관찰:**
- 세 trial 모두 저주파 비율 ≈ 46–48%로 매우 균일
- 고주파(엣지·텍스처) 비율도 일정 → **주파수 측면의 씬 구성은 안정적**
- 따라서 FFT 지표는 이상 프레임 탐지 기준으로 단독 사용 시 민감도 낮음

### 2-5. RGB 채널 밸런스

- trial_1·2: R≈G≈B (±5 이내) → **색 중립적 조명**, 채널 편향 없음
- trial_3: 세 채널 모두 어둡고 R/B 약간 높음 → 조명 반사각 차이

---

## 3. 가우시안 기반 이미지 정제 방법

### 기본 원리

각 메트릭 $x$가 에피소드·카메라 단위로 가우시안 $\mathcal{N}(\mu, \sigma^2)$을 따른다고 가정하면,  
**이상치 프레임** = $|x - \mu| > k\sigma$ (통상 $k=2$ 또는 $3$)

```
정상 구간 유지율 (이론):  k=2 → 95.4%,  k=3 → 99.7%
```

### 3-1. Per-Episode, Per-Camera 필터링 (기본)

```python
def gaussian_filter_frames(metric_series: np.ndarray, k: float = 2.0) -> np.ndarray:
    """
    Returns boolean mask: True = 정상 프레임 유지
    """
    mu, sigma = metric_series.mean(), metric_series.std()
    return np.abs(metric_series - mu) <= k * sigma
```

각 에피소드·카메라 조합에서 독립적으로 μ·σ를 추정한 후 마스킹.

**적용 순서:**
1. 밝기 필터 (k=2.5): 로봇 팔 광원 차폐·플래시 제거
2. 선명도 필터 (k=2.0): 모션 블러·아웃포커스 프레임 제거
3. 대비 필터 (k=2.0): 씬이 거의 단색인 비정상 프레임 제거

### 3-2. 멀티카메라 교차 필터링 (권장)

같은 time-step에서 **left·center·right 세 카메라가 모두 정상**인 프레임만 유지:

```python
def cross_camera_filter(masks: dict[str, np.ndarray]) -> np.ndarray:
    """
    masks: {'left': bool_array, 'center': bool_array, 'right': bool_array}
    Returns: AND of all masks (모든 카메라가 정상인 프레임)
    """
    combined = np.ones(len(list(masks.values())[0]), dtype=bool)
    for m in masks.values():
        combined &= m
    return combined
```

단일 카메라 이상치보다 씬 자체의 문제 (조명 급변, 오브젝트 충돌 등)를 더 잘 잡아냄.

### 3-3. 슬라이딩 윈도우 가우시안 (Phase-Aware 필터링)

삽입 동작의 Phase별로 다른 μ·σ를 가짐 (예: approach 구간 vs. insertion 구간):

```python
def sliding_gaussian_filter(
    metric_series: np.ndarray,
    window: int = 30,      # 3초 @ 10Hz
    k: float = 2.0
) -> np.ndarray:
    """
    로컬 윈도우 내에서 μ·σ를 추정해 Phase 변화에 적응.
    """
    n = len(metric_series)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2)
        local = metric_series[lo:hi]
        mu, sigma = local.mean(), local.std()
        if sigma > 0:
            mask[i] = abs(metric_series[i] - mu) <= k * sigma
    return mask
```

### 3-4. 예상 필터링 결과 (추정)

EDA 수치 기반 시뮬레이션:

| 에피소드 | 카메라 | 예상 제거율 (k=2) |
|---|---|---|
| trial_1 | left | ~3% (σ 작음) |
| trial_1 | center | ~8% (Laplacian σ 큼) |
| trial_2 | center | ~10% |
| trial_3 | center | ~18% (밝기·대비 불안정) |
| trial_3 | right | ~20% |

---

## 4. 카메라별 특성 요약

| 카메라 | 특성 | 역할 추정 |
|---|---|---|
| **left** | 밝기 안정 (σ 최소), 선명도 최고 | 포트 삽입 앵글 (side-view) |
| **center** | Laplacian σ 가장 큼, trial_3에서 어두움 | 정면·그리퍼 앵글 |
| **right** | trial_3에서 대비 σ 20 → 가장 불안정 | 반대 side-view |

---

## 5. 실용 권장사항

### 데이터 수집 단계 (collect_data.py 연계)
- **trial_3 데이터 비중을 높일 것**: 밝기·대비 분산이 크다 = 다양한 상황 포함 = 학습에 유리하지만, 노이즈도 많음
- `--diversify` 옵션으로 보드 위치를 랜덤화하면 brightness 분포가 더 넓어짐 → 가우시안 필터의 μ 추정을 에피소드 단위가 아닌 trial-type 단위로 해야 함

### 전처리 파이프라인 제안
```
steps.jsonl 로드
  └─ 각 step에서 left/center/right brightness·blur 계산
       └─ per-episode 가우시안 파라미터 추정
            └─ mask = brightness_ok & blur_ok & cross_camera_ok
                 └─ 정상 프레임만 lerobot Dataset에 포함
```

### 임계값 가이드

| 메트릭 | 권장 k | 근거 |
|---|---|---|
| Brightness | k=2.5 | 로봇 팔 차폐는 일시적, 너무 좁으면 정상 범위도 제거 |
| Laplacian (blur) | k=2.0 | 모션 블러는 명확한 이상치, 엄격하게 |
| Contrast | k=2.0 | 대비 급락 = 씬 이상 신호 |
| FFT ratio | — | 현재 편차 작음, 추가 데이터 확보 후 재평가 |

---

## 6. 향후 분석 항목

- **Phase 레이블 연동**: steps.jsonl의 `phase` 필드 (approach/insert/retract)와 메트릭 분포를 교차 분석하면 phase별 필터 파라미터 최적화 가능
- **픽셀 차이 기반 motion 지표**: 연속 프레임 간 `|I_t - I_{t-1}|` 평균으로 정지 구간(중복 프레임) 탐지
- **F/T 센서 연동 필터**: `wrist_wrench.force.z` 급변 구간은 삽입 접촉 순간 → 해당 프레임은 별도 가중치 부여 가능
- **데이터 확충 후 재EDA**: 현재 3 에피소드(~1600 프레임)는 통계적으로 부족; 최소 50 에피소드 이상에서 distribution이 안정화됨
