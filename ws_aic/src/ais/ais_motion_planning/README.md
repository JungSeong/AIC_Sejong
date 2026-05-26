# Stage 1 Motion Planning — 도구 모음

Stage 1 (이동) 모션 플래닝 모듈 관련 개발 도구 및 Vision 통합 스크립트들.

## 파일 구성

| 파일 | 역할 |
|------|------|
| `collect_dataset.py` | YOLO 학습용 자동 라벨링 데이터 수집 (단일 시나리오) |
| `collect_dataset_v2.py` | + 로봇 자동 자세 순회 (다양한 뷰포인트 자동 수집) |
| `train_yolo.py` | YOLOv8 포트 검출 모델 학습 |
| `port_detector_3d.py` | YOLO 검출 + 스테레오 삼각측량 → 3D 좌표 |
| `yolo_viewer.py` | YOLO bbox 실시간 시각화 (3 카메라) |
| `stereo_test.py` | 스테레오 파이프라인 수학 검증 |
| `check_frames.py` | 현재 시뮬레이션의 TF 프레임 목록 출력 |

## 핵심 검증 결과

### Stage 1 모션 플래닝 (ground_truth=true)
- 3회 반복 × 3 Trials = **9/9 모두 사양 만족**
- axial 오차 0.1~0.5cm, radial 1.2~2.1cm, 시간 4초
- 반복 편차 0.1~0.2cm (매우 안정적)

### 스테레오 파이프라인
- 수학적 검증 오차: **0.000 mm**
- YOLO 기반 실전 오차: **약 17 mm**

### YOLO 포트 검출 모델
- mAP@0.5: **99.47%**
- 학습 데이터: 약 1,400장 (자동 라벨)

## 사용 흐름

```
1. 시뮬레이터 실행 (ground_truth=true)
   distrobox enter -r aic_eval -- /entrypoint.sh ...

2. 데이터 수집
   pixi run python tools/stage1_mp/collect_dataset_v2.py --episodes 500

3. YOLO 학습
   pixi run python tools/stage1_mp/train_yolo.py --epochs 50

4. 3D 추론 (검증)
   pixi run python tools/stage1_mp/port_detector_3d.py \
     --model ../model/ais_yolo/weights/best.pt --compare --continuous

5. 정책 실행 (StagedPolicy)
   pixi run ros2 run aic_model aic_model \
     --ros-args -p use_sim_time:=true -p policy:=my_policy_node.StagedPolicy
```

## Stage 1 사양 (Stage 2 인수인계)

```
종료 조건:
  축선상 거리:  Z_OFFSET ± 1.5 cm  (현재 Z_OFFSET = 7 cm)
  xy 이탈:     ≤ 2.5 cm
  소요 시간:   ≤ 5 초
  종료 속도:   ≤ 1 cm/s
```

StagedPolicy가 반환하는 `Stage1Result` 구조체를 Stage 2가 입력으로 사용.
