# AIC Sejong

[한국어](README.ko.md) | [English](README.en.md)

Intrinsic 및 Open Robotics가 주관한 AI for Industry Challenge의 솔루션 코드입니다 (70th/166 Teams) <br>

[![Hugging Face Hub](https://img.shields.io/badge/Hugging%20Face-aic--sejong--team-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/aic-sejong-team)

## 대회 설명
AI for Industry Challenge는 UR5e 로봇 팔이 케이블을 지정된 포트에 삽입하는 산업 자동화 태스크에서, 시뮬레이션 환경 기반 정책의 인식 정확도와 삽입 성공률을 평가하는 챌린지입니다.

<details>
<summary><strong>[1] 케이블 삽입 태스크 및 정책 구성</strong></summary>

참가자는 카메라 관측, 로봇 상태, 힘/토크(Force/Torque) 센서 정보를 활용하여 포트 위치를 추정하고, 케이블 삽입을 수행하는 정책을 개발해야 합니다.

| 구성 요소 | 역할 |
|-----------|------|
| UR5e 로봇 팔 | 케이블 삽입 동작 수행 |
| YOLO-pose | 포트 자세와 주요 지점 검출 |
| 멀티뷰 삼각측량 | 여러 카메라 관측을 활용한 3D 포트 위치 추정 |
| Vision 및 F/T 센서 | 삽입 실패 감지와 재시도 정책 판단 |
| Gazebo/AIC Simulator | 반복 실험, 정책 검증, 데이터 수집 환경 |

본 솔루션은 포트 위치 추정 오차를 줄이기 위해 yaw/XYZ 정렬 로직을 구현하고, 삽입 실패 시나리오에 대응하기 위한 센서 기반 재시도 흐름을 함께 구성했습니다.

</details>

<details>
<summary><strong>[2] 데이터 수집 및 협업 관리</strong></summary>

반복 실험 데이터 수집 병목을 줄이기 위해 Gazebo 기반 자동 수집 노드를 구현하고, YOLO 학습 데이터와 LeRobot 형식 데이터셋을 관리했습니다.

| 항목 | 활용 목적 |
|------|-----------|
| Gazebo 자동 수집 노드 | 반복 실험 데이터 생성 및 수집 시간 단축 |
| YOLO 학습 데이터 | 포트 검출 및 자세 추정 모델 학습 |
| LeRobot 데이터셋 | 정책 학습 및 재현 가능한 실험 관리 |
| GitHub | 코드, 실험 산출물, 협업 이슈 관리 |
| Hugging Face Hub | 모델과 데이터셋 공유 및 로드 |
| Notion | 일정, 역할, 회의 내용 문서화 |

프로젝트 산출물은 [Hugging Face Hub](https://huggingface.co/aic-sejong-team)를 통해 관리할 수 있도록 구성했습니다.

</details>

## Key Contributions

```
1. YOLO-pose 기반 포트 자세 및 멀티뷰 삼각측량 기반 포트 위치 추정, 위치 추정 오차 감소를 위한 yaw/XYZ 정렬 로직 구현 (XX% 성능 개선)
2. 삽입 실패 시나리오 대비를 위한 Vision 및 F/T 센서 기반 재시도 로직 구현 (YY% 성능 개선)
3. 반복 실험 데이터 수집 병목을 줄이기 위한 Gazebo 기반 데이터 자동 수집 노드 구현
4. 분산된 실험 산출물과 협업 흐름을 정리하기 위한 GitHub/Hugging Face Hub/Notion 관리
5. 오로카 네이버 카페 및 모두의 연구소에서 팀원 구인, 일정·역할 조율 및 문서화
```
<br>

## 모델 및 데이터

프로젝트 모델과 데이터셋 산출물은 [Hugging Face Hub](https://huggingface.co/aic-sejong-team)에서 관리합니다.

| 리소스 | 링크 |
|--------|------|
| Organization | [aic-sejong-team](https://huggingface.co/aic-sejong-team) |
| LeRobot Dataset | [aic-sejong-team/aic-dataset](https://huggingface.co/datasets/aic-sejong-team/aic-dataset) |
| Entrance Dataset | [aic-sejong-team/aic-entrance-dataset](https://huggingface.co/datasets/aic-sejong-team/aic-entrance-dataset) |
| ACT Policy | [aic-sejong-team/act_AIC](https://huggingface.co/aic-sejong-team/act_AIC) |

## 시작하기

### 1. 의존성 설치
```bash
git clone https://github.com/JungSeong/AIC_Sejong.git ~/AIC_Sejong
cd ~/AIC_Sejong/ws_aic/src
pixi install
```

### 2. Eval 컨테이너 준비
```bash
export DBX_CONTAINER_MANAGER=docker
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
```

### 3. 정책 실행
```bash
# Terminal 1
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true

# Terminal 2
cd ~/AIC_Sejong/ws_aic/src
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=final_policy.FinalPolicy
```

## 저장소 구조

| 경로 | 역할 |
|------|------|
| `data/` | 대회 train/dev/test 메타데이터와 submission 파일 |
| `ws_aic/src/aic/` | AIC 공식 저장소 및 ROS 2 기반 평가 환경 |
| `ws_aic/src/ais/` | 팀 자체 개발 패키지 |
| `ws_aic/src/ais/ais_motion_planning/` | YOLO + 멀티뷰 기반 포트 검출 및 모션 플래닝 |
| `ws_aic/src/ais/ais_auto_capture/` | Gazebo 기반 자동 데이터 수집 |
| `ws_aic/src/ais/ais_yolo_train/` | YOLO 학습 데이터 수집 및 평가 |
| `ws_aic/src/ais/ais_retry_classifier/` | 삽입 실패 감지 및 재시도 판단 실험 |
| `ws_aic/src/ais/ais_load_model_from_hf/` | Hugging Face Hub 모델/데이터셋 업로드 및 로드 유틸리티 |
| `ws_aic/src/ais/ais_eda/` | 멀티뷰 편향 및 위치 추정 오차 분석 노트북 |
| `ws_aic/src/docs/` | 실험 문서와 세션별 요약 |
| `readme/` | 한국어/영문 README 문서 |

## 참고 링크

- [Hugging Face Hub](https://huggingface.co/aic-sejong-team)
- [모션 플래닝 패키지](../ws_aic/src/ais/ais_motion_planning/README.md)
- [자동 데이터 수집 패키지](../ws_aic/src/ais/ais_auto_capture/README.md)
- [재시도 분류기 패키지](../ws_aic/src/ais/ais_retry_classifier/README.md)
- [실험 의사코드 문서](<../ws_aic/src/docs/psuedo code/pseudo_code.md>)
