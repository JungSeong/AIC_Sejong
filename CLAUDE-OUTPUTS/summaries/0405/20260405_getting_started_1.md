# 시작 가이드

> 날짜: 2026-04-05
> 원본: `ws_aic/src/aic/docs/getting_started.md`

---

## 아키텍처

두 컴포넌트로 구성:
- **Evaluation Component** (제공) — 시뮬레이션, 로봇, 센서, 채점 실행. Docker 이미지(`aic_eval`)로 제공
- **Participant Model** (직접 구현) — 센서 데이터 처리 후 로봇 명령을 내리는 ROS 2 노드

> ROS 2 공식 평가 배포판: **ROS 2 Kilted Kaiju**

---

## 최소 시스템 사양

| 항목 | 로컬 개발 | 클라우드 평가 |
|------|-----------|--------------|
| CPU | 4–8 코어 | 64 vCPU |
| RAM | 32 GB+ | 256 GiB |
| GPU | NVIDIA RTX 2070+ | NVIDIA L4 |
| VRAM | 8 GB+ | 24 GiB |
| OS | Ubuntu 24.04 | — |

---

## 필수 도구 설치

| 도구 | 역할 | 설치 |
|------|------|------|
| Docker | 컨테이너 실행 | [공식 문서](https://docs.docker.com/engine/install/) |
| Distrobox | `aic_eval` 컨테이너-호스트 통합 | `sudo apt install distrobox` |
| Pixi | ROS 2 패키지·의존성 관리 | `curl -fsSL https://pixi.sh/install.sh \| sh` |
| NVIDIA Container Toolkit | GPU 가속 (선택) | NVIDIA 공식 문서 참조 |

---

## 빠른 시작 3단계

### Step 1 — 워크스페이스 설정
```bash
mkdir -p ~/ws_aic/src && cd ~/ws_aic/src
git clone https://github.com/intrinsic-dev/aic
cd aic && pixi install
```

### Step 2 — 평가 컨테이너 실행
```bash
export DBX_CONTAINER_MANAGER=docker
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
distrobox enter -r aic_eval
# 컨테이너 내부에서:
/entrypoint.sh ground_truth:=false start_aic_engine:=true
```

정상 실행 시 **Gazebo**(시뮬레이션)와 **RViz**(시각화) 두 창이 열리며, `aic_model` 노드 대기 로그 출력.

### Step 3 — 예시 정책 실행
```bash
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.WaveArm
```

결과: 3번의 Trial 자동 진행, 각 Trial 채점 결과 출력. 결과 저장 위치: `~/aic_results/`

---

## 주의사항

- Pixi 환경 내 패키지 변경 사항은 자동 반영 안 됨 → `pixi reinstall <package_name>` 필요
- 컨테이너 시작 후 30초 이내에 `aic_model` 노드 연결 필요
- GPU 없는 환경은 성능 저하 가능 (RTF 저하) → `troubleshooting.md` 참조

---

*다음 단계: `submission.md`에서 정책 컨테이너화 방법 확인*
