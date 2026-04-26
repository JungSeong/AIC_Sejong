# 트러블슈팅

> 날짜: 2026-04-05
> 원본: `ws_aic/src/aic/docs/troubleshooting.md`

---

## 문제별 해결 방법

### 1. Gazebo RTF(실시간 비율)가 낮음

**원인 A — 이중 GPU 시스템에서 통합 GPU 사용**
```bash
# GPU 확인
glxinfo -B    # 이산 GPU 정보가 표시되어야 함
nvidia-smi    # aic_eval 실행 중 gz sim 프로세스 확인

# 이산 GPU로 강제 전환
sudo prime-select nvidia
# 로그아웃 후 재로그인 필요
```

**원인 B — GPU 없는 환경**

`aic_description/world/aic.sdf`에서 Global Illumination 비활성화:
```xml
<enabled>false</enabled>  <!-- 2곳 수정 -->
```
> ⚠️ GI 비활성화 시 시각적 품질 저하 → 비전 기반 정책에 영향 가능

---

### 2. Zenoh shared memory 경고

```
WARN Watchdog Validator ... error setting scheduling priority for thread
```

**무시해도 무방**. 정상 동작에 영향 없음. `CAP_SYS_NICE` 권한 없어 스케줄링 우선순위 설정 실패하는 것이며, 일반적인 워크로드에서는 문제 없음.

shared memory 정상 동작 확인:
```bash
ls -lh /dev/shm | grep zenoh   # Zenoh SHM 파일 확인
```

---

### 3. NVIDIA RTX 50xx 카드에서 PyTorch 미지원

```
UserWarning: NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible...
```

`pixi.toml`에 PyTorch 버전 오버라이드 추가:
```toml
[pypi-options.dependency-overrides]
torch = ">=2.7.1"
torchvision = ">=0.22.1"
```

---

### 4. `Error: no such container aic_eval`

Distrobox가 기본으로 Podman을 사용하기 때문. Docker로 강제 설정:
```bash
export DBX_CONTAINER_MANAGER=docker
```

---

## 빠른 참조

| 증상 | 원인 | 해결 |
|------|------|------|
| Gazebo RTF 저하 | 통합 GPU 사용 | `sudo prime-select nvidia` |
| Gazebo RTF 저하 | GPU 없음 | SDF에서 GI 비활성화 |
| Zenoh 경고 메시지 | 스케줄링 권한 부족 | 무시 |
| RTX 50xx PyTorch 오류 | CUDA sm_120 미지원 | `pixi.toml` 버전 오버라이드 |
| no such container 오류 | Podman 기본 설정 | `DBX_CONTAINER_MANAGER=docker` |
