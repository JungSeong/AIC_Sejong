# Docker 이미지 빌드 & 로컬 평가 & 제출 가이드

> 정책 노드 완성 후 제출까지의 전체 흐름 정리.
> 소스: `docs/submission.md`, `docker/docker-compose.yaml`, `docker/aic_model/Dockerfile`

---

## 1. 전체 흐름

```
① Dockerfile 수정
      ↓
② docker-compose.yaml 수정
      ↓
③ pixi.lock 갱신
      ↓
④ 로컬 이미지 빌드
      ↓
⑤ 로컬 검증 (docker compose up)
      ↓
⑥ ECR 업로드
      ↓
⑦ 포털 등록
```

---

## 2. Dockerfile 수정

기존 `docker/aic_model/Dockerfile`을 복사해서 내 패키지용으로 만든다.

```bash
cd ~/aic_sejong/ws_aic/src/aic
mkdir -p docker/my_policy_node
cp docker/aic_model/Dockerfile docker/my_policy_node/Dockerfile
```

`docker/my_policy_node/Dockerfile`에서 수정할 두 곳:

```dockerfile
# ① 내 패키지 COPY 추가 (pixi.toml COPY 바로 아래)
COPY my_policy_node /ws_aic/src/aic/my_policy_node

# ② CMD를 내 정책 클래스로 변경 (파일 맨 아래)
CMD ["--ros-args", "-p", "policy:=my_policy_node.Baseline"]
```

---

## 3. docker-compose.yaml 수정

`docker/docker-compose.yaml`의 `model` 서비스에서 두 곳 수정:

```yaml
services:
  model:
    image: my-solution:v1
    build:
      dockerfile: docker/my_policy_node/Dockerfile   # ← 내 Dockerfile 경로
      context: ..
    command: --ros-args -p policy:=my_policy_node.Baseline  # ← 내 정책 클래스
    # ... 나머지 항목은 그대로 유지
```

---

## 4. pixi.lock 갱신

Dockerfile 내부는 `pixi install --locked`로 빌드하므로 **lock 파일이 최신 상태**여야 한다.

```bash
cd ~/aic_sejong/ws_aic/src/aic
pixi install   # pixi.lock 자동 갱신
```

> **전제 조건:** 루트 `pixi.toml`의 `[dependencies]`에 아래가 등록되어 있어야 함.
> ```toml
> ros-kilted-my-policy-node = { path = "my_policy_node" }
> ```

---

## 5. 로컬 이미지 빌드

```bash
cd ~/aic_sejong/ws_aic/src/aic
docker compose -f docker/docker-compose.yaml build model
```

> **디스크 공간 부족 시** 빌드 실패. 먼저 정리:
> ```bash
> docker system prune -a
> df -h /
> ```

---

## 6. 로컬 검증 (필수)

**제출 전 반드시 로컬에서 검증.** 포털 제출 후 실패 시에도 일일 제출 횟수(팀당 1회) 차감.

```bash
cd ~/aic_sejong/ws_aic/src/aic
docker compose -f docker/docker-compose.yaml up
```

`eval` + `model` 컨테이너가 함께 올라오면서 시뮬레이션이 돌아야 정상.

---

## 7. ECR 업로드

### 7-1. AWS 인증 (최초 1회, 12시간 유효)

```bash
aws configure --profile <team_name>
# Access Key ID, Secret Access Key, region(us-east-1), output(json) 입력

export AWS_PROFILE=<team_name>

aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  973918476471.dkr.ecr.us-east-1.amazonaws.com
```

> 12시간 후 만료 → 동일 명령 재실행.

### 7-2. 이미지 태그 & 푸시

```bash
# 태그 (버전 번호는 매 제출마다 증가)
docker tag my-solution:v1 \
  973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v1

# 푸시
docker push \
  973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v1
```

> **ECR 태그는 불변** — 동일 태그 덮어쓰기 불가. `:v2`, `:v3` 등으로 매번 올려야 한다.

---

## 8. 포털 등록

1. 포털 로그인 → `AI for Industry Challenge` → `Submit`
2. `Qualification` 단계 선택
3. `OCI Image` 필드에 전체 URI 입력

```
973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v1
```

4. `Submit` 클릭

---

## 9. 평가 상태 모니터링

포털 → `My Submissions` → `Qualification` 필터

| 상태 | 의미 |
|------|------|
| **Submitted** | URI 수신 완료 |
| **Queued** | 평가 노드 대기 중 |
| **Running** | 시뮬레이션 평가 진행 중 (보통 5~15분) |
| **Finished** | 평가 완료, 리더보드 반영 |
| **Failed** | 런타임 오류 또는 타임아웃 |

---

## 10. 제출 제한 사항

| 항목 | 내용 |
|------|------|
| 하루 제출 횟수 | **팀당 1회** |
| 총 제출 횟수 | 무제한 |
| ECR 로그인 유효시간 | 12시간 |
| ECR 태그 | 불변 (덮어쓰기 불가) |
