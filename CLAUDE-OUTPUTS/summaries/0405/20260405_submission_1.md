# 제출 가이드라인

> 날짜: 2026-04-05
> 원본: `ws_aic/src/aic/docs/submission.md`

---

## 제출 흐름

```
1. Docker 이미지 준비·빌드  →  2. 로컬 검증  →  3. AWS ECR 업로드  →  4. 포털 등록  →  5. 결과 모니터링
```

---

## 1. 이미지 빌드

### 기본 방법 (aic_model 사용)
`docker/docker-compose.yaml`에서 정책 경로만 변경:
```yaml
command: --ros-args -p policy:=aic_model.MyPolicy
```

### 커스텀 방법 (별도 패키지)
```bash
mkdir -p docker/my_policy && cp docker/aic_model/Dockerfile docker/my_policy/
```
`Dockerfile`에 패키지 추가:
```dockerfile
COPY my_policy_node /ws_aic/src/aic/my_policy_node
CMD ["--ros-args", "-p", "policy:=my_policy_node.MyPolicy"]
```

### 빌드 실행
```bash
docker compose -f docker/docker-compose.yaml build model
```

---

## 2. 로컬 검증 (필수)

```bash
docker compose -f docker/docker-compose.yaml up
```
> ⚠️ 건너뛰지 말 것. 로컬 실패 시 포털에서 자동 거부되며 일일 제출 횟수에 포함됨.

---

## 3. ECR 업로드

```bash
# 인증
aws configure --profile <team_name>
export AWS_PROFILE=<team_name>
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com

# 태그 및 푸시
docker tag localhost/my-solution:v1 \
  973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v1
docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v1
```

> ECR 태그 **불변** — 동일 태그 덮어쓰기 불가. 새 제출마다 `:v2`, `:v3` 등으로 증가 필요.

---

## 4. 포털 등록

1. 포털 로그인 → `AI for Industry Challenge` → `Submit`
2. `Qualification` 단계 선택
3. `OCI Image` 필드에 전체 이미지 URI 입력
4. `Submit` 클릭

---

## 5. 평가 상태 모니터링

| 상태 | 의미 |
|------|------|
| **Submitted** | URI 수신 완료 |
| **Queued** | 평가 노드 대기 중 |
| **Running** | 시뮬레이션 평가 진행 중 (보통 5~15분) |
| **Finished** | 평가 완료, 리더보드 반영 |
| **Failed** | 런타임 오류 또는 타임아웃 |

---

## 주요 제한 사항

| 항목 | 내용 |
|------|------|
| 하루 제출 횟수 | **팀당 1회** |
| 총 제출 횟수 | 무제한 |
| 제출 주체 | 팀원 누구나 가능 |
| ECR 로그인 유효시간 | 12시간 (만료 시 재인증 필요) |

---

*커스텀 Dockerfile 필요 시: `custom_dockerfile.md` / 접근 제어 확인: `access_control.md`*
