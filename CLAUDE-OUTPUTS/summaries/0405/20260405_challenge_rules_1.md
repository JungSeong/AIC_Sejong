# 챌린지 규칙 및 기술 사양

> 날짜: 2026-04-05
> 원본: `ws_aic/src/aic/docs/challenge_rules.md`

---

## 1. 대회 정신

공식 인터페이스(`aic_interfaces.md` 정의)만 사용해 일반화되고 강건한 AI 모델을 개발한다.

---

## 2. 금지 행위

### 직접 조작 금지
- 로봇·태스크보드·시뮬레이션 환경 직접 상태 변경 (텔레포트, 강제 삽입 등)
- `/scoring`, `/gazebo`, `/gz_server` 네임스페이스 접근
- 엔티티 스폰/디스폰 서비스 호출
- `/clock`, `/world_stats`, `/pause_physics`, 시뮬레이션 리셋 등

### 무단 접근 금지
- 클라우드 평가 인프라 역공학 또는 취약점 악용
- 악성 코드·백도어가 포함된 컨테이너 제출
- 제출 포털 우회

### 정보 유출 금지
- 평가 중 시뮬레이션 내부 상태 재사용
- 공식 툴킷 범위 외 평가 인프라 데이터로 모델 훈련

> **훈련 중**에는 ground truth 포함 모든 내부 상태 사용 가능. **평가 중**에는 불가.

---

## 3. 위반 제재

- 컨테이너 감사, 행동 검증, 지표 분석을 통한 자동·수동 검토
- 위반 시 **실격 및 상금 박탈**

---

## 4. `aic_model` 기술 사양

| 요건 | 내용 |
|------|------|
| 노드 이름 | 반드시 `aic_model` |
| 인터페이스 | ROS 2 Lifecycle 노드 구현 필수 |
| `unconfigured` 상태 | 로봇 명령 퍼블리시 금지 |
| `configured` 전환 | 60초 이내 완료 (모델 로딩 포함) |
| `configured` 상태 | 로봇 명령 퍼블리시 금지, `/insert_cable` 목표 거부 |
| `active` 전환 | 60초 이내 완료 |
| `active` 상태 | `/insert_cable` 목표 수락, 취소 가능 |
| 태스크 제한시간 | `Task.msg`의 `time_limit` 필드 내 완료 |
| `deactivate` 전환 | 60초 이내 `configured`로 복귀 |
| `cleanup` 전환 | 60초 이내 `unconfigured`로 복귀 |
| `shutdown` 전환 | 60초 이내 완료. 명령 퍼블리셔 제거 필수 |

---

*관련 문서: `aic_interfaces.md` / `submission.md` / `access_control.md`*
