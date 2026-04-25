# LeRobot v0.5.1 API 마이그레이션 및 변환 로직 디버깅 요약

> 날짜: 2026-04-23
> 작성자: Gemini CLI
> 관련 이슈: LeRobot v0.5.x 버전의 파괴적 API 변경 대응

---

## 1. 개요
데이터 수집 후 LeRobot 형식으로 변환하는 과정에서 발생한 `lerobot` v0.5.1 라이브러리의 API 변경 이슈를 분석하고, 이를 해결하기 위한 단계적 시도와 사고 과정을 기록함.

---

## 2. 사고 과정 및 디버깅 단계

### 단계 1: 패키지 구조 변경 및 Import 에러
- **현상**: `lerobot.common.datasets` 임포트 실패.
- **분석**: v0.5.1부터 패키지 구조가 대폭 변경되어 `common` 네임스페이스가 제거됨.
- **조치**: 임포트 경로를 `lerobot.datasets.lerobot_dataset`으로 수정.

### 단계 2: 에피소드 관리 API 변경
- **현상**: `AttributeError: 'LeRobotDataset' object has no attribute 'start_new_episode'`.
- **분석**: 명시적인 에피소드 시작 메서드가 제거되고 `add_frame`과 `save_episode` 조합으로 자동 관리되도록 변경됨. 또한 `consolidate()`가 `finalize()`로 대체됨.
- **조치**: 코드 내 미지원 메서드 제거 및 `finalize()` 적용.

### 단계 3: 'task' 피처의 시스템화 및 스키마 충돌
- **현상**: `ValueError: Missing features: {'task'}`.
- **분석**: 최신 버전에서는 `task`가 사용 정의 피처가 아닌 **시스템 필수 컬럼**으로 격상됨. 데이터 타입은 `int64` 인덱스여야 하며, 데이터셋 생성 시 정의한 `features` 스키마와 내부 시스템 스키마 간의 충돌 발생.
- **조치**: `features` 정의에서 `task`를 제거하고 `add_frame` 시점에 정수형(`int`) 인덱스를 주입하도록 수정.

### 단계 4: 메타데이터 및 태스크 매핑 이슈 (진행 중)
- **현상**: `TypeError: argument of type 'builtin_function_or_method' is not iterable` (save_episode 호출 시).
- **분석**: `dataset.meta.tasks`에 직접 리스트를 할당하는 방식이 내부 메서드와 충돌하거나, `LeRobotDataset`의 메타데이터 구조가 이전 버전과 다르게 `pd.Index` 또는 특정 객체 형태를 기대함.
- **가설**: `save_episode`가 호출될 때 내부적으로 프레임에 기록된 `task` 인덱스를 실제 이름과 대조하는 과정에서 매핑 테이블이 누락되거나 잘못된 접근이 발생하고 있음.

---

## 3. 핵심 기술 인사이트
- **버전 고정의 필요성**: `lerobot-robot-ros` 의존성이 `lerobot >= 0.5.0`을 강제하고 있어 하위 버전(0.4.3)으로의 다운그레이드가 불가능함.
- **시스템 피처 규격**: 최신 LeRobot 스키마는 HuggingFace `datasets`의 엄격한 유효성 검사를 따르며, 특히 `task`와 `index` 필드는 예약어(Reserved)에 준하는 처리가 필요함.

---

## 4. 향후 과제
- `LeRobotDataset.create` 시점에 태스크 정보를 주입하는 정확한 최신 인자 확인 (메타데이터 수동 조작 대신 공식 API 탐색).
- `dataset.meta` 객체의 내부 구조를 `inspect`하여 `tasks` 속성이 프로퍼티인지 메서드인지 재확인 후 매핑 로직 수정.
