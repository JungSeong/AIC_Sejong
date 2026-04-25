# Pixi Run 작업 경로 에러 수정 요약

> 날짜: 2026-04-23
> 작성자: Gemini CLI
> 관련 코드: `code/collect_data.py`

---

## 1. 개요
데이터 수집 후 LeRobot 형식 변환 시 `pixi run` 명령어가 `pixi.toml`을 찾지 못해 발생하는 `CalledProcessError`를 해결함.

---

## 2. 상세 수정 사항

### A. 작업 경로(CWD) 명시 (`code/collect_data.py`)
- `subprocess.run`을 통해 `code/convert_to_lerobot.py`를 실행할 때, `cwd` 파라미터를 추가함.
- `PIXI_WS` (`ws_aic/src/aic/`) 디렉토리를 작업 경로로 지정하여 `pixi` 설정 파일을 정상적으로 인식하도록 함.

---

## 3. 기대 효과
- 에피소드 수집 완료 후 중단 없이 자동으로 LeRobot 데이터셋 변환 및 HuggingFace 업로드 프로세스가 진행됨.
- `pixi.toml` 위치와 상관없이 안정적으로 외부 스크립트 호출 가능.
