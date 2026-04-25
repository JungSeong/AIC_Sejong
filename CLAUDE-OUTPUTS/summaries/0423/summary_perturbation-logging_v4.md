# Pixi Run 스크립트 경로 에러 수정 요약 (절대 경로)

> 날짜: 2026-04-23
> 작성자: Gemini CLI
> 관련 코드: `code/collect_data.py`

---

## 1. 개요
`pixi run` 실행 시 작업 경로(`cwd`) 변경으로 인해 발생한 `No such file or directory` 에러를 해결함.

---

## 2. 상세 수정 사항

### A. 스크립트 경로 절대 경로화 (`code/collect_data.py`)
- `convert_to_lerobot.py`의 경로를 `Path.home() / "aic_sejong" / "code" / "convert_to_lerobot.py"`와 같이 절대 경로로 지정.
- `subprocess.run`에서 `cwd`가 `ws_aic/src/aic/`로 설정되어 있더라도, 프로젝트 루트에 있는 변환 스크립트를 정상적으로 호출 가능하도록 수정.

---

## 3. 기대 효과
- 데이터 수집부터 변환, 업로드까지의 전체 자동화 파이프라인이 경로 문제없이 안정적으로 동작함.
