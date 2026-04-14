# Working Rules

## General

- 모든 응답은 한국어로
- 코드 수정 시 변경된 부분만 명시
- 불필요한 설명은 생략하고, 핵심말 말할 것

## File Handling

- 출력 파일은 항상 CLAUDE-OUTPUTS/ 에 저장
- 기존 파일 덮어쓰기 전 확인
- 폴더 없으면 자동 생성

## Output Folder Structure & Naming Rule

| 폴더 | 용도 |
|------|------|
| `CLAUDE-OUTPUTS/summaries/<오늘_날짜>` | 문서 요약 |
| `CLAUDE-OUTPUTS/reviews/<오늘_날짜>` | 코드 리뷰 |
| `CLAUDE-OUTPUTS/debug/<오늘_날짜>` | 디버깅 분석 |
| `CLAUDE-OUTPUTS/notes/<오늘_날짜>` | 자유 메모 |

- 오늘 날짜 디렉토리는 'mmdd'의 형식으로 작성

## File Naming Rule

- 형식: `[type]_[subject]_v[number].md`
- 예시: `summary_aic-docs_v1.md`

## Documentation

- 가장 처음에 전체 내용을 2-3줄로 요약하여 정리할 것
- 마크다운 형식 준수
- 중요한 문장은 **볼드** 처리
- 핵심 단어는 <u>밑줄</u> 처리
- 섹션 간 `---` 구분선 사용
- 표(table)로 구조화할 수 있는 내용은 반드시 표로
- 코드 블록에 언어 명시
- 수식은 LaTEX로 깔끔하게
- 섹션 구분 명확히

## Output Format 예시

| 항목 | 내용 |
|------|------|
| Trial 1 | **정책 수렴 검증** + <u>NIC 랜덤 위치</u> 대응 |

## Code

- 기존 코드 스타일 유지
- 코드 수정 시 변경 이유 주석으로 명시
