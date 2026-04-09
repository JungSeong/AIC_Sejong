# Basic Templates

Claude에게 질의할 시 아래의 형식을 지켜서 질의하면 됩니다

## 문서 요약 요청

ws_aic/src/aic/docs/ 에 있는 하위 파일들 읽고
CLAUDE-OUTPUTS/summaries/[날짜]/[주제]_[번호].md로 정리해줘

## 코드 리뷰 요청

[파일경로] 리뷰하고
CLAUDE-OUTPUTS/reviews/[날짜]/[주제]_[번호].md로 저장해줘

## 디버깅 요청

에러 분석하고
CLAUDE-OUTPUTS/debug/[날짜]/[주제]_[번호].md로 저장해줘

