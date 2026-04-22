---
name: paper-research
description: |
  ML/로보틱스 선행 연구 조사 및 요약 스킬.
  사용자가 연구 주제나 키워드를 말하면 자동으로 관련 논문과 기술 블로그를 검색하고,
  PDF를 다운로드하여 Figure를 crop하고, working-rules 형식의 마크다운 요약을 생성한다.
  "논문 찾아줘", "선행 연구 조사", "관련 논문 요약", "survey" 등의 표현에 반드시 이 스킬을 사용할 것.
  AIC 프로젝트(케이블 삽입 로봇) 연관성 섹션을 항상 포함한다.
---

# Paper Research Skill

## 개요

사용자가 주제를 주면 아래 순서로 수행한다:

1. 논문 검색 및 선별
2. PDF 다운로드
3. Figure crop → PNG 저장
4. working-rules 형식 마크다운 요약 생성
5. 기술 블로그 조사 및 요약

---

## Step 1. 논문 검색 및 선별

`paper_search` MCP 도구로 검색한다. 쿼리는 영어로 변환해 사용한다.

```
검색 수: 최대 12개
선별 기준:
  - 주제 직접성 (핵심 키워드 포함 여부)
  - 인용 가능성 (venue: NeurIPS / ICRA / IROS / CoRL / RSS / arXiv)
  - AIC 프로젝트 연관성 (케이블 삽입, 로봇 조작, 비전-힘 제어, 모방학습 등)
최종 선별: 관련성 높은 3~5편
```

검색 후 선별된 논문 목록을 사용자에게 먼저 보여주고 확인을 받는다.
사용자가 "다 해줘" 또는 별도 지시가 없으면 자동으로 진행한다.

---

## Step 2. PDF 다운로드

선별된 논문마다 아래 경로에 PDF를 저장한다:

```
저장 경로: ~/aic_sejong/paper/<slug>.pdf
slug 규칙: 논문 제목을 소문자-하이픈 형식으로 변환
  예) "Vision-Force Fused Curriculum Learning" → vision-force-curriculum.pdf
```

arXiv 논문 다운로드:

```bash
wget -O ~/aic_sejong/paper/<slug>.pdf https://arxiv.org/pdf/<arxiv_id>.pdf
```

다운로드 실패 시 웹 검색으로 대체 URL을 찾는다.

---

## Step 3. Figure Crop → PNG 저장

PDF에서 핵심 Figure를 추출해 PNG로 저장한다.

저장 경로:

```
~/aic_sejong/CLAUDE-OUTPUTS/summaries/<mmdd>/figures/<slug>/
  fig1_architecture.png
  fig2_results.png
```

Python 추출 코드:

```python
from pdf2image import convert_from_path
from PIL import Image
import os

pdf_path = "/home/swlinux/aic_sejong/paper/<slug>.pdf"
out_dir  = "/home/swlinux/aic_sejong/CLAUDE-OUTPUTS/summaries/<mmdd>/figures/<slug>"
os.makedirs(out_dir, exist_ok=True)

pages = convert_from_path(pdf_path, dpi=150)

# 논문을 보고 Figure가 있는 페이지와 crop 영역 지정
# crop: (left, upper, right, lower) 픽셀 단위
page = pages[<page_index>]   # 0-based
w, h = page.size
cropped = page.crop((0, 0, w, h // 2))  # 예: 상단 절반
cropped.save(f"{out_dir}/fig1_architecture.png")
```

선택 기준:

- 전체 아키텍처 Figure (필수)
- 정량 결과 Table 또는 핵심 그래프
- 방법론 다이어그램

---

## Step 4. 마크다운 요약 작성 (논문)

저장 경로:

```
~/aic_sejong/CLAUDE-OUTPUTS/summaries/<mmdd>/summary_<slug>.md
```

반드시 아래 working-rules 형식을 따를 것:

```markdown
# summary: <논문 제목>

> <저자 et al., venue year. arXiv ID 또는 DOI>

**<전체 내용 2~3줄 요약 — 핵심 기여와 성과를 볼드로>**

---

## 1. Introduction
...

---

## 2. Method

### Figure 1 — <Figure 제목>

![Figure 1](figures/<slug>/fig1_architecture.png)

> <Figure에 대한 한국어 설명>

---

### 수식이 있는 경우

$$<LaTeX 수식>$$

> <수식 의미를 쉬운 한국어로. 각 변수가 무엇인지 풀어서 설명>

---

## 3. Experiment

| 데이터셋 | 방법 | 성능 | 비고 |
|---------|------|------|------|

---

## 4. Conclusion
...

---

## AIC 프로젝트 연관성

| 이 논문 | 우리 프로젝트 적용 가능성 |
|---------|----------------------|
| ...     | ...                  |

> **참고할 핵심 아이디어**: <구체적인 적용 방안 1~2줄>
```

형식 규칙:

- 중요 문장: **볼드**
- 핵심 단어: <u>밑줄</u>
- 섹션 간 `---` 구분선
- 표로 구조화 가능한 내용은 반드시 표로
- 수식은 LaTeX, 아래에 한국어 설명 필수
- Figure는 crop한 PNG를 상대 경로로 참조

---

## Step 5. 기술 블로그 조사

논문 조사와 병행하여, 동일 주제에 대한 기술 블로그/실전 구현 사례를 웹 검색한다.

검색 방법:

```
WebSearch 쿼리 예시:
  - "<주제> implementation blog site:medium.com OR site:towardsdatascience.com"
  - "<주제> tutorial github"
  - "<주제> 구현 후기 site:zenn.dev OR site:qiita.com"
  - "<주제> practical guide 2023 OR 2024"
```

선별 기준:

- 실제 구현 코드 포함 여부
- 논문과 다른 실용적 인사이트 제공 여부
- AIC 프로젝트에 직접 적용 가능한 팁 포함 여부

저장 경로:

```
~/aic_sejong/CLAUDE-OUTPUTS/summaries/<mmdd>/blogs_<topic-slug>.md
```

블로그 요약 형식:

```markdown
# 기술 블로그 조사: <주제>

> 조사일: <날짜>

---

## 1. <블로그 제목>

> 출처: [<제목>](<URL>) — <저자/플랫폼>, <연도>

**요약**: <2~3줄 핵심 내용>

### 핵심 인사이트
- <논문에서 다루지 않는 실전 팁>
- <구현 시 주의사항>

### AIC 적용 가능성
> <우리 프로젝트에 바로 쓸 수 있는 부분>

---

## 2. <다음 블로그>
...

---

## 종합 정리

| 출처 | 핵심 아이디어 | AIC 적용 가능성 |
|------|------------|--------------|
| ...  | ...        | ...          |
```

---

## 전체 실행 체크리스트

- [ ] paper_search 검색 완료 (최대 12개)
- [ ] 관련 논문 최대 3~5편 선별, 사용자 확인
- [ ] PDF 다운로드 → paper/<slug>.pdf
- [ ] Figure crop → summaries/<mmdd>/figures/<slug>/
- [ ] 마크다운 요약 → summaries/<mmdd>/summary_<slug>.md
- [ ] AIC 프로젝트 연관성 섹션 포함
- [ ] 기술 블로그 WebSearch 조사 (2~3건)
- [ ] 블로그 요약 → summaries/<mmdd>/blogs_<topic-slug>.md
- [ ] 완료 후 git diff 실행하여 변경 내역 표시
