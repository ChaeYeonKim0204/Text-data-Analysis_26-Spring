# Text-data-Analysis_26-Spring

뉴스 수집 노트북과 텍스트 분석 파이프라인 정리 저장소

- `notebooks/crawling/press/`: 네이버 뉴스 언론사별 수집 노트북
- `pipeline_py/`: 수집 결과를 전처리하고 분석 결과를 만드는 파이썬 파이프라인
- `resources/`: 감성사전과 한글 폰트 등 실행에 필요한 보조 파일

실행 방법: [pipeline_py/README.md](pipeline_py/README.md)

## 디렉터리 구조

- `pipeline_py/`: 전처리, 토큰화, LDA, 전체 텍스트 분석, ABSA, 딥러닝 논조, NLI 스탠스 분석 코드
- `notebooks/crawling/press/`: 언론사별 URL 수집, 본문 수집, 전처리, 토큰화 노트북
- `notebooks/crawling/analysis/`: 기존 분석 노트북 보관
- `notebooks/crawling/_archive/`: 사용하지 않는 이전 수집 트랙 보관
- `data/news/`: 통합 본문 CSV와 파이프라인 실행 결과가 놓이는 위치
- `resources/`: KNU 감성사전, NanumGothic 폰트
- `docs/`: 진행상황과 전처리/토큰화 설명 문서

## 주요 흐름

수집: Colab 노트북

`언론사_네이버뉴스_url_수집 → 언론사_네이버뉴스_본문_수집_bs4 (+마지막 셀에서 통합본 생성)`

분석 입력: 통합 본문 CSV

`preprocess → tokenize_kiwi → lda_topics → fulltext_analysis → absa_sentiment → dl_tone → nli_stance`

```bash
pip install -r pipeline_py/requirements.txt   # + HF 모델 2개 1회 다운로드(README 참고)
python pipeline_py/run_all.py
```

외국어 전용 기사 등 분석 제외 기준: `pipeline_py/README.md`

## 실행 환경

- 수집 노트북: Colab, `selenium`, `beautifulsoup4`, `requests`, `pandas`
- 분석 파이프라인: Python 3.11, `pipeline_py/requirements.txt`
- 분석 기간: `260505_260511`

## 작업 원칙

- 노트북과 산출 파일명 유지
- 수집 결과 위치: `data/news/`
- 분석 코드 위치: `pipeline_py/`
- 대용량 CSV와 생성 산출물은 Git 제외

진행상황: [docs/분석_진행상황.md](docs/분석_진행상황.md)  
전처리 규칙: [docs/전처리_토큰화_안내.md](docs/전처리_토큰화_안내.md)
