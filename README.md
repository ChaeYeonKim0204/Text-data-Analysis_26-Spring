# Text-data-Analysis_26-Spring

뉴스 수집(Jupyter 노트북) + 팀 분석 코드 통합 파이프라인(py 모듈) 저장소

- **수집**: 네이버 뉴스 LPOD 기반 언론사별 기사 수집 노트북 (Colab 실행)
- **분석**: 팀원들의 분석 노트북을 모듈화해 한 번에 돌리는 **`pipeline_py/`** — 전처리 → 토큰화 → LDA → 전체텍스트 분석 → ABSA → 딥러닝 논조 → 제로샷 NLI 스탠스

## 디렉터리 구조

- `pipeline_py/`: **메인 분석 파이프라인** — `run_all.py` 한 번으로 전 단계 실행 (상세는 [pipeline_py/README.md](pipeline_py/README.md))
- `notebooks/crawling/press/`: 언론사 수집·전처리·토큰화 노트북 (수집은 Colab에서 실행)
- `notebooks/crawling/analysis/`: 01~05 분석 노트북 — 옛 트랙, 보존용 (현행 분석은 pipeline_py)
- `notebooks/crawling/_archive/`: 통신3사·방송사 직접 수집 등 폐기 트랙 보존
- `data/news/`: 분석 입력(통합 본문 CSV)과 파이프라인 중간·최종 CSV
- `resources/`: KNU 감성사전, NanumGothic 폰트 (파이프라인 번들)
- `docs/`: 진행상황, 전처리/토큰화 안내 문서

## 주요 흐름

수집 (Colab, 노트북):

`언론사_네이버뉴스_url_수집 → 언론사_네이버뉴스_본문_수집_bs4 (+마지막 셀에서 통합본 생성)`

분석 (로컬, py — 통합본 CSV 1개가 입력):

`preprocess → tokenize_kiwi → lda_topics → fulltext_analysis → absa_sentiment → dl_tone → nli_stance`

```bash
pip install -r pipeline_py/requirements.txt   # + HF 모델 2개 1회 다운로드(README 참고)
python pipeline_py/run_all.py
```

외국어 전용 기사(한글 0자, 연합뉴스 영문·일문 wire 82건)는 파이프라인이 자동 제외함 — 기준·검증은 pipeline_py/README.md 참고

## 실행 환경

- 수집 노트북: Colab (`_colab.ipynb`) — `selenium`, `beautifulsoup4`, `requests`, `pandas`
- 분석 파이프라인: Python 3.11 + `pipeline_py/requirements.txt` (정확 버전 핀) — GPU 있으면 자동 사용, 없으면 CPU
- 고정 분석기간 260505_260511 재현 전용

## 작업 원칙

- 노트북·산출 파일명은 파이프라인 계약이므로 함부로 바꾸지 않음
- 수집·통합은 Colab/Drive에서만 실행, 로컬 `data/news/`는 가져온 사본
- `data/news/` 산출물·대용량 CSV는 Git에 올리지 않음

자세한 작업 규칙은 [AGENTS.md](AGENTS.md), 진행상황은 [docs/분석_진행상황.md](docs/분석_진행상황.md), 전처리 규칙은 [docs/전처리_토큰화_안내.md](docs/전처리_토큰화_안내.md) 참고
