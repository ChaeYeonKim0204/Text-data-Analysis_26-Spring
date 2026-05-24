# Text-data-Analysis_26-Spring

Jupyter notebook 기반의 뉴스 수집·전처리·토픽/감성 분석 저장소입니다. 현재 파이프라인은 두 축으로 나뉩니다.

- 통신 3사 키워드 수집 파이프라인: `SKT`, `SK텔레콤`, `KT`, `LG U+`, `LG유플러스`
- 언론사 뉴스 분석 파이프라인: 수집본을 전처리·토큰화한 뒤 LDA, 매체그룹 비교, 감성, 워드클라우드 분석

## 디렉터리 구조

- `notebooks/crawling/`: 수집, 통계, 전처리, 분석 노트북
- `data/crawling/`: 통신 3사 수집 중간산출물과 통계 파일
- `data/news/`: 언론사 분석용 입력 데이터
- `resources/`: 감성사전 등 공용 리소스
- `outputs/`: 분석 결과 CSV, PNG
- `docs/`: 진행상황, 전처리/토큰화 안내 문서

## 주요 흐름

통신 3사 수집 흐름:

`url_수집 -> 본문_수집_bs4 / 본문_수집 -> 본문_재시도_selenium -> 본문_통합`

언론사 분석 흐름:

`언론사_네이버뉴스_url_수집 -> 언론사_네이버뉴스_본문_수집_bs4 -> 언론사_네이버뉴스_전처리 -> 언론사_네이버뉴스_분석토큰화 -> 01~05 분석 노트북`

각 노트북은 파일 기반으로 이어지며, 별도 오케스트레이터는 없습니다. 보통 위에서 아래로 셀을 순서대로 실행합니다.

## 실행 환경

- 로컬: Jupyter/VS Code Notebook
- Colab: `_colab.ipynb` 노트북 사용
- 주요 패키지: `selenium`, `beautifulsoup4`, `requests`, `pandas`
- 분석 단계 추가 패키지: `kiwipiepy`, `gensim`, `matplotlib`, `wordcloud`

로컬에서 Selenium을 쓸 때는 Chrome이 필요합니다. 일부 노트북은 Colab Drive 경로와 로컬 경로를 둘 다 처리하도록 작성돼 있습니다.

## 작업 원칙

- 노트북 이름과 산출 파일명은 파이프라인 계약이므로 함부로 바꾸지 않습니다.
- 로직을 수정할 때는 로컬판과 `_colab` 변형이 함께 바뀌어야 하는지 확인합니다.
- `data/news/`, `outputs/`, 대용량 합본 CSV는 생성 산출물로 보고 기본적으로 Git에 올리지 않습니다.

자세한 작업 규칙은 [AGENTS.md](AGENTS.md), 현재 분석 진행은 [docs/분석_진행상황.md](docs/분석_진행상황.md), 전처리 규칙은 [docs/전처리_토큰화_안내.md](docs/전처리_토큰화_안내.md)를 참고하면 됩니다.
