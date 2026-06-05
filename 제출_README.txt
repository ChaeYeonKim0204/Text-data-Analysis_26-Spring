텍데분 — 통신/언론사 네이버뉴스 텍스트 분석 (분석기간 2026-05-05 ~ 05-11)

[폴더 구성]
  수집한 데이터/
    1. 크롤링 결과/        네이버 뉴스 본문 통합 원자료
    2. 전처리·토큰화 결과/  정제 본문 + Kiwi 형태소 토큰
    3. 분석 결과/          LDA 토픽, 감성, 딥러닝 논조(DL), 제로샷 NLI 스탠스, 차트
    사전/                 KNU 감성사전(SentiWord_info.json)
  코드/
    노트북/               수집·전처리(press) + 분석(analysis) ipynb
    pipeline_py/          노트북과 '동일 작업'을 하는 모듈화 .py (재현용)
    PIPELINE_PLAN.md      설계·검증 문서
  발표.pdf

[재현 방법 — pipeline_py]
  1) Python 3.11 가상환경
  2) pip install -r 코드/pipeline_py/requirements.txt  (+ torch, transformers)
  3) HF 모델 2회 다운로드: snunlp/KR-FinBert-SC, pongjin/roberta_with_kornli
  4) 입력(통합본문)을 repo의 data/news/ 에 두고:  python pipeline_py/run_all.py
  ※ pipeline_py는 노트북을 thin-nbconvert + 경로패치만 한 것으로, 알고리즘·파라미터·시드 동일.
    preprocess/tokenize/lda/fulltext/absa는 value-identical 검증됨.
    dl_tone/nli_stance는 시드 고정(=42)·로컬 모델; GPU/torch 빌드에 따라 부동소수 미세차 가능.

[모델·환경]
  - LDA: gensim, K=6, random_state=42
  - 감성/논조: KNU 사전 ABSA + snunlp/KR-FinBert-SC (BERT 감성)
  - 스탠스: pongjin/roberta_with_kornli (Zero-shot NLI)
