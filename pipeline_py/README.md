# 텍데분 분석 파이프라인 (모듈화 py)

팀 분석 노트북들을 **동일 작업 수행** py로 모듈화한 것. 설계·검증 근거는 repo 루트 `PIPELINE_PLAN.md`(서브에이전트3+codex3 검토 6라운드 수렴).
**원칙: 알고리즘·파라미터·시드는 노트북과 byte/value-동일. 바꾼 건 경로(config)·폰트·노트북전용 라인 제거뿐.**

## 구성
```
pipeline_py/
  config.py            # 중앙 경로(repo-상대 자동도출 → scp/이동해도 동작)
  preprocess.py        # 통합_본문 → 전처리_본문(+hangul)
  tokenize_kiwi.py     # 전처리_본문 → 분석토큰 (Kiwi)
  lda_topics.py        # 분석토큰 → analysis_언론사_lda결과_재수집.csv (gensim)
  fulltext_analysis.py # 전처리_본문+분석토큰+KNU → F01~F15 차트(11) (sklearn/networkx/wordcloud)
  absa_sentiment.py    # 전처리_본문+KNU → A01~A03 차트
  dl_tone.py           # 전처리_본문 → DL_문장별_논조분석.csv 등 (snunlp/KR-FinBert-SC)  *모델필요
  nli_stance.py        # DL_문장별 → NLI_문장별_스탠스분석.csv 등 (pongjin/roberta_with_kornli)  *모델필요
  run_all.py           # 외부 서브프로세스 오케스트레이터(clean-before-run, 순서대로)
  requirements.txt
```
입력은 `<repo>/data/news/`의 `통합_본문_bs4_언론사_260505_260511.csv`. KNU 사전·NanumGothic은 `<repo>/resources/`에 번들(scp로 함께 이동).

## 실행
```bash
# 1회: pip install -r pipeline_py/requirements.txt  (+ torch/transformers, HF모델 2개 — requirements.txt 주석 참고)
python pipeline_py/run_all.py        # 전체
python pipeline_py/preprocess.py     # 개별도 가능
```

## 검증 상태(로컬 CPU)
- preprocess / tokenize_kiwi : **value-identical** (전 컬럼·tokens 완전일치)
- lda_topics : dominant_topic **100% 일치**, topic_prob 차 ~1.8e-5 (gensim/numpy 버전 부동소수 허용오차)
- fulltext / absa : 차트 정상 생성, 핵심 집계 재현(의제기사 1,796 일치, 분류정확도 83.1%) — PNG는 폰트 cross-OS라 픽셀불일치 허용(데이터 동일)
- dl_tone / nli_stance : 코드 완성(경로패치). **실행·검증은 학교 GPU에서**(torch+모델 필요). `device=-1`(CPU)는 동일재현용 verbatim 유지 — GPU로 바꾸면 부동소수 미세차 가능(§9).

## 고정 분석기간
이 파이프라인은 **260505_260511** 한 주를 재현. 다른 주차는 범위 밖(원 노트북 재파라미터화 필요).
