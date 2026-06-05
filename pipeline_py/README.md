# 텍데분 분석 파이프라인 (모듈화 py)

팀 분석 노트북들을 **동일 작업 수행** py로 모듈화한 end-to-end 파이프라인. 설계·검증 근거는 repo 루트 `PIPELINE_PLAN.md`(서브에이전트3+codex3 검토 6라운드 수렴).
**원칙: 알고리즘·파라미터·시드는 팀 노트북과 byte/value-동일.** 바꾼 건 경로(config)·폰트·노트북전용 라인 제거, 그리고 팀 결정으로 추가한 **외국어 전용 기사 제외** 하나뿐.

## 구성
```
pipeline_py/
  config.py            # 중앙 경로(repo-상대 자동도출 → scp/이동해도 동작)
  preprocess.py        # 통합_본문 → 전처리_본문(+hangul_chars/ratio, is_foreign 플래그)
  tokenize_kiwi.py     # 전처리_본문 → 분석토큰 (Kiwi) — 외국어 전용 기사 제외 지점
  lda_topics.py        # 분석토큰 → analysis_언론사_lda결과_재수집.csv (gensim K=6)
  fulltext_analysis.py # 전처리_본문+분석토큰+KNU → F01~F15 차트(11) (sklearn/networkx/wordcloud)
  absa_sentiment.py    # 전처리_본문+KNU → A01~A03 차트
  dl_tone.py           # 전처리_본문 → DL_문장별_논조분석.csv 등 (snunlp/KR-FinBert-SC)  *모델필요
  nli_stance.py        # DL_문장별 → NLI_문장별_스탠스분석.csv 등 (pongjin/roberta_with_kornli)  *모델필요
  run_all.py           # 외부 서브프로세스 오케스트레이터(clean-before-run, 순서대로)
  requirements.txt     # 검증된 정확 버전 핀(torch/transformers 포함)
```
입력은 `<repo>/data/news/`의 `통합_본문_bs4_언론사_260505_260511.csv` 1개. KNU 사전·NanumGothic은 `<repo>/resources/`에 번들.

## 외국어 전용 기사 제외 (팀 결정)
- `preprocess`가 `is_foreign = (한글 글자 수 == 0)` 플래그 산출 — 한글이 1자라도 있으면 절대 안 걸림(over-removal 구조적 차단)
- `tokenize_kiwi` 입력에서 제외(11,990 → 11,908) → 분석토큰·LDA 자동 상속, fulltext/absa/dl_tone은 필터에 `~is_foreign` 명시
- 실측(260505_260511): 연합뉴스 영문 52 + 일문 30 = 82건, 전부 한글 0자. DL/NLI는 한글 키워드(`호르무즈|이란`) 게이트라 원래 미포함 — 결과 불변
- 임계 설계는 codex+서브에이전트 2라운드 검토 수렴(비율 0.2 대신 0자 기준 — 보존 기사 최저 한글비율 0.2444와 충돌 없음)

## 실행
```bash
# 1회 준비
pip install -r pipeline_py/requirements.txt
python -c "from transformers import AutoTokenizer,AutoModelForSequenceClassification as M; \
  [ (AutoTokenizer.from_pretrained(x), M.from_pretrained(x)) for x in \
    ('snunlp/KR-FinBert-SC','pongjin/roberta_with_kornli') ]"   # HF 모델 2개 캐시(~1.8G)

python pipeline_py/run_all.py        # 전체 one-shot
python pipeline_py/preprocess.py     # 개별 실행도 가능
```
GPU 있으면 dl_tone/nli_stance가 자동으로 CUDA 사용, 없으면 CPU로 동작(결과 라벨 동일, 확률 미세차만).

## 검증 상태 (2026-06-05/06 완료)
- **run_all one-shot 완주 확인** — 전처리 11,990(외국어 플래그 82) → 토큰 11,908 → LDA 11,908 → 차트 14 → DL 1,796기사/10,815문장 → NLI 10,214쌍
- preprocess / tokenize_kiwi : 정본 노트북과 **value-identical** (정제 정규식 35개 byte-identical, 출력 불변량 재계산 일치)
- lda / fulltext / absa : frozen params 전부 일치(gensim K6·rs42, TF-IDF 20000·min_df3, LogReg C5, KNU min/max-|pol| 정책 구분 유지)
- dl_tone / nli_stance : **팀 ref 대비 라벨 100% 일치**(DL dl_sentiment 10,815/10,815, NLI stance_pred 단일행위자 4,951/4,951), 확률차 ≤1.8e-5
- 최종 교차검증: 5차원(체인무결성/1단계충실도/분석3종/DL·NLI재현/완전성) × codex+서브에이전트 쌍 = **10인 전원 CRITICAL 없음**

## 참고
- transformers 5 호환: nli_stance가 `return_token_type_ids=False` 사용 — kornli RoBERTa(type_vocab_size=1)에서 slow tokenizer의 0/1 쌍 id 크래시 방지, fast tokenizer 원동작(전부 0)과 동일
- ref/NLI(4,951행)는 행위자 explode 이전의 옛 산출 — 현행 파이프라인 10,214쌍의 단일행위자 부분집합과 100% 일치
- 고정 분석기간 **260505_260511** 재현 전용. 다른 주차는 범위 밖(원 노트북 재파라미터화 필요)
