# 텍데분 분석 파이프라인 — 실행 가이드

## 이게 뭐냐면

팀원들이 각자 만든 분석 노트북 7개(전처리 → 토큰화 → LDA → 전체텍스트 분석 → ABSA 감성 → 딥러닝 논조 → NLI 스탠스)를 **파이썬 파일로 합쳐놓은 것**. 노트북을 하나하나 열어서 셀을 누를 필요 없이, **명령어 한 줄이면 전 과정이 자동으로** 돎

```
[입력]  통합_본문_bs4_언론사_260505_260511.csv  (기사 11,990건짜리 파일 1개)
   ↓   python pipeline_py/run_all.py  ← 이거 한 줄
[출력]  분석 CSV 4개 + 차트 PNG 16개 + 요약 문서 2개
```

결과는 원래 노트북이 내던 것과 동일함을 검증 완료 (딥러닝/NLI는 팀 산출물과 100% 일치 확인). 달라진 건 딱 하나 — 팀 결정대로 **외국어로만 된 기사 82건(연합뉴스 영문·일문 기사)을 자동으로 빼고** 분석함

---

## 준비물

| 준비물 | 설명 |
|---|---|
| 컴퓨터 | Windows/Mac/리눅스 아무거나. **GPU 없어도 됨** (있으면 더 빠를 뿐) |
| Python 3.11 | 아래 1단계에서 설치 |
| 입력 CSV 1개 | `통합_본문_bs4_언론사_260505_260511.csv` (약 29MB) — 팀 드라이브에 있음 |
| 인터넷 | 처음 1회만 필요 (패키지·AI모델 다운로드, 총 7GB 정도) |
| 디스크 여유 | 10GB 정도 |

---

## 따라하기 (처음부터 끝까지)

### 1단계 — Python 3.11 준비

이미 아나콘다(Anaconda/Miniconda)가 있다면 터미널(Windows는 Anaconda Prompt)에서:

```bash
conda create -n news-analysis python=3.11 -y
conda activate news-analysis
```

아나콘다가 없다면 https://www.anaconda.com/download 에서 설치 후 위 명령 실행
(`conda activate news-analysis`는 **터미널을 새로 열 때마다** 다시 입력해야 함)

### 2단계 — 이 저장소 받기

```bash
git clone https://github.com/ChaeYeonKim0204/Text-data-Analysis_26-Spring.git
cd Text-data-Analysis_26-Spring
```

git이 없으면 GitHub 페이지의 초록색 `Code` 버튼 → `Download ZIP` 받아서 압축 풀고, 터미널에서 그 폴더로 이동(`cd 폴더경로`)해도 똑같음

### 3단계 — 필요한 패키지 설치 (1회, 5~15분)

```bash
pip install -r pipeline_py/requirements.txt
```

설치 목록에 버전이 박혀 있어서 누가 깔아도 같은 환경이 만들어짐. 빨간 글씨 에러 없이 끝나면 성공

### 4단계 — AI 모델 2개 다운로드 (1회, 약 1.8GB)

딥러닝 논조 분석과 NLI 스탠스 분석이 쓰는 모델. 아래를 통째로 복사해서 실행:

```bash
python -c "from transformers import AutoTokenizer,AutoModelForSequenceClassification as M; [ (AutoTokenizer.from_pretrained(x), M.from_pretrained(x)) for x in ('snunlp/KR-FinBert-SC','pongjin/roberta_with_kornli') ]"
```

진행바가 쭉 올라가다가 조용히 끝나면 성공. 모델은 내 컴퓨터에 저장되니 다음부터는 인터넷 없어도 됨

### 5단계 — 입력 데이터 넣기

팀 드라이브에서 `통합_본문_bs4_언론사_260505_260511.csv`를 받아서 **`data/news/` 폴더 안에** 넣음:

```
Text-data-Analysis_26-Spring/
└── data/
    └── news/
        └── 통합_본문_bs4_언론사_260505_260511.csv   ← 여기!
```

파일명을 바꾸면 안 됨 (파이프라인이 이 이름 그대로 찾음)

### 6단계 — 실행!

```bash
python pipeline_py/run_all.py
```

7단계가 순서대로 돌면서 화면에 진행 상황이 출력됨. 중간에 멈춰 보여도 기다리면 됨 (딥러닝 단계가 제일 오래 걸림). 마지막에 이 줄이 보이면 끝:

```
파이프라인 완료.
```

**소요 시간**: GPU 있으면 30분 안쪽, CPU만 있으면 1~2시간 정도. GPU는 알아서 감지하니 따로 설정할 거 없음

---

## 실행하면 뭐가 나오나

| 위치 | 파일 | 내용 |
|---|---|---|
| `data/news/` | 전처리_본문_언론사_*.csv | 기사 본문 정제 + 플래그 (11,990건) |
| `data/news/` | 분석토큰_언론사_*.csv | 형태소 분석된 단어들 (11,908건) |
| `data/news/` | analysis_언론사_lda결과_재수집.csv | 기사별 LDA 토픽 배정 (토픽 6개) |
| `pipeline_py/산출물_차트/` | F01~F15.png (11장) | 전체텍스트 분석 차트 (빈도·워드클라우드·감성·분류 등) |
| `pipeline_py/산출물_차트/` | A01~A03.png (3장) | ABSA 감성 차트 (그룹별 국가/의제 감성) |
| `pipeline_py/딥러닝_논조분석_산출물/` | CSV 4 + PNG 2 + 요약 md | 호르무즈 기사 문장별 딥러닝 논조 |
| `pipeline_py/제로샷NLI_스탠스분석_산출물/` | CSV 4 + PNG 1 + 요약 md | 미국/이란/이스라엘 스탠스 분석 |

다시 실행하면 이전 산출물을 알아서 지우고 새로 만드니 그냥 또 돌리면 됨

---

## 막혔을 때 (FAQ)

**Q. `python: command not found` 또는 `conda: command not found`**
→ 1단계의 가상환경 활성화(`conda activate news-analysis`)를 안 한 상태. 터미널 새로 열었으면 다시 입력

**Q. `FileNotFoundError: 통합_본문...csv`**
→ 5단계 확인 — 파일이 `data/news/` 안에, 정확히 그 이름으로 있어야 함

**Q. 모델 다운로드에서 멈춤/실패**
→ 인터넷 연결 확인 후 4단계 명령을 다시 실행 (받다 만 부분부터 이어받음)

**Q. 딥러닝 단계가 너무 느림**
→ CPU만 있으면 원래 오래 걸림 (1시간 이상 정상). 끄지 말고 기다리거나 GPU 있는 컴퓨터에서 실행

**Q. 차트의 글자가 □□□로 깨짐**
→ 결과 숫자에는 영향 없음. `resources/NanumGothic.ttf`가 같이 받아졌는지 확인

**Q. 중간에 빨간 에러로 멈췄어요**
→ 에러 메시지 마지막 10줄 정도를 캡처해서 채연한테 보내주면 됨

---

## 자주 안 물어봐도 되는 기술 상세

<details>
<summary>펼쳐보기 (수정하려는 사람용)</summary>

- **원칙**: 알고리즘·파라미터·시드는 원본 팀 노트북과 동일 (nbconvert 변환 + 경로만 `config.py`로 중앙화). 설계 문서는 repo 루트 `PIPELINE_PLAN.md`
- **외국어 기사 제외**: `preprocess`가 `is_foreign = (한글 글자 수 == 0)` 플래그 생성 → `tokenize` 입력에서 제외(11,990→11,908, LDA 자동 상속), fulltext/absa/dl_tone은 필터에 `~is_foreign` 명시. 한글이 1자라도 있으면 절대 안 걸리는 기준이라 멀쩡한 기사가 지워질 위험 없음. codex+서브에이전트 2라운드 검토 수렴
- **검증 상태**: run_all one-shot 완주 / 전처리·토큰화 노트북과 value-identical(정규식 35개 byte 동일) / LDA·fulltext·absa frozen params 일치 / DL·NLI 팀 ref 대비 라벨 100%·확률차 ≤1.8e-5 / 최종 5차원×(codex+서브에이전트) 10인 교차검증 전원 CRITICAL 없음
- **transformers 5 호환**: nli_stance가 `return_token_type_ids=False` 사용 — kornli RoBERTa(type_vocab_size=1)에서 slow tokenizer가 문장쌍에 0/1을 부여해 크래시하는 문제 방지. fast tokenizer 원동작(쌍 모두 0)과 의미 동일
- **ref/NLI 행수 차이**: 팀 공유본(4,951행)은 행위자 explode 이전 버전 — 현행 10,214쌍의 단일행위자 부분집합과 100% 일치
- **고정 분석기간**: 260505_260511 전용. 다른 주차는 각 모듈의 파일명 리터럴·기간 상수 수정 필요
- CPU 전용 torch 설치: `pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu`

</details>
