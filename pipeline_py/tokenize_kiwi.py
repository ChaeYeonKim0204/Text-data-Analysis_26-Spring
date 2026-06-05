# Colab에서 실행할 때만 아래 3줄 주석 해제 — 로컬/WSL에서는 그대로 두기
# from google.colab import drive
# drive.mount('/content/drive')

from pathlib import Path
import os, re, html, unicodedata
import pandas as pd

# Colab 기본 경로 — Drive 미마운트면 except로 빠져 로컬/WSL 경로 사용
# [PIPELINE patch] 머신고정 try/except 제거 → repo-상대 config (이식성)
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent)); import config as _cfg
PROJECT_DIR = _cfg.PROJECT_DIR
DATA_DIR = _cfg.DATA_DIR
os.chdir(PROJECT_DIR)

# 입력 전처리본 찾기 — 한글 파일명 NFC 정규화 후 매칭
def normalize_name(p): return unicodedata.normalize('NFC', p.name)
# [PIPELINE patch] 자동선택([-1]) → 고정 분석기간 정확 리터럴 파일명
_PINNED = '전처리_본문_언론사_260505_260511.csv'
INPUT_PATH = next(p for p in DATA_DIR.iterdir()
                  if p.is_file() and normalize_name(p) == _PINNED)
print('입력 전처리본:', INPUT_PATH.name)
df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')
print('행수:', len(df), '/ 컬럼:', list(df.columns))

# --- Kiwi와 불용어·핵심어 세팅 ---
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords

# typos 인자는 kiwipiepy 버전에 따라 init/analyze 위치가 달라 호환 위해 try
try:
    kiwi = Kiwi(typos='basic')   # 기본 오타 교정
except TypeError:
    kiwi = Kiwi()
stopwords = Stopwords()          # 내장 불용어

# 뉴스 상투어 불용어 시드 — 1차 LDA 결과 보고 계속 보강
NEWS_STOP = {
    '기자','특파원','논설위원','뉴스','보도','취재','기사','오늘','이날','어제','내일',
    '관련','위해','통해','지난','당시','이번','최근','대해','경우','정도','가운데','이후','현재','상황','모습',
    '사진','자료','제공','출처','무단','전재','재배포','배포','금지','연합뉴스','뉴스원','뉴시스',
    '뉴스1','연합뉴스TV','제보','구독','한겨레','조선일보','한국경제','매일경제','방송','앵커','리포트','종합','속보','단독','촬영','영상','편집',
    # 보강 — Kiwi가 한 토큰으로 만드는 합성 제작진어·매체/캡션 노이즈 (제작/구성/연출/디자인 같은 이중용도 낱말은 본문 과제거 우려로 제외)
    '영상편집','영상취재','촬영기자','자막그래픽','디지털뉴스부','뉴미디어부',
    '갈무리','일러스트','에디터','이데일리','자막뉴스','현장영상','취재파일','포토','서울앤','웨더아이',
    # 매체명 (Phase 1B 보강) — log-odds 그룹특화어 오염 정리, 한자/단일라틴 거름과 결합
    'KBS','MBC','SBS','YTN','라디오',
}
for w in NEWS_STOP:
    stopwords.add((w, 'NNG'))
stopwords_set = set(NEWS_STOP)
try:
    stopwords_set |= {w for w, _ in stopwords.stopwords}
except Exception:
    pass

# keep_terms — 분석에서 보존할 핵심 의제어(영문 약어·한자·고유표현). resources/keep_terms.txt 있으면 합침
KEEP_TERMS = {
    'AI','HMM','ETF','MOU','LNG','CEO','GDP','OLED','UAE','JWST','SNS','TV','IT','EU','UN',
    '美','韓','中','日','北','與','野','靑','尹','호르무즈','반도체','금리','환율','관세',
    '장동혁','박민식','정원오','오세훈','하정우',   # 6·3 지방선거 주요 인물(분해 방지) — 주간 갱신
}
kt_file = PROJECT_DIR / 'resources' / 'keep_terms.txt'
if kt_file.exists():
    KEEP_TERMS |= {ln.strip() for ln in kt_file.read_text(encoding='utf-8').splitlines() if ln.strip()}
keep_terms_set = set(KEEP_TERMS)
stopwords_set -= keep_terms_set                 # 핵심어가 불용어에 있으면 빼기
for term in keep_terms_set:
    kiwi.add_user_word(term, tag='NNG')         # 핵심어를 명사로 사용자 사전 등록 → 분해 방지
print('불용어:', len(stopwords_set), '/ keep_terms:', len(keep_terms_set))

# --- 전처리·토큰화 함수 ---
import emoji

# 명사류만 — 일반명사·고유명사·외국어(영문)·한자. 숫자(SN)·조사·어미·기호는 제외
KEEP_POS = {'NNG', 'NNP', 'SL', 'SH'}

def pre_clean(text):
    t = html.unescape(str(text))                # &amp; 같은 엔티티 복원
    t = emoji.replace_emoji(t, replace=' ')     # 이모지 제거
    return t

def tokenize_doc(text):
    toks = []
    for tok in kiwi.tokenize(pre_clean(text)):
        f = tok.form
        if f in keep_terms_set:                                  # 핵심 의제어는 무조건 보존
            keep = True
        elif tok.tag in KEEP_POS and len(f) >= 2 and f not in stopwords_set:
            keep = True
        else:
            keep = False
        if not keep:
            continue
        f = f.replace(' ', '_').strip('·:/.,“”\'"')   # 다어절 NNP 내부공백->_ (공백조인 저장 분해 방지), 양끝 기호 정리
        if f:
            toks.append(f)
    return toks

# 동작 확인
sample = "국민의힘 장동혁 대표가 美·韓 협력과 AI·반도체 정책을 강조했다 김재훈 기자 연합뉴스"
print('샘플 토큰:', tokenize_doc(sample))

# --- 전체 토큰화 + 저장 ---
try:
    from tqdm.auto import tqdm
    tqdm.pandas(desc='토큰화')
    tok_lists = df['text'].progress_map(tokenize_doc)
except Exception:
    tok_lists = df['text'].map(tokenize_doc)

df['tokens'] = tok_lists.map(lambda xs: ' '.join(xs))   # 공백으로 이어붙여 저장(재로딩 시 split 한 줄)
df['n_tokens'] = tok_lists.map(len)

out_cols = ['article_id', 'press', 'media_group', 'date', 'article_category', 'title_cleaned', 'n_tokens', 'tokens']
period = re.search(r'(\d{6}_\d{6})', normalize_name(INPUT_PATH)).group(1)
OUT_PATH = DATA_DIR / f'분석토큰_언론사_{period}.csv'
df[out_cols].to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
print('저장 완료:', OUT_PATH)
print('n_tokens 분포:'); print(df['n_tokens'].describe())

# --- 검증·현황 ---
from collections import Counter
print('=== 매체그룹별 평균 토큰 수 ===')
print(df.groupby('media_group')['n_tokens'].mean().round(1))
print('\n=== 전체 상위 빈도 단어 30 (불용어·상투어가 안 보여야 정상) ===')
allc = Counter()
for s in df['tokens']:
    allc.update(s.split())
for w, c in allc.most_common(30):
    print(f'  {w} {c}')
print('\n=== 토큰 샘플 3건 ===')
for s in df['tokens'].head(3):
    print(' ', s[:160])
