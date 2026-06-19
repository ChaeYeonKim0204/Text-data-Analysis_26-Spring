# -*- coding: utf-8 -*-
"""
② 분석 토큰화 — 전처리한 기사 내용을 Kiwi로 나눈 뒤 분석용 단어 생성. 외국어 전용 기사 제외, LDA 입력 자동 제외
읽는 파일: data/news/전처리_본문_언론사_260505_260511.csv
저장 파일: data/news/분석토큰_언론사_260505_260511.csv (11,908건 — 외국어 82건 제외, tokens는 단어를 띄어쓰기로 이어 붙여 저장)
"""
# Colab에서 실행할 때만 아래 3줄 주석 해제 — 로컬/WSL에서는 그대로 두기
# from google.colab import drive
# drive.mount('/content/drive')

from pathlib import Path
import os, re, html, unicodedata
import pandas as pd

# 같은 폴더의 설정 파일 불러오기
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent)); import config as _cfg
PROJECT_DIR = _cfg.PROJECT_DIR
DATA_DIR = _cfg.DATA_DIR
os.chdir(PROJECT_DIR)   # 프로젝트 폴더 기준 파일 읽기·저장

# 입력 전처리본 찾기
def normalize_name(p): return unicodedata.normalize('NFC', p.name)
# 이번 분석 기간 전처리본만 사용
_PINNED = '전처리_본문_언론사_260505_260511.csv'
INPUT_PATH = next(p for p in DATA_DIR.iterdir()
                  if p.is_file() and normalize_name(p) == _PINNED)
print('입력 전처리본:', INPUT_PATH.name)
df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')   # 한글 CSV 파일 불러오기
print('행수:', len(df), '/ 컬럼:', list(df.columns))

# 외국어 전용 기사 제외(팀 결정) — 분석토큰에서 빠지므로 LDA도 자동 제외
before_foreign = len(df)
df = df[~df['is_foreign']].copy().reset_index(drop=True)   # 제외 후 번호를 다시 정리
print(f'외국어 전용 기사 제외: {before_foreign:,} → {len(df):,}')

# Kiwi와 불용어·핵심어 세팅
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords

# Kiwi 설정 — 가능하면 기본 오타 보정 사용
try:
    kiwi = Kiwi(typos='basic')   # 기본 오타 교정
except TypeError:
    kiwi = Kiwi()
stopwords = Stopwords()          # 내장 불용어

# 뉴스 상투어 불용어 기본 목록 — 1차 LDA 결과 반영해 계속 보강
NEWS_STOP = {
    '기자','특파원','논설위원','뉴스','보도','취재','기사','오늘','이날','어제','내일',
    '관련','위해','통해','지난','당시','이번','최근','대해','경우','정도','가운데','이후','현재','상황','모습',
    '사진','자료','제공','출처','무단','전재','재배포','배포','금지','연합뉴스','뉴스원','뉴시스',
    '뉴스1','연합뉴스TV','제보','구독','한겨레','조선일보','한국경제','매일경제','방송','앵커','리포트','종합','속보','단독','촬영','영상','편집',
    # 제작진·캡션에서 자주 나오는 단어
    '영상편집','영상취재','촬영기자','자막그래픽','디지털뉴스부','뉴미디어부',
    '갈무리','일러스트','에디터','이데일리','자막뉴스','현장영상','취재파일','포토','서울앤','웨더아이',
    # 매체명
    'KBS','MBC','SBS','YTN','라디오',
}
for w in NEWS_STOP:
    stopwords.add((w, 'NNG'))
stopwords_set = set(NEWS_STOP)
try:
    stopwords_set |= {w for w, _ in stopwords.stopwords}
except Exception:
    pass

# 분석에서 꼭 남길 단어
# 영어 약어, 한자, 주요 인물명처럼 잘게 쪼개지면 의미가 약해지는 단어를 넣음
KEEP_TERMS = {
    'AI','HMM','ETF','MOU','LNG','CEO','GDP','OLED','UAE','JWST','SNS','TV','IT','EU','UN',
    '美','韓','中','日','北','與','野','靑','尹','호르무즈','반도체','금리','환율','관세',
    '장동혁','박민식','정원오','오세훈','하정우',   # 6·3 지방선거 주요 인물
}
kt_file = PROJECT_DIR / 'resources' / 'keep_terms.txt'
if kt_file.exists():
    KEEP_TERMS |= {ln.strip() for ln in kt_file.read_text(encoding='utf-8').splitlines() if ln.strip()}
keep_terms_set = set(KEEP_TERMS)
stopwords_set -= keep_terms_set                 # 핵심어가 불용어에 있으면 제외
for term in keep_terms_set:
    kiwi.add_user_word(term, tag='NNG')         # 핵심어 분절 방지 등록
print('불용어:', len(stopwords_set), '/ keep_terms:', len(keep_terms_set))

# 전처리·토큰화 함수
import emoji

# 분석에 쓸 단어만 남김 — 명사, 영문, 한자는 남기고 숫자·조사·기호는 제외
KEEP_POS = {'NNG', 'NNP', 'SL', 'SH'}

def pre_clean(text):
    t = html.unescape(str(text))                # &amp;처럼 깨져 보이는 글자를 원래 문자로 바꿈
    t = emoji.replace_emoji(t, replace=' ')     # 이모지 제거
    return t

def tokenize_doc(text):
    toks = []
    for tok in kiwi.tokenize(pre_clean(text)):
        f = tok.form
        if f in keep_terms_set:                                  # 꼭 남길 단어는 무조건 남김
            keep = True
        elif tok.tag in KEEP_POS and len(f) >= 2 and f not in stopwords_set:   # 한 글자 단어와 불용어 제외
            keep = True
        else:
            keep = False
        if not keep:
            continue
        f = f.replace(' ', '_').strip('·:/.,“”\'"')   # 단어 양끝의 기호 정리
        if f:
            toks.append(f)
    return toks

# 전체 토큰화 + 저장
try:
    from tqdm.auto import tqdm   # 설치되어 있으면 진행률 표시
    tqdm.pandas(desc='토큰화')
    tok_lists = df['text'].progress_map(tokenize_doc)
except Exception:
    tok_lists = df['text'].map(tokenize_doc)

df['tokens'] = tok_lists.map(lambda xs: ' '.join(xs))   # 토큰을 공백으로 이어 붙여 저장
df['n_tokens'] = tok_lists.map(len)

out_cols = ['article_id', 'press', 'media_group', 'date', 'article_category', 'title_cleaned', 'n_tokens', 'tokens']
period = re.search(r'(\d{6}_\d{6})', normalize_name(INPUT_PATH)).group(1)
OUT_PATH = DATA_DIR / f'분석토큰_언론사_{period}.csv'
df[out_cols].to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
print('저장 완료:', OUT_PATH)
print('n_tokens 분포:'); print(df['n_tokens'].describe())

# 검증·현황
from collections import Counter
print('=== 매체그룹별 평균 토큰 수 ===')
print(df.groupby('media_group')['n_tokens'].mean().round(1))
print('\n=== 전체 상위 빈도 단어 30 (빼기로 한 단어들이 거의 안 보여야 정상) ===')
allc = Counter()
for s in df['tokens']:
    allc.update(s.split())
for w, c in allc.most_common(30):
    print(f'  {w} {c}')
