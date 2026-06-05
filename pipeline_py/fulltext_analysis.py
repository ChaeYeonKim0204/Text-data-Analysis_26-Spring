#!/usr/bin/env python
# coding: utf-8

# # 📰 전체 뉴스 텍스트 데이터 분석 — 9개 언론사 (2026.05.05~05.11)
# 
# **담당 파트:** 텍스트 데이터 분석기법 사용 — *전체 뉴스를 바탕으로*
# **데이터:** YTN·연합뉴스 재수집 반영 정본 (`과제2/` 폴더, 11,990건 → 비기사 필터 후 10,435건)
# **작성 원칙:** 교수님 강의(Topic3·Topic4) 방법론을 기반으로 하되, 시각화·검색·요약 부분을 **고도화**하여 적용
# 
# ---
# 
# ## 📑 분석 구성 & 강의 파일 매핑
# 
# | # | 분석 | 적용 기법 | 근거 강의 파일 |
# |---|------|----------|---------------|
# | 0 | 환경설정 & 데이터 로드 | — | `Topic2/02` (Kiwi 전처리) |
# | 1 | 데이터 개요 & 업로드 패턴 | EDA·히트맵 | — |
# | 2 | **단어 빈도 TOP20 & Word Cloud** | WordCloud, 2-gram | `Topic3/01~03. WordCloud` |
# | 3 | **감성 분석** | KNU 감성사전 | `Topic4/02. Sentiment Analysis(2)` |
# | 4 | **문서 검색 (Document Retrieval)** | TF-IDF + 코사인유사도 | `Topic3/04. Document_retrieval` |
# | 5 | **텍스트 분류 (Text Classification)** | TF-IDF + 로지스틱회귀 | `Topic4/03. Sentiment Analysis(3)` |
# | 6 | **텍스트 요약 (Text Summarization)** | TextRank (networkx) | `Topic4/04. Text Summarization` |
# | 7 | (심화) 미디어그룹 변별어 | 가중 로그-오즈 | — |
# | 8 | 종합 요약 | — | — |
# 
# > 과제 필수 요건은 **3개 이상의 분석 기법** 적용입니다. 본 노트북은 Word Cloud·Sentiment Analysis·Document Retrieval·Text Classification·Text Summarization **5개 기법**을 전체 뉴스에 적용했습니다.
# > (Topic Modeling(LDA)은 *특정 토픽 기준 언론사별 비교* 파트에서 다른 팀원이 담당 — `analysis_언론사_lda결과_재수집.csv` 참조)
# 
# ---
# 
# ### ⚠️ 감성 분석 방법론 주의 (해석 시 필독)
# KNU 감성사전은 **단어 단위** 사전이고, 제공된 분석토큰은 **명사 위주**라 사전 매칭률이 토큰의 약 3.7%입니다(기사 86%는 감성어 1개 이상 포함).
# → 이 점수는 **"기사가 긍정/부정 어휘를 얼마나 담고 있는가(소재 감성)"** 를 측정하며, 기자의 *논조·논평 방향*과 반드시 일치하지는 않습니다.
# 해석은 *"A 언론사가 긍정 어휘를 상대적으로 더 많이 사용"* 수준으로 신중하게 서술하세요.

# ## 0️⃣ 환경 설정 & 데이터 로드
# 
# > 💡 **팀원 안내**: 아래 `ROOT` 경로만 본인 환경에 맞게 수정하면 전체 셀이 순서대로 실행됩니다.
# > 필요 패키지: `pip install wordcloud scikit-learn gensim networkx seaborn kiwipiepy`

# In[1]:


# ── 라이브러리 ──
import json, re, warnings, time
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
# [PIPELINE strip] %matplotlib inline
# 한글 폰트 (Windows: 맑은 고딕). Mac은 'AppleGothic'으로 변경
import matplotlib.font_manager as _fm; _fm.fontManager.addfont(str(_cfg.FONT_PATH)); plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 110

from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import networkx as nx

print('라이브러리 로드 완료')


# In[2]:


# ── 경로 설정 (★본인 환경에 맞게 ROOT만 수정) ──
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent)); import config as _cfg
QA   = _cfg.DATA_DIR  # [PIPELINE patch] 디렉토리만 교체, 파일명 리터럴 verbatim
OUT  = _cfg.CHART_OUT; OUT.mkdir(parents=True, exist_ok=True)
KNU  = _cfg.KNU_PATH
FONT = str(_cfg.FONT_PATH)  # [PIPELINE patch] NanumGothic (워드클라우드, PNG 전용)

# 언론사 고유 색상 팔레트 (시각화 일관성)
PALETTE = {'YTN':'#0a3d62','연합뉴스':'#3c6382','KBS':'#1e3799','한국경제':'#b71540',
           '매일경제':'#e58e26','SBS':'#079992','조선일보':'#6F1E51','MBC':'#7158e2','한겨레':'#218c74'}
print('경로 설정 완료:', QA)


# In[3]:


# ── 데이터 로드 ──
# 1) 전처리 본문 파일 : 메타데이터 + 본문/제목 + 비기사 플래그
# 2) 분석토큰 파일    : Kiwi로 형태소 분석된 토큰 (Topic2/02 방식, 명사 위주)
pre = pd.read_csv(QA/'전처리_본문_언론사_260505_260511.csv', encoding='utf-8-sig')
tok = pd.read_csv(QA/'분석토큰_언론사_260505_260511.csv',   encoding='utf-8-sig')

# 토큰 문자열 → 리스트  (예: "이란 미국 후보" → ['이란','미국','후보'])
tok['tokens_list'] = tok['tokens'].fillna('').str.split()

# 두 파일을 article_id로 병합 (본문/제목/플래그를 토큰 테이블에 결합)
# 주의: title_cleaned 는 토큰 파일(tok)에 이미 있으므로 pre 쪽에서는 제외해 컬럼 충돌 방지
df = tok.merge(
    pre[['article_id','pubdate','title','body_cleaned','text','body_length',
         'is_weather','is_closing','is_within_press_dup','is_cross_press_dup']],
    on='article_id', how='left')

print(f'전체: {len(df):,}건 / {df["press"].nunique()}개 언론사')
df[['press','article_category','title_cleaned','n_tokens']].head(3)


# In[4]:


# ── 비(非)기사 필터링 ──
# 날씨·증시마감 등 정형 기사와 '동일 매체 내 중복'은 분석에서 제외한다.
# (단, '교차 매체 중복'=같은 사건을 여러 언론사가 보도한 것은 언론사 비교에 필요하므로 유지)
before = len(df)
df = df[~df['is_weather'] & ~df['is_closing'] & ~df['is_within_press_dup']].copy().reset_index(drop=True)

# 시간 파생 변수
df['pubdate'] = pd.to_datetime(df['pubdate'], errors='coerce')
df['hour']    = df['pubdate'].dt.hour      # 업로드 시각(0~23)
df['date']    = df['pubdate'].dt.date      # 날짜
df['weekday'] = df['pubdate'].dt.weekday   # 요일(0=월 ~ 6=일)

PRESS_ORDER = df['press'].value_counts().index.tolist()  # 기사 수 내림차순 언론사 순서
print(f'필터: {before:,} → {len(df):,}건 ({before-len(df):,}건 제외)')


# ## 1️⃣ 데이터 개요 & 업로드 패턴
# 
# 전체 뉴스가 **어떤 언론사·카테고리·시간대**에 분포하는지 먼저 조망합니다.
# 특히 언론사 × 시간대 히트맵은 *각 매체의 발행 리듬(편집 전략)* 을 드러냅니다.

# In[5]:


# 언론사별 / 카테고리별 기사 수
fig, axes = plt.subplots(1, 2, figsize=(16,5))
vc = df['press'].value_counts()
axes[0].bar([str(x) for x in vc.index], vc.values, color=[PALETTE.get(p,'#888') for p in vc.index])
for i,v in enumerate(vc.values): axes[0].text(i, v+5, str(v), ha='center', fontsize=9)
axes[0].set_title('언론사별 기사 수', fontsize=13, fontweight='bold'); axes[0].tick_params(axis='x', rotation=20)

cc = df['article_category'].value_counts()
axes[1].bar([str(x) for x in cc.index], cc.values, color='#3c6382')
for i,v in enumerate(cc.values): axes[1].text(i, v+5, str(v), ha='center', fontsize=9)
axes[1].set_title('카테고리별 기사 수', fontsize=13, fontweight='bold'); axes[1].tick_params(axis='x', rotation=20)
plt.tight_layout(); plt.savefig(OUT/'F01_F02_분포.png', dpi=140, bbox_inches='tight'); plt.show()


# In[6]:


# 일자별 & 시간대별 업로드량
wd = ['월','화','수','목','금','토','일']
day_vol  = df.groupby('date').size()
hour_vol = df.groupby('hour').size().reindex(range(24), fill_value=0)
xl = [f"{d.strftime('%m/%d')}({wd[d.weekday()]})" for d in day_vol.index]

fig, axes = plt.subplots(1, 2, figsize=(16,4.5))
axes[0].plot(range(len(day_vol)), day_vol.values, marker='o', lw=2, color='#b71540')
for i,v in enumerate(day_vol.values): axes[0].text(i, v+8, str(v), ha='center', fontsize=9)
axes[0].set_xticks(range(len(day_vol))); axes[0].set_xticklabels(xl)
axes[0].set_title('일자별 기사 수', fontsize=13, fontweight='bold')

axes[1].bar(hour_vol.index, hour_vol.values, color='#3c6382')
axes[1].set_xticks(range(0,24,2)); axes[1].set_xlabel('시각(시)')
axes[1].set_title('시간대별 업로드량', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(OUT/'F03_F04_업로드패턴.png', dpi=140, bbox_inches='tight'); plt.show()


# In[7]:


# 언론사 × 시간대 히트맵 — 매체별 발행 리듬
piv = df.pivot_table(index='press', columns='hour', values='article_id',
                     aggfunc='count', fill_value=0).reindex(PRESS_ORDER)
plt.figure(figsize=(13,5))
sns.heatmap(piv, cmap='YlOrRd', linewidths=.3, linecolor='white', cbar_kws={'label':'기사 수'})
plt.title('언론사 × 시간대 업로드 히트맵', fontsize=13, fontweight='bold')
plt.xlabel('시각(시)'); plt.ylabel('')
plt.tight_layout(); plt.savefig(OUT/'F05_언론사_시간대_히트맵.png', dpi=140, bbox_inches='tight'); plt.show()


# **📌 해석**
# - **경제지(한국경제·매일경제)** 는 **16~18시(증시 마감 직후)** 에 발행이 집중 — 마감 시황·해설 기사 패턴.
# - **KBS** 는 **21~22시(메인 뉴스 직후)**, **연합뉴스·YTN** 은 오후 내내 고르게 — 통신·방송사의 실시간 송고 특성.
# - **조선일보의 0시 급증**은 시각이 명시되지 않은 기사가 자정으로 처리됐을 가능성이 있어, 발표 전 원본 `pubdate` 확인을 권장합니다(데이터 품질 체크 포인트).

# ## 2️⃣ 단어 빈도 TOP 20 & 워드클라우드  ·  `Topic3/01~03. WordCloud`
# 
# 강의의 `WordCloud(font_path, background_color, width, height).generate_from_frequencies(freq)` 방식을 그대로 쓰되,
# **① 빈도 막대그래프**, **② 미디어그룹별 6분할 워드클라우드**, **③ 2-gram(연속 단어쌍) 빈도** 로 **고도화**했습니다.

# In[8]:


# 전체 토큰 빈도 집계
all_tokens = Counter()
for t in df['tokens_list']: all_tokens.update(t)
top20 = all_tokens.most_common(20)

plt.figure(figsize=(10,6))
ws = [w for w,_ in top20][::-1]; csv = [c for _,c in top20][::-1]
plt.barh(ws, csv, color='#1e3799')
for i,c in enumerate(csv): plt.text(c, i, f' {c:,}', va='center', fontsize=9)
plt.title('전체 뉴스 단어 빈도 TOP 20', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(OUT/'F06_단어빈도TOP20.png', dpi=140, bbox_inches='tight'); plt.show()


# In[9]:


# 워드클라우드 (전체 + 미디어그룹별)
def make_wc(counter, cmap):
    return WordCloud(font_path=FONT, background_color='white', width=800, height=500,
                     max_words=150, colormap=cmap, prefer_horizontal=0.9
                    ).generate_from_frequencies(counter)

groups = df['media_group'].value_counts().index.tolist()
cmaps  = ['Blues','Oranges','Greens','Purples','Reds']
fig, axes = plt.subplots(2, 3, figsize=(20,11)); axes = axes.flatten()
axes[0].imshow(make_wc(all_tokens,'viridis').to_array()); axes[0].set_title('전체 뉴스', fontsize=14, fontweight='bold'); axes[0].axis('off')
for k,g in enumerate(groups):
    c = Counter()
    for t in df[df['media_group']==g]['tokens_list']: c.update(t)
    axes[k+1].imshow(make_wc(c, cmaps[k%len(cmaps)]).to_array())
    axes[k+1].set_title(str(g), fontsize=14, fontweight='bold'); axes[k+1].axis('off')
for j in range(len(groups)+1, 6): axes[j].axis('off')
plt.tight_layout(); plt.savefig(OUT/'F07_워드클라우드.png', dpi=130, bbox_inches='tight'); plt.show()


# In[10]:


# 2-gram(연속 단어쌍) 빈도 — 단일 단어보다 문맥(개념 단위)을 잘 포착
def bigrams(t): return list(zip(t, t[1:]))
bg = Counter()
for t in df['tokens_list']: bg.update(bigrams(t))
top_bg = bg.most_common(20)

plt.figure(figsize=(10,6))
bl = [f'{a}·{b}' for (a,b),_ in top_bg][::-1]; bc = [c for _,c in top_bg][::-1]
plt.barh(bl, bc, color='#6F1E51')
for i,c in enumerate(bc): plt.text(c, i, f' {c:,}', va='center', fontsize=9)
plt.title('전체 뉴스 2-gram(연속 단어쌍) 빈도 TOP 20', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(OUT/'F08_바이그램TOP20.png', dpi=140, bbox_inches='tight'); plt.show()


# **📌 해석** — 단일 단어 TOP은 `이란·미국·후보·대통령`, 2-gram TOP은 `호르무즈·해협`, `트럼프·대통령`, `지방·선거`, `미국·이란`.
# 이 주(5/5~11)의 **양대 의제는 "이란-미국 호르무즈 해협 긴장(국제)" 과 "지방선거 정국(국내 정치)"** 임을 빈도만으로 확인할 수 있습니다.
# 미디어그룹별 워드클라우드에서 **경제그룹은 'AI·시장·기업', 정치색 매체는 '대통령·국민'** 이 두드러져 매체 성격 차이가 드러납니다.

# ## 3️⃣ 감성 분석  ·  `Topic4/02. Sentiment Analysis(2)`
# 
# 강의의 **KNU 한국어 감성어 사전 + `calculate_score()`** 함수를 그대로 사용합니다.
# (토큰이 사전에 있으면 극성을 합산, 같은 단어는 |극성| 큰 값 1개만 사용)
# **고도화**: ① 기사 길이 편향 보정을 위해 *100토큰당 점수*로 정규화 → 언론사·카테고리·제목/본문을 공정 비교, ② 실제로 많이 쓰인 긍/부정 어휘 TOP.

# In[11]:


# KNU 감성사전 로드
with open(KNU, encoding='utf-8-sig') as f:
    sentiword = json.load(f)
df_senti = pd.DataFrame(sentiword)
df_senti['polarity'] = df_senti['polarity'].astype(int)

# [강의 원본 함수] 토큰 리스트의 감성 점수 = 사전에 매칭된 단어들의 극성 합
def calculate_score(token_list, ds=df_senti, col='word'):
    s = ds[ds[col].isin(token_list)]
    s = s.sort_values('polarity', key=lambda x: np.abs(x)).drop_duplicates(col, keep='first')
    return s['polarity'].sum()

df['senti']    = df['tokens_list'].map(calculate_score)            # 원점수(절대 합)
df['ntok']     = df['tokens_list'].map(len)
df['senti100'] = df['senti'] / df['ntok'].replace(0, np.nan) * 100 # 100토큰당 정규화(길이 보정)
df['sclass']   = df['senti'].map(lambda s: '긍정' if s>0 else ('부정' if s<0 else '중립'))

print(df['sclass'].value_counts(normalize=True).mul(100).round(1).astype(str)+'%')


# In[12]:


# (1) 언론사별 평균 감성  (2) 분포 박스플롯
fig, axes = plt.subplots(1, 2, figsize=(17,5))
ps = df.groupby('press')['senti100'].mean().reindex(PRESS_ORDER)
axes[0].bar([str(x) for x in ps.index], ps.values,
            color=['#4CAF50' if v>=0 else '#d6604d' for v in ps.values])
for i,v in enumerate(ps.values): axes[0].text(i, v, f'{v:+.2f}', ha='center',
                                              va='bottom' if v>=0 else 'top', fontsize=9)
axes[0].axhline(0, color='gray', ls='--', lw=.8); axes[0].tick_params(axis='x', rotation=20)
axes[0].set_title('언론사별 평균 감성 (100토큰당)', fontsize=13, fontweight='bold')

data = [df[df['press']==p]['senti100'].dropna().values for p in PRESS_ORDER]
bp = axes[1].boxplot(data, labels=PRESS_ORDER, showfliers=False, patch_artist=True)
for patch,p in zip(bp['boxes'], PRESS_ORDER):
    patch.set_facecolor(PALETTE.get(p,'#888')); patch.set_alpha(.7)
axes[1].axhline(0, color='gray', ls='--', lw=.8); axes[1].tick_params(axis='x', rotation=20)
axes[1].set_title('언론사별 감성 점수 분포', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(OUT/'F09_F10_언론사감성.png', dpi=140, bbox_inches='tight'); plt.show()


# In[13]:


# (3) 카테고리별 감성  (4) 제목 vs 본문 감성
fig, axes = plt.subplots(1, 2, figsize=(17,5))
cs = (df.groupby('article_category')
        .agg(n=('senti100','size'), s=('senti100','mean'))
        .query('n>=30').sort_values('s'))
axes[0].barh([str(x) for x in cs.index], cs['s'],
             color=['#d6604d' if v<0 else '#4CAF50' for v in cs['s']])
for i,(idx,r) in enumerate(cs.iterrows()):
    axes[0].text(r['s'], i, f" {r['s']:+.2f}(n={int(r['n'])})", va='center', fontsize=9)
axes[0].axvline(0, color='gray', ls='--', lw=.8)
axes[0].set_title('카테고리별 평균 감성 (100토큰당)', fontsize=13, fontweight='bold')

df['title_tokens'] = df['title_cleaned'].fillna('').str.split()
df['title_senti']  = df['title_tokens'].map(calculate_score)
cmp = df.groupby('press').agg(제목=('title_senti','mean'), 본문=('senti100','mean')).reindex(PRESS_ORDER)
x = np.arange(len(cmp)); w = 0.38
axes[1].bar(x-w/2, cmp['제목'], w, label='제목(절대합)', color='#82589F')
axes[1].bar(x+w/2, cmp['본문'], w, label='본문(100토큰당)', color='#B33771')
axes[1].set_xticks(x); axes[1].set_xticklabels(cmp.index, rotation=20)
axes[1].axhline(0, color='gray', ls='--', lw=.8); axes[1].legend()
axes[1].set_title('제목 vs 본문 감성', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(OUT/'F11_F12_카테고리_제목본문.png', dpi=140, bbox_inches='tight'); plt.show()


# In[14]:


# (5) 실제로 많이 쓰인 긍정/부정 감성어 TOP15
pol = df_senti.drop_duplicates('word').set_index('word')['polarity'].to_dict()
pos_w, neg_w = Counter(), Counter()
for t in df['tokens_list']:
    for w in t:
        p = pol.get(w)
        if   p and p>0: pos_w[w] += 1
        elif p and p<0: neg_w[w] += 1

fig, axes = plt.subplots(1, 2, figsize=(15,6))
for ax,(data,title,col) in zip(axes, [(pos_w.most_common(15),'긍정 감성어 TOP15','#4CAF50'),
                                       (neg_w.most_common(15),'부정 감성어 TOP15','#d6604d')]):
    ll = [w for w,_ in data][::-1]; vv = [c for _,c in data][::-1]
    ax.barh(ll, vv, color=col)
    for i,v in enumerate(vv): ax.text(v, i, f' {v:,}', va='center', fontsize=8)
    ax.set_title(title, fontsize=12, fontweight='bold')
plt.tight_layout(); plt.savefig(OUT/'F13_긍부정어TOP.png', dpi=140, bbox_inches='tight'); plt.show()


# **📌 해석** — 전체 뉴스 감성은 **긍정 40.6% / 부정 38.1% / 중립 21.3%** 로, 경제지 단독 분석(긍정 우세)과 달리 **균형적**입니다.
# 카테고리별로는 **경제·스포츠가 양(+)**, **사회·세계가 음(−)** — 이란-미국 긴장·사건사고가 부정 어휘를 끌어올린 결과로 보입니다.
# 제목 감성은 대체로 0 부근(중립)인 반면 본문은 변동이 커, *헤드라인은 중립적으로 뽑고 본문에서 색채가 드러나는* 경향이 관찰됩니다.

# ## 4️⃣ 문서 검색 (Document Retrieval)  ·  `Topic3/04. Document_retrieval`
# 
# 강의의 **TF-IDF 벡터화 → 코사인 유사도** 파이프라인을 뉴스에 적용해 **뉴스 검색 엔진**을 만듭니다.
# - `find_similar(idx)` : 특정 기사와 **내용이 가장 비슷한 기사** 찾기 (→ *어느 언론사가 같은 사건을 어떻게 보도했나* 비교에 활용)
# - `search_keyword(query)` : **키워드 질의**로 관련 기사 검색

# In[15]:


# 토큰을 공백으로 join → TF-IDF 입력 텍스트
df['text_join'] = df['tokens_list'].map(lambda t: ' '.join(t))
tfidf = TfidfVectorizer(max_features=20000, min_df=3)
mat   = tfidf.fit_transform(df['text_join'])
print('TF-IDF 행렬:', mat.shape)   # (문서 수, 어휘 수)

def find_similar(idx, topn=5):
    sims  = cosine_similarity(mat[idx], mat).flatten()
    order = [i for i in sims.argsort()[::-1] if i != idx][:topn]
    return [(i, round(float(sims[i]),3)) for i in order]

def search_keyword(query, topn=5):
    qv    = tfidf.transform([query])
    sims  = cosine_similarity(qv, mat).flatten()
    order = sims.argsort()[::-1][:topn]
    return [(i, round(float(sims[i]),3)) for i in order]


# In[16]:


# 데모 (1): 0번 기사와 유사한 기사 찾기
idx = 0
print(f'[기준 기사] ({df.iloc[idx]["press"]}) {df.iloc[idx]["title_cleaned"]}\n')
print('▼ 가장 유사한 기사 5건')
for i,s in find_similar(idx, 5):
    print(f'  유사도 {s} | [{df.iloc[i]["press"]:5s}] {str(df.iloc[i]["title_cleaned"])[:50]}')


# In[17]:


# 데모 (2): 키워드 질의 검색  (원하는 키워드로 자유롭게 바꿔 보세요)
query = '이란 호르무즈 원유 봉쇄'
print(f'[검색어] {query}\n▼ 관련 기사 5건')
for i,s in search_keyword(query, 5):
    print(f'  유사도 {s} | [{df.iloc[i]["press"]:5s}] {str(df.iloc[i]["title_cleaned"])[:50]}')


# **📌 활용** — `find_similar`는 **같은 사건의 교차 보도**를 자동으로 묶어줍니다(예: 장동혁 단일화 발언 → MBC·조선·매경 동일 사건). 토픽별 언론사 비교 파트에서 *대표 기사 선정·중복 사건 매칭*에 그대로 쓸 수 있습니다.

# ## 5️⃣ 텍스트 분류 (Text Classification)  ·  `Topic4/03. Sentiment Analysis(3)`
# 
# 강의에서 감성 분류에 쓴 **TF-IDF + 로지스틱 회귀** 구조를 **뉴스 카테고리(정치·경제·사회·세계 …) 자동 분류기**로 확장합니다.
# 본문만으로 카테고리를 얼마나 맞히는지 = *카테고리 간 어휘가 얼마나 변별되는지* 를 정량화합니다.

# In[18]:


# 표본이 충분한 카테고리(200건 이상)만 사용
cat_keep = df['article_category'].value_counts()
cat_keep = cat_keep[cat_keep >= 200].index.tolist()
dcls = df[df['article_category'].isin(cat_keep)].copy()

Xc = tfidf.transform(dcls['text_join'])      # 위에서 학습한 TF-IDF 재사용
yc = dcls['article_category'].values
x_tr, x_te, y_tr, y_te = train_test_split(Xc, yc, test_size=0.2, stratify=yc, random_state=42)

clf = LogisticRegression(max_iter=1000, C=5.0)
clf.fit(x_tr, y_tr)
y_pred = clf.predict(x_te)
acc = accuracy_score(y_te, y_pred)
print(f'카테고리 자동 분류 정확도: {acc*100:.1f}%  (카테고리 {len(cat_keep)}종)\n')
print(classification_report(y_te, y_pred, digits=3))


# In[19]:


# 혼동행렬 히트맵
labels = sorted(set(yc))
cm = confusion_matrix(y_te, y_pred, labels=labels)
plt.figure(figsize=(8,6.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f'카테고리 자동 분류 혼동행렬 (정확도 {acc*100:.1f}%)', fontsize=12, fontweight='bold')
plt.xlabel('예측'); plt.ylabel('실제')
plt.tight_layout(); plt.savefig(OUT/'F14_분류_혼동행렬.png', dpi=140, bbox_inches='tight'); plt.show()


# **📌 해석** — 본문 TF-IDF만으로 **약 83% 정확도**. 스포츠·세계·정치는 어휘가 뚜렷해 잘 맞히고, **IT는 경제와 혼동**(IT↔경제 용어 공유), **사회**는 다양한 주제가 섞여 오분류를 흡수하는 *포괄 카테고리* 역할을 합니다. → 카테고리 경계가 모호한 기사 검수에 활용 가능.

# ## 6️⃣ 텍스트 요약 (Text Summarization)  ·  `Topic4/04. Text Summarization`
# 
# 강의는 `spaCy + pytextrank`를 사용하지만, 한국어 모델(`ko_core_news_sm`) 설치가 필요합니다.
# 여기서는 **추가 설치 없이** 돌아가도록 **TextRank 알고리즘을 직접 구현**(고도화)했습니다.
# > 원리: 문장들을 TF-IDF로 벡터화 → 문장 간 코사인 유사도 그래프 → **PageRank**로 중심 문장 선정 (강의 04의 TextRank와 동일한 아이디어).

# In[20]:


from kiwipiepy import Kiwi
_kiwi = Kiwi()

def textrank_summary(text, n=3):
    '''기사 본문에서 핵심 문장 n개를 추출 (TextRank).'''
    sents = [s.text.strip() for s in _kiwi.split_into_sents(text) if len(s.text.strip()) > 10]
    if len(sents) <= n:
        return sents
    vec = TfidfVectorizer().fit_transform(sents)      # 문장 단위 TF-IDF
    sim = cosine_similarity(vec); np.fill_diagonal(sim, 0)
    scores = nx.pagerank(nx.from_numpy_array(sim), max_iter=100)  # PageRank 중심성
    ranked = sorted(((scores[i], i, s) for i,s in enumerate(sents)), reverse=True)
    top = sorted(ranked[:n], key=lambda x: x[1])      # 상위 n개를 원문 순서로 정렬
    return [s for _,_,s in top]


# In[21]:


# 데모: 본문이 충분히 긴 기사를 골라 3문장 요약
sample = df[df['body_length'] > 800].iloc[10]
print(f'[기사] ({sample["press"]}) {sample["title_cleaned"]}\n')
print('▼ TextRank 3문장 요약')
for s in textrank_summary(sample['text'], 3):
    print(' •', s)


# **📌 활용** — 기사별 핵심 문장 자동 추출은 **대량 기사 빠른 스캐닝**과 **토픽 대표 문장 뽑기**에 유용합니다.
# (뉴스 본문은 소제목이 마침표 없이 붙는 경우가 있어 첫 문장이 길게 잡힐 수 있습니다 → `body_cleaned` 사용·길이 필터로 완화)

# ## 7️⃣ (심화) 미디어그룹 변별어 분석 — 가중 로그-오즈
# 
# 강의 범위를 넘어선 **심화 기법**입니다. 단순 빈도는 'AI·미국'처럼 어디서나 흔한 단어가 상위를 차지하므로,
# **가중 로그-오즈비(Monroe et al. 2008)** 로 *"그 그룹에서 유독 더 많이 쓰는 단어"* 를 통계적으로 추출합니다.

# In[22]:


def logodds_group(target, a0=1000.0, min_total=20, topn=12):
    '''target 미디어그룹 vs 나머지 전체의 변별어를 z-score로 산출.'''
    cT, cR = Counter(), Counter()
    for grp,t in zip(df['media_group'], df['tokens_list']):
        (cT if grp==target else cR).update(t)
    vocab = {w for w in set(cT)|set(cR) if cT[w]+cR[w] >= min_total}
    tot = sum(cT[w]+cR[w] for w in vocab); nT = sum(cT[w] for w in vocab); nR = sum(cR[w] for w in vocab)
    z = {}
    for w in vocab:
        aw = a0*(cT[w]+cR[w])/tot
        lo = np.log((cT[w]+aw)/(nT+a0-cT[w]-aw)) - np.log((cR[w]+aw)/(nR+a0-cR[w]-aw))
        z[w] = lo/np.sqrt(1/(cT[w]+aw) + 1/(cR[w]+aw))
    return sorted(z.items(), key=lambda x: -x[1])[:topn]

groups = df['media_group'].value_counts().index.tolist()
fig, axes = plt.subplots(1, len(groups), figsize=(5*len(groups),6))
if len(groups)==1: axes=[axes]
for ax,g in zip(axes, groups):
    tw = logodds_group(g)
    ll = [w for w,_ in tw][::-1]; zz = [z for _,z in tw][::-1]
    ax.barh(ll, zz, color='#3c6382'); ax.set_title(f'{g}\n변별어', fontsize=11, fontweight='bold')
plt.suptitle('미디어그룹별 변별어 (가중 로그-오즈, 그룹 vs 나머지)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(OUT/'F15_미디어그룹_변별어.png', dpi=130, bbox_inches='tight'); plt.show()


# ## 8️⃣ 종합 요약
# 
# | 분석 | 핵심 발견 | 활용 포인트 |
# |------|-----------|------------|
# | 업로드 패턴 | 경제지 16-18시·KBS 21-22시 집중 / 조선 0시 급증(확인 필요) | 매체별 발행 리듬 = 편집 전략 |
# | 빈도·워드클라우드 | 양대 의제 = **이란 호르무즈(국제) · 지방선거(정치)** | 주간 의제 한눈 파악 |
# | 감성 | 전체 긍40·부38·중21로 균형 / 경제·스포츠(+), 사회·세계(−) | 카테고리별 감성 톤 비교 |
# | 문서 검색 | TF-IDF+코사인으로 교차 보도 자동 매칭 | 같은 사건 언론사 비교 |
# | 텍스트 분류 | 본문만으로 카테고리 **83%** 분류 / 사회=포괄 카테고리 | 카테고리 변별력·검수 |
# | 텍스트 요약 | TextRank로 핵심 3문장 추출 | 대량 기사 스캐닝 |
# | 변별어 | 경제=시장·기업, 정치색=대통령·국민 | 매체 성격 정량화 |
# 
# ---
# 
# ### 📂 산출물
# - 본 노트북: `과제2/전체뉴스_텍스트분석.ipynb`
# - 차트 이미지: `과제2/산출물_차트/F01~F15*.png`
# 
# ### 🔗 강의 파일 매핑 요약
# - WordCloud → `Topic3/01~03` · Document Retrieval → `Topic3/04`
# - Sentiment → `Topic4/02` · Text Classification → `Topic4/03` · Text Summarization → `Topic4/04`
# - 전처리(Kiwi) → `Topic2/02` · (참고) Topic Modeling → `Topic4/05~06` *(다른 팀원 담당)*
