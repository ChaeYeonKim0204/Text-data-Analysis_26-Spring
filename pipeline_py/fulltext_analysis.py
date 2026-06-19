#!/usr/bin/env python
# coding: utf-8
# ④ 전체텍스트 분석
# 전처리 본문과 분석토큰 파일 읽기, 빈도·감성·검색·분류·요약 결과 생성

# 전체 뉴스 텍스트 데이터 분석 — 9개 언론사 (2026.05.05~05.11)
# 전체 기사 데이터에 여러 분석 방법을 한 번에 적용하는 구성
# YTN·연합뉴스 재수집 데이터 반영, 비기사 제외 후 10,435건 분석
# 강의에서 배운 빈도, 감성, 검색, 분류, 요약 방법을 뉴스 데이터에 맞춰 적용
#
# 분석 내용
# 0. 데이터 불러오기와 기본 확인
# 1. 언론사·카테고리·시간대 분포
# 2. 단어 빈도와 워드클라우드
# 3. KNU 감성사전으로 감성 점수 보기
# 4. 비슷한 기사나 키워드 관련 기사 찾기
# 5. 본문 단어만 보고 기사 카테고리 맞히기
# 6. 긴 기사에서 중심 문장 뽑기
# 7. 미디어그룹별로 특히 많이 쓰인 단어 보기
# LDA 토픽 분석 별도 단계, 이 파일에서는 미사용
#
# 감성 분석은 해석 범위를 넓게 잡지 않도록 주의
# KNU 사전은 단어별 점수표이고, 분석토큰은 명사 위주라 사전에 안 잡히는 단어도 많음
# 이 값은 기사 안의 긍정/부정 단어 포함 정도를 보는 용도로 사용

# 환경 설정 & 데이터 로드
# 아래 경로 설정만 맞으면 전체 코드 순차 실행
# 필요 패키지: pip install wordcloud scikit-learn gensim networkx seaborn kiwipiepy



# ── 라이브러리 ──
import json, re, warnings, time
warnings.filterwarnings('ignore')   # 분석 결과 출력의 불필요한 안내문 섞임 방지
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
# 그래프에서 마이너스 기호가 깨지지 않게 설정
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




# ── 경로 설정 ──
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent)); import config as _cfg
QA   = _cfg.DATA_DIR
OUT  = _cfg.CHART_OUT; OUT.mkdir(parents=True, exist_ok=True)
KNU  = _cfg.KNU_PATH
FONT = str(_cfg.FONT_PATH)  # 그래프 한글 표시용 폰트
import matplotlib.font_manager as _fm; _fm.fontManager.addfont(FONT); plt.rc('font', family='NanumGothic')

# 언론사별로 그래프 색을 같게 쓰기 위한 설정
PALETTE = {'YTN':'#0a3d62','연합뉴스':'#3c6382','KBS':'#1e3799','한국경제':'#b71540',
           '매일경제':'#e58e26','SBS':'#079992','조선일보':'#6F1E51','MBC':'#7158e2','한겨레':'#218c74'}
print('경로 설정 완료:', QA)




# ── 데이터 로드 ──
# 1) 전처리 본문 파일 : 출처 정보 + 본문/제목 + 비기사 표시 컬럼
# 2) 분석토큰 파일    : Kiwi로 형태소 분석된 토큰 (Topic2/02 방식, 명사 위주)
pre = pd.read_csv(QA/'전처리_본문_언론사_260505_260511.csv', encoding='utf-8-sig')
tok = pd.read_csv(QA/'분석토큰_언론사_260505_260511.csv',   encoding='utf-8-sig')

# 토큰 문자열 → 리스트  (예: "이란 미국 후보" → ['이란','미국','후보'])
tok['tokens_list'] = tok['tokens'].fillna('').str.split()   # 저장된 토큰 문자열을 다시 단어 리스트로 변환

# 두 파일을 article_id로 합침 (본문/제목/표시 컬럼을 토큰 표에 붙임)
# title_cleaned는 토큰 파일에 이미 있으므로 한 번만 사용
df = tok.merge(
    pre[['article_id','pubdate','title','body_cleaned','text','body_length',
         'is_weather','is_closing','is_within_press_dup','is_cross_press_dup','is_foreign']],
    on='article_id', how='left')

print(f'전체: {len(df):,}건 / {df["press"].nunique()}개 언론사')




# ── 비(非)기사 필터링 ──
# 날씨·증시마감처럼 늘 비슷한 기사와 '동일 매체 내 중복'은 분석에서 제외함
# (단, '교차 매체 중복'=같은 사건을 여러 언론사가 보도한 것은 언론사 비교에 필요하므로 유지)
before = len(df)
df = df[~df['is_weather'] & ~df['is_closing'] & ~df['is_within_press_dup'] & ~df['is_foreign']].copy().reset_index(drop=True)

# 시간 관련 컬럼 만들기
df['pubdate'] = pd.to_datetime(df['pubdate'], errors='coerce')
df['hour']    = df['pubdate'].dt.hour      # 업로드 시각(0~23)
df['date']    = df['pubdate'].dt.date      # 날짜
df['weekday'] = df['pubdate'].dt.weekday   # 요일(0=월 ~ 6=일)

PRESS_ORDER = df['press'].value_counts().index.tolist()  # 기사 수 내림차순 순서를 고정해 모든 차트에서 언론사 순서를 일관 유지
print(f'필터: {before:,} → {len(df):,}건 ({before-len(df):,}건 제외)')


# 데이터 개요 & 업로드 패턴
# 
# 전체 뉴스가 어떤 언론사·카테고리·시간대에 몰려 있는지 먼저 확인
# 언론사 × 시간대 히트맵은 매체별 기사 업로드 시간 차이를 보기 위해 사용



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




# 일자별 & 시간대별 업로드량
wd = ['월','화','수','목','금','토','일']
day_vol  = df.groupby('date').size()
hour_vol = df.groupby('hour').size().reindex(range(24), fill_value=0)   # 기사 없는 시간대는 0으로 채움
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




# 언론사 × 시간대 히트맵 — 매체별 업로드 시간 패턴 확인
piv = df.pivot_table(index='press', columns='hour', values='article_id',
                     aggfunc='count', fill_value=0).reindex(PRESS_ORDER)
plt.figure(figsize=(13,5))
sns.heatmap(piv, cmap='YlOrRd', linewidths=.3, linecolor='white', cbar_kws={'label':'기사 수'})
plt.title('언론사 × 시간대 업로드 히트맵', fontsize=13, fontweight='bold')
plt.xlabel('시각(시)'); plt.ylabel('')
plt.tight_layout(); plt.savefig(OUT/'F05_언론사_시간대_히트맵.png', dpi=140, bbox_inches='tight'); plt.show()


# 경제지(한국경제·매일경제)는 16~18시(증시 마감 직후) 발행 집중, 마감 시황·해설 기사 패턴
# KBS는 21~22시(메인 뉴스 직후), 연합뉴스·YTN은 오후 내내 고르게 — 통신사와 방송사는 오후 기사량이 계속 유지됨
# 조선일보 0시 급증은 시각 정보가 없어서 자정으로 처리됐을 가능성이 있어 원본 pubdate 확인 필요

# 단어 빈도 TOP 20 & 워드클라우드  ·  Topic3/01~03. WordCloud
# 
# 강의에서 사용한 WordCloud 방식에 빈도 막대그래프와 연속 단어쌍 빈도를 함께 확인



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




# 2-gram(연속 단어쌍) 빈도 — 단어 1개만 볼 때보다 주제를 더 잘 볼 수 있음
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


# 빈도 상위 단어에서는 이란·미국·호르무즈 쪽 국제 이슈와 지방선거 관련 정치 이슈가 크게 나타남
# 워드클라우드에서는 경제그룹 쪽에 AI·시장·기업 단어가 많고, 정치 관련 매체는 대통령·국민 쪽 단어가 눈에 띔

# 감성 분석  ·  Topic4/02. Sentiment Analysis(2)
# 
# 강의에서 쓴 KNU 한국어 감성어 사전과 calculate_score() 방식을 사용
# (토큰이 사전에 있으면 긍정/부정 점수를 더하고, 같은 단어는 점수가 큰 값 1개만 사용)
# 기사 길이가 다르므로 100토큰당 점수로 바꿔 언론사·카테고리·제목/본문을 비교
# 실제로 많이 쓰인 긍정/부정 단어도 따로 확인



# KNU 감성사전 로드
with open(KNU, encoding='utf-8-sig') as f:
    sentiword = json.load(f)
df_senti = pd.DataFrame(sentiword)
df_senti['polarity'] = df_senti['polarity'].astype(int)

# 토큰 중 감성사전에 있는 단어들의 점수를 더해 기사 감성 점수로 씀
def calculate_score(token_list, ds=df_senti, col='word'):
    s = ds[ds[col].isin(token_list)]
    s = s.sort_values('polarity', key=lambda x: np.abs(x)).drop_duplicates(col, keep='first')
    return s['polarity'].sum()

df['senti']    = df['tokens_list'].map(calculate_score)            # 원점수(절대 합)
df['ntok']     = df['tokens_list'].map(len)
df['senti100'] = df['senti'] / df['ntok'].replace(0, np.nan) * 100 # 기사 길이 차이를 줄이기 위해 100토큰당 점수로 계산
df['sclass']   = df['senti'].map(lambda s: '긍정' if s>0 else ('부정' if s<0 else '중립'))

print(df['sclass'].value_counts(normalize=True).mul(100).round(1).astype(str)+'%')




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


# 전체 뉴스 감성은 긍정 40.6% / 부정 38.1% / 중립 21.3% 로, 경제지 단독 분석(긍정이 더 많았던 결과)과 달리 균형적
# 카테고리별로는 경제·스포츠가 양(+), 사회·세계가 음(−) — 이란-미국 긴장·사건사고가 부정 어휘를 끌어올린 결과로 해석
# 제목 감성은 대체로 0 부근(중립)이고, 본문은 긍정/부정 점수 차이가 더 크게 나타남

# 문서 검색 (Document Retrieval)  ·  Topic3/04. Document_retrieval
# 
# 본문 단어를 바탕으로 비슷한 기사를 찾음
# find_similar(idx) — 특정 기사와 내용이 비슷한 기사 찾기
# search_keyword(query) — 검색어와 관련 있는 기사 찾기



# 토큰을 공백으로 join → TF-IDF 입력 텍스트
df['text_join'] = df['tokens_list'].map(lambda t: ' '.join(t))
tfidf = TfidfVectorizer(max_features=20000, min_df=3)   # 너무 드문 단어는 검색·분류에 도움이 적어 제외
mat   = tfidf.fit_transform(df['text_join'])
print('TF-IDF 결과 크기:', mat.shape)   # (문서 수, 어휘 수)

def find_similar(idx, topn=5):
    sims  = cosine_similarity(mat[idx], mat).flatten()
    order = [i for i in sims.argsort()[::-1] if i != idx][:topn]
    return [(i, round(float(sims[i]),3)) for i in order]

def search_keyword(query, topn=5):
    qv    = tfidf.transform([query])
    sims  = cosine_similarity(qv, mat).flatten()
    order = sims.argsort()[::-1][:topn]
    return [(i, round(float(sims[i]),3)) for i in order]




# 예시 (1): 0번 기사와 유사한 기사 찾기
idx = 0
print(f'[기준 기사] ({df.iloc[idx]["press"]}) {df.iloc[idx]["title_cleaned"]}\n')
print('▼ 가장 유사한 기사 5건')
for i,s in find_similar(idx, 5):
    print(f'  유사도 {s} | [{df.iloc[i]["press"]:5s}] {str(df.iloc[i]["title_cleaned"])[:50]}')




# 예시 (2): 키워드로 기사 검색  (원하는 키워드로 자유롭게 바꿔도 됨)
query = '이란 호르무즈 원유 봉쇄'
print(f'[검색어] {query}\n▼ 관련 기사 5건')
for i,s in search_keyword(query, 5):
    print(f'  유사도 {s} | [{df.iloc[i]["press"]:5s}] {str(df.iloc[i]["title_cleaned"])[:50]}')


# 활용 — 같은 사건을 여러 언론사가 어떻게 보도했는지 비교할 때 사용할 수 있음

# 텍스트 분류 (Text Classification)  ·  Topic4/03. Sentiment Analysis(3)
# 
# 본문 단어를 보고 카테고리를 맞히는 방식도 함께 사용
# 본문 단어만 보고 카테고리를 얼마나 맞히는지 확인



# 표본이 충분한 카테고리(200건 이상)만 사용
cat_keep = df['article_category'].value_counts()
cat_keep = cat_keep[cat_keep >= 200].index.tolist()
dcls = df[df['article_category'].isin(cat_keep)].copy()

Xc = tfidf.transform(dcls['text_join'])      # 위에서 학습한 TF-IDF 재사용
yc = dcls['article_category'].values
x_tr, x_te, y_tr, y_te = train_test_split(Xc, yc, test_size=0.2, stratify=yc, random_state=42)   # 카테고리 비율이 학습/검증 데이터에 비슷하게 들어가도록 나눔

clf = LogisticRegression(max_iter=1000, C=5.0)   # 뉴스 카테고리 분류 모델
clf.fit(x_tr, y_tr)
y_pred = clf.predict(x_te)
acc = accuracy_score(y_te, y_pred)
print(f'카테고리 자동 분류 정확도: {acc*100:.1f}%  (카테고리 {len(cat_keep)}종)\n')
print(classification_report(y_te, y_pred, digits=3))




# 혼동행렬 히트맵
labels = sorted(set(yc))
cm = confusion_matrix(y_te, y_pred, labels=labels)
plt.figure(figsize=(8,6.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f'카테고리 자동 분류 혼동행렬 (정확도 {acc*100:.1f}%)', fontsize=12, fontweight='bold')
plt.xlabel('예측'); plt.ylabel('실제')
plt.tight_layout(); plt.savefig(OUT/'F14_분류_혼동행렬.png', dpi=140, bbox_inches='tight'); plt.show()


# 본문 단어만으로도 꽤 맞히지만, IT처럼 비슷한 단어가 많은 갈래는 헷갈릴 수 있음

# 텍스트 요약 (Text Summarization)  ·  Topic4/04. Text Summarization
# 
# 강의는 spaCy + pytextrank를 사용하지만, 한국어 모델(ko_core_news_sm) 설치가 필요함
# 여기서는 추가 설치 없이 돌아가도록 TextRank 방식을 직접 코드로 적음
# 원리: 서로 비슷한 문장끼리 연결한 뒤, 중심에 가까운 문장을 요약문으로 선택



from kiwipiepy import Kiwi
_kiwi = Kiwi()   # 기사 본문을 문장으로 나누기 위해 사용

def textrank_summary(text, n=3):
    '''기사 본문에서 핵심 문장 n개를 추출함 (TextRank).'''
    sents = [s.text.strip() for s in _kiwi.split_into_sents(text) if len(s.text.strip()) > 10]
    if len(sents) <= n:
        return sents
    vec = TfidfVectorizer().fit_transform(sents)      # 문장을 숫자로 바꿔 서로 얼마나 비슷한지 비교
    sim = cosine_similarity(vec); np.fill_diagonal(sim, 0)   # 자기 자신과의 비교는 제외
    scores = nx.pagerank(nx.from_numpy_array(sim), max_iter=100)  # 다른 문장들과 많이 연결되는 문장에 높은 점수 부여
    ranked = sorted(((scores[i], i, s) for i,s in enumerate(sents)), reverse=True)
    top = sorted(ranked[:n], key=lambda x: x[1])      # 상위 n개를 원문 순서로 재정렬 — 앞뒤 문맥 살림
    return [s for _,_,s in top]




# 예시: 본문이 충분히 긴 기사를 골라 3문장 요약
sample = df[df['body_length'] > 800].iloc[10]
print(f'[기사] ({sample["press"]}) {sample["title_cleaned"]}\n')
print('▼ TextRank 3문장 요약')
for s in textrank_summary(sample['text'], 3):
    print(' •', s)


# 긴 기사에서 핵심 문장을 빠르게 확인할 때 사용
# (뉴스 본문은 소제목이 마침표 없이 붙는 경우가 있어 첫 문장이 길게 잡힐 수 있음 → body_cleaned 사용·길이 필터로 완화)

# (심화) 미디어그룹별 특징 단어 분석
# 
# 단순 빈도는 'AI·미국'처럼 어디서나 흔한 단어가 상위에 올 수 있음
# 여기서는 다른 그룹보다 해당 그룹에서 더 자주 쓰인 단어를 뽑아 비교



def logodds_group(target, a0=1000.0, min_total=20, topn=12):
    '''선택한 미디어그룹에서 유독 많이 나온 단어를 계산함.'''
    cT, cR = Counter(), Counter()
    for grp,t in zip(df['media_group'], df['tokens_list']):
        (cT if grp==target else cR).update(t)
    vocab = {w for w in set(cT)|set(cR) if cT[w]+cR[w] >= min_total}   # 너무 적게 나온 단어는 비교에서 제외
    tot = sum(cT[w]+cR[w] for w in vocab); nT = sum(cT[w] for w in vocab); nR = sum(cR[w] for w in vocab)
    z = {}
    for w in vocab:
        aw = a0*(cT[w]+cR[w])/tot   # 적게 나온 단어가 너무 크게 보이지 않도록 조정
        lo = np.log((cT[w]+aw)/(nT+a0-cT[w]-aw)) - np.log((cR[w]+aw)/(nR+a0-cR[w]-aw))
        z[w] = lo/np.sqrt(1/(cT[w]+aw) + 1/(cR[w]+aw))
    return sorted(z.items(), key=lambda x: -x[1])[:topn]

groups = df['media_group'].value_counts().index.tolist()
fig, axes = plt.subplots(1, len(groups), figsize=(5*len(groups),6))
if len(groups)==1: axes=[axes]
for ax,g in zip(axes, groups):
    tw = logodds_group(g)
    ll = [w for w,_ in tw][::-1]; zz = [z for _,z in tw][::-1]
    ax.barh(ll, zz, color='#3c6382'); ax.set_title(f'{g}\n특징 단어', fontsize=11, fontweight='bold')
plt.suptitle('미디어그룹별 특징 단어 (해당 그룹 vs 나머지)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(OUT/'F15_미디어그룹_변별어.png', dpi=130, bbox_inches='tight'); plt.show()
