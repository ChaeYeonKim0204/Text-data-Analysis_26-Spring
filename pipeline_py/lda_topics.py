#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 필요한 패키지 설치 (한 번만)
# [PIPELINE strip] get_ipython().system(' pip install gensim wordcloud matplotlib pyLDAvis')


# # 전체 매체 통합 LDA 분석 — 의제 설정 비교
# 
# 분석 대상: 9개 매체 × 4개 매체그룹 (재수집 데이터)
# - 경제: 한국경제, 매일경제
# - 통신·보도: 연합뉴스(재수집), YTN(재수집)
# - 정치색: 조선일보, 한겨레
# - 지상파: KBS, MBC, SBS
# 
# 기간: 2026.05.05 ~ 05.11 (1주)
# 
# ## 분석 흐름
# 1. 데이터 로딩 + 매체 분포 검증
# 2. LDA 입력 준비 (Dictionary, Corpus)
# 3. LDA 학습 (토픽 수 K=6)
# 4. 토픽별 키워드 + 매체그룹별 비중
# 5. 결과 저장
# 
# > 주의: 연합뉴스가 1차 수집 시 이상 발생하여 재수집됨 (3,959 → 1,519건).
# > 일부 영문 토큰이 섞여 있을 수 있으나 LDA 결과에 큰 영향은 없음.

# In[2]:


"""
데이터 로딩 — 고정 분석기간 분석토큰 리터럴 파일명 매칭(NFC 정규화)
"""
import pandas as pd
import glob
from pathlib import Path

import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent)); import config as _cfg
DATA_DIR = _cfg.DATA_DIR  # [PIPELINE patch] repo-상대 config

# [PIPELINE patch] 원본의 mtime-glob 자동감지(최신 파일 우선)는 비결정적이라 제거 → 고정 분석기간 정확 리터럴 파일명
import unicodedata as _ud
_PINNED='분석토큰_언론사_260505_260511.csv'
TOKEN_FILE = next(p for p in DATA_DIR.iterdir()
                  if p.is_file() and _ud.normalize('NFC', p.name)==_PINNED)
print(f"입력 파일: {TOKEN_FILE.name}")
print(f"수정 시각: {pd.Timestamp.fromtimestamp(TOKEN_FILE.stat().st_mtime)}")

df = pd.read_csv(TOKEN_FILE, encoding='utf-8-sig')
df_user = df.copy()
df_user['tokens_list'] = df_user['tokens'].fillna('').str.split()

print(f"\n--- 전체 매체그룹 통합 분석 ---")
print(f"총 {len(df_user):,}건\n")

print("[매체그룹별]")
print(df_user['media_group'].value_counts())

print("\n[매체별]")
print(df_user['press'].value_counts())

print("\n[일자별 × 매체그룹]")
print(df_user.groupby(['date', 'media_group']).size().unstack(fill_value=0))


# In[3]:


"""
LDA 입력 준비
- Dictionary: 단어 → ID 매핑
- 너무 드물거나 흔한 단어 필터 (no_below, no_above)
- Corpus: 각 문서를 (단어ID, 빈도) 튜플 리스트로
"""
from gensim import corpora

texts = df_user['tokens_list'].tolist()

dictionary = corpora.Dictionary(texts)
print(f"원본 단어 수: {len(dictionary):,}")

# 10건 미만 출현 단어 제거 + 50% 이상 문서에 등장하는 흔한 단어 제거
dictionary.filter_extremes(no_below=10, no_above=0.5)
print(f"필터 후 단어 수: {len(dictionary):,}")

# BoW(Bag of Words) Corpus
corpus = [dictionary.doc2bow(text) for text in texts]
print(f"문서 수: {len(corpus):,}")


# In[4]:


"""
LDA 학습 — 토픽 수 K=6
- 팀의 통합 분석(K=6, 시드 안정성 게이트)과 일관성 유지
- random_state=42: 재현 가능
- passes=10: 학습 반복 (정확도와 속도 균형)
- alpha='auto': 토픽 분포 자동 학습
"""
from gensim.models import LdaModel
import time

NUM_TOPICS = 6   # 팀 통합 분석과 일관성 (이전 K=5에서 변경)

start = time.time()
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    random_state=42,
    chunksize=200,
    passes=10,
    alpha='auto',
    per_word_topics=False,
)
print(f"학습 완료 ({time.time()-start:.1f}초)")

# 각 토픽의 상위 단어 (가중치 포함)
print(f"\n=== LDA 토픽 {NUM_TOPICS}개 ===\n")
for topic_id, topic_words in lda_model.print_topics(num_words=12):
    print(f"[Topic {topic_id}]")
    print(f"  {topic_words}\n")


# In[5]:


"""
토픽 해석을 위해 상위 키워드만 깔끔하게
사람이 직접 토픽 이름 붙이는 단계
"""
print("=" * 70)
print("토픽별 상위 15개 키워드 (해석용)")
print("=" * 70)
for i in range(NUM_TOPICS):
    words = lda_model.show_topic(i, topn=15)
    word_list = [w for w, _ in words]
    print(f"\nTopic {i}:")
    print(f"  {' / '.join(word_list)}")


# In[6]:


"""
각 기사가 어느 토픽에 속하는지 (가장 확률 높은 토픽)
"""
import numpy as np

dominant_topics = []
topic_probs = []

for bow in corpus:
    topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
    topic_dist = sorted(topic_dist, key=lambda x: -x[1])
    if topic_dist:
        dominant_topics.append(topic_dist[0][0])
        topic_probs.append(topic_dist[0][1])
    else:
        dominant_topics.append(-1)
        topic_probs.append(0.0)

df_user['dominant_topic'] = dominant_topics
df_user['topic_prob'] = topic_probs

# 토픽별 기사 수 + 비중
print("=" * 70)
print("토픽별 기사 분포 (1주일 의제 비중)")
print("=" * 70)
topic_dist = df_user['dominant_topic'].value_counts().sort_index()
for topic_id, count in topic_dist.items():
    print(f"  Topic {topic_id}: {count:>5,}건 ({count/len(df_user)*100:>5.1f}%)")

print(f"\n→ 1주일 최대 의제: Topic {topic_dist.idxmax()} ({topic_dist.max():,}건)")


# In[7]:


"""
매체그룹별 토픽 비중 - 발표 핵심 슬라이드 한 장
"""
# 1) 매체그룹별 (4개 그룹 비교 - 발표 메인)
print("=" * 70)
print("매체그룹별 토픽 비중 (%) - 의제설정 분석 핵심")
print("=" * 70)
tb_group = df_user.groupby(['media_group', 'dominant_topic']).size().unstack(fill_value=0)
tb_group_pct = tb_group.div(tb_group.sum(axis=1), axis=0) * 100
print(tb_group_pct.round(1))

print("\n--- 절대 건수 ---")
print(tb_group)

# 2) 개별 매체별 (참고용 - 9개 매체)
print("\n" + "=" * 70)
print("개별 매체별 토픽 비중 (%) - 참고용")
print("=" * 70)
tb_press = df_user.groupby(['press', 'dominant_topic']).size().unstack(fill_value=0)
tb_press_pct = tb_press.div(tb_press.sum(axis=1), axis=0) * 100
print(tb_press_pct.round(1))


# In[8]:


"""
1주일 동안 어느 토픽이 언제 폭증했나
- 4개 매체그룹 통합 기준
"""
print("=" * 70)
print("일자별 × 토픽별 기사 수")
print("=" * 70)
topic_by_date = df_user.groupby(['date', 'dominant_topic']).size().unstack(fill_value=0)
print(topic_by_date)


# In[9]:


"""
LDA 결과를 CSV로 저장 - 감성분석/요약/워드클라우드에서 활용
파일명에 '재수집' 표시 (이전 결과와 구분)
"""
from datetime import datetime

# 신규 데이터 LDA 결과임을 명시하는 파일명
out_path = DATA_DIR / 'analysis_언론사_lda결과_재수집.csv'

# 저장 전에 tokens_list 컬럼 제거 (CSV 저장 시 문자열로 깨지므로)
df_save = df_user.drop(columns=['tokens_list'])
df_save.to_csv(out_path, index=False, encoding='utf-8-sig')

print(f"저장 완료: {out_path}")
print(f"행 수: {len(df_save):,}")
print(f"컬럼: {list(df_save.columns)}")
print(f"\n저장 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

