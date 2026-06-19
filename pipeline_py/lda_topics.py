#!/usr/bin/env python
# coding: utf-8

# 필요한 패키지 설치 (한 번만)
# !pip install gensim wordcloud matplotlib pyLDAvis

# 전체 매체 통합 LDA 분석 — 의제 설정 비교
# 분석 대상: 9개 매체 × 4개 매체그룹 (재수집 데이터)
# 경제: 한국경제, 매일경제 / 통신·보도: 연합뉴스(재수집), YTN(재수집) / 정치색: 조선일보, 한겨레 / 지상파: KBS, MBC, SBS
# 기간: 2026.05.05 ~ 05.11 (1주)
# 분석 흐름: CSV 불러오기 → 기사별 단어 묶음 만들기 → 주제별 단어 묶음 분석 → 키워드와 비중 확인 → 결과 저장
# 주의: 연합뉴스는 1차 수집 때 문제가 있어서 재수집함 (3,959건 → 1,519건). 일부 영어 토큰이 섞일 수 있지만 LDA 결과에는 큰 영향 없음

import pandas as pd
from pathlib import Path

# 같은 폴더의 설정 파일 불러오기
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent)); import config as _cfg
DATA_DIR = _cfg.DATA_DIR

# 이번 분석 기간의 토큰 파일 사용
import unicodedata as _ud
_PINNED='분석토큰_언론사_260505_260511.csv'
TOKEN_FILE = next(p for p in DATA_DIR.iterdir()
                  if p.is_file() and _ud.normalize('NFC', p.name)==_PINNED)
print(f"입력 파일: {TOKEN_FILE.name}")
print(f"수정 시각: {pd.Timestamp.fromtimestamp(TOKEN_FILE.stat().st_mtime)}")

df = pd.read_csv(TOKEN_FILE, encoding='utf-8-sig')   # 한글 CSV 파일 불러오기
df_user = df.copy()
df_user['tokens_list'] = df_user['tokens'].fillna('').str.split()   # 저장된 토큰 문자열을 다시 단어 리스트로 변환

print(f"\n--- 전체 매체그룹 통합 분석 ---")
print(f"총 {len(df_user):,}건\n")

print("[매체그룹별]")
print(df_user['media_group'].value_counts())

print("\n[매체별]")
print(df_user['press'].value_counts())

print("\n[일자별 × 매체그룹]")
print(df_user.groupby(['date', 'media_group']).size().unstack(fill_value=0))

from gensim import corpora

texts = df_user['tokens_list'].tolist()

dictionary = corpora.Dictionary(texts)
print(f"원본 단어 수: {len(dictionary):,}")

# 너무 적게 나오거나 너무 자주 나오는 단어는 토픽 구분에 도움이 적어서 제외
dictionary.filter_extremes(no_below=10, no_above=0.5)
print(f"제외한 뒤 단어 수: {len(dictionary):,}")

# 분석에 넣을 단어 묶음 생성
corpus = [dictionary.doc2bow(text) for text in texts]
print(f"문서 수: {len(corpus):,}")

from gensim.models import LdaModel
import time

NUM_TOPICS = 6   # 팀 통합 분석과 일관성 (이전 K=5에서 변경)

start = time.time()
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    random_state=42,      # 같은 데이터로 다시 돌렸을 때 결과가 크게 흔들리지 않게 고정
    chunksize=200,        # 한 번에 학습할 기사 수
    passes=10,            # 전체 기사 묶음을 10번 반복해서 학습
    alpha='auto',
    per_word_topics=False,
)
print(f"토픽 분석 완료 ({time.time()-start:.1f}초)")

# 각 토픽에서 자주 같이 나온 상위 단어 확인 (단어 옆 숫자는 영향 정도)
print(f"\n=== LDA 토픽 {NUM_TOPICS}개 ===\n")
for topic_id, topic_words in lda_model.print_topics(num_words=12):
    print(f"[Topic {topic_id}]")
    print(f"  {topic_words}\n")

print("=" * 70)
print("토픽별 상위 15개 키워드 (뜻을 파악할 때 볼 키워드)")
print("=" * 70)
for i in range(NUM_TOPICS):
    words = lda_model.show_topic(i, topn=15)
    word_list = [w for w, _ in words]
    print(f"\nTopic {i}:")
    print(f"  {' / '.join(word_list)}")

import numpy as np

dominant_topics = []
topic_probs = []

for bow in corpus:
    topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)   # 각 기사에 대한 토픽 비중 계산
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

# 1) 매체그룹별 비교: 발표에서 가장 중요하게 볼 부분
print("=" * 70)
print("매체그룹별 토픽 비중 (%) - 어떤 그룹이 어떤 의제를 많이 다뤘는지 확인")
print("=" * 70)
tb_group = df_user.groupby(['media_group', 'dominant_topic']).size().unstack(fill_value=0)
tb_group_pct = tb_group.div(tb_group.sum(axis=1), axis=0) * 100   # 매체그룹 안에서 토픽 비중 계산
print(tb_group_pct.round(1))

print("\n--- 실제 기사 수 ---")
print(tb_group)

# 2) 개별 매체별 (참고용 - 9개 매체)
print("\n" + "=" * 70)
print("개별 매체별 토픽 비중 (%) - 참고용")
print("=" * 70)
tb_press = df_user.groupby(['press', 'dominant_topic']).size().unstack(fill_value=0)
tb_press_pct = tb_press.div(tb_press.sum(axis=1), axis=0) * 100
print(tb_press_pct.round(1))

print("=" * 70)
print("일자별 × 토픽별 기사 수")
print("=" * 70)
topic_by_date = df_user.groupby(['date', 'dominant_topic']).size().unstack(fill_value=0)
print(topic_by_date)

from datetime import datetime

# 결과 저장
out_path = DATA_DIR / 'analysis_언론사_lda결과_재수집.csv'

# tokens_list는 분석 중에만 쓰는 컬럼이라 저장 전에 제외
df_save = df_user.drop(columns=['tokens_list'])
df_save.to_csv(out_path, index=False, encoding='utf-8-sig')

print(f"저장 완료: {out_path}")
print(f"행 수: {len(df_save):,}")
print(f"컬럼: {list(df_save.columns)}")
print(f"\n저장 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
