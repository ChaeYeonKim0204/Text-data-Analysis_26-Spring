#!/usr/bin/env python
# coding: utf-8

# # 🎯 (심화) 대상-앵커 감성분석 — 언론사별 의제 보도 차이
# 
# **담당 파트:** 각 언론사별 보도 차이점 / **의제:** 이란–미국 호르무즈 해협 긴장 (2026.05.05~11)
# 
# ## 왜 이 분석이 필요한가
# 일반 감성분석(KNU 사전)은 *기사 전체*의 긍/부정만 알려줄 뿐, **"어떤 대상(국가)에 대해" "어떤 의제에 대해"** 긍·부정인지는 구분하지 못합니다.
# 이것을 NLP에서는 **속성기반 감성분석(ABSA, Aspect-Based Sentiment Analysis)** 이라 부릅니다.
# 
# 여기서는 유료 LLM 없이, 수업에서 배운 **KNU 사전 + Kiwi**를 확장하여 ABSA를 **근사(approximate)** 합니다.
# 
# > **원리 (대상-앵커 / target-anchored 방식)**
# > 1. 기사를 **문장 단위**로 분리 (`Topic2/02` Kiwi)
# > 2. 각 문장에 등장한 **행위자(국가)** 와 **의제(aspect)** 를 사전으로 탐지
# > 3. 그 **문장의 KNU 감성**을 등장한 대상에 귀속 (`Topic4/02` 방식)
# > 4. **미디어그룹별로 집계** → "그룹 × 국가", "그룹 × 의제" 감성 매트릭스
# 
# > ⚠️ **방법론 한계 (반드시 명시)**
# > - 한 문장에 두 대상이 동시 등장하면 감성이 둘 다에 귀속됨(타깃 모호) → **'단독 언급' 문장만** 따로 집계해 보완
# > - 측정값은 **'그 대상 주변의 어휘 톤'** 이지 진짜 *스탠스(입장)* 가 아님. 예: 이란을 부정적으로 서술해도 이란 *입장을 인용* 한 것일 수 있음
# > - 진짜 스탠스/입장까지 보려면 로컬 딥러닝(KoBERT·KLUE NLI) 또는 LLM이 필요 (노트북 끝 참고)

# In[1]:


import json, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt, seaborn as sns
# [PIPELINE strip] %matplotlib inline
plt.rcParams['axes.unicode_minus']=False  # 폰트 등록은 _cfg import 후(아래)
plt.rcParams['figure.dpi']=110
from kiwipiepy import Kiwi

import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent)); import config as _cfg
QA=_cfg.DATA_DIR  # [PIPELINE patch] 디렉토리만 교체, 파일명 리터럴 verbatim
OUT=_cfg.CHART_OUT; OUT.mkdir(parents=True, exist_ok=True)
KNU=_cfg.KNU_PATH
import matplotlib.font_manager as _fm; _fm.fontManager.addfont(str(_cfg.FONT_PATH)); plt.rc('font', family='NanumGothic')  # [fix] _cfg 정의 후 폰트 등록(use-before-import 버그 수정)


# In[2]:


# 데이터 + KNU 감성사전
pre=pd.read_csv(QA/'전처리_본문_언론사_260505_260511.csv',encoding='utf-8-sig')
pre=pre[~pre['is_weather']&~pre['is_closing']&~pre['is_within_press_dup']&~pre['is_foreign']].copy()
sw=json.load(open(KNU,encoding='utf-8-sig')); dse=pd.DataFrame(sw); dse['polarity']=dse['polarity'].astype(int)
# 단어→극성 (|극성| 큰 값 우선)
POL=(dse.sort_values('polarity',key=lambda x:x.abs(),ascending=False)
       .drop_duplicates('word',keep='first').set_index('word')['polarity'].to_dict())

# 호르무즈/이란 의제 기사만 추출
issue=pre[pre['text'].fillna('').str.contains('호르무즈|이란')].copy().reset_index(drop=True)
print(f'의제 기사: {len(issue)}건 / 미디어그룹별:', dict(issue['media_group'].value_counts()))


# In[3]:


# 행위자(국가)·의제(aspect) 사전  — 도메인 지식으로 직접 구성
ACTOR={'미국':['미국','워싱턴','트럼프','백악관','미군','펜타곤','국무부'],
       '이란':['이란','테헤란','혁명수비대','이란군','이란산'],
       '이스라엘':['이스라엘','네타냐후']}
ASPECT={'원유·에너지':['원유','유가','기름','석유','에너지','정유','휘발유'],
        '봉쇄·항로':['봉쇄','호르무즈','해협','항로','통항','차단','선박'],
        '군사·충돌':['공격','교전','미사일','폭격','전쟁','무력','타격','보복','도발','군사'],
        '외교·제재':['협상','외교','회담','제재','핵','합의','대화','중재']}

kiwi=Kiwi()
def sent_senti(sent):
    '''문장의 KNU 감성합·토큰수 (명사·동사·형용사 등 내용어)'''
    toks=[t.form for t in kiwi.tokenize(sent) if t.tag.startswith(('N','V','M','I','X'))]
    s=sum(POL.get(w,0) for w in toks); n=len(toks)
    return s, n


# In[4]:


# 문장 순회 → 대상별 감성 인스턴스 수집  (약 50초 소요)
rec_actor=[]; rec_aspect=[]
for _,row in issue.iterrows():
    grp=row['media_group']; txt=row['text']
    if not isinstance(txt,str): continue
    for s in kiwi.split_into_sents(txt):
        st=s.text.strip()
        if len(st)<10: continue
        actors=[a for a,kws in ACTOR.items() if any(k in st for k in kws)]
        aspects=[a for a,kws in ASPECT.items() if any(k in st for k in kws)]
        if not actors and not aspects: continue
        sc,nt=sent_senti(st)
        if nt==0: continue
        s100=sc/nt*100                       # 100토큰당 정규화
        for a in actors:  rec_actor.append((grp,a,s100,len(actors)==1))   # excl=단독언급여부
        for a in aspects: rec_aspect.append((grp,a,s100))
da=pd.DataFrame(rec_actor,columns=['group','actor','senti','excl'])
dp=pd.DataFrame(rec_aspect,columns=['group','aspect','senti'])
GROUP_ORDER=['지상파','통신·보도','경제','정치색']
print(f'행위자 문장 {len(da):,} / 의제 문장 {len(dp):,}')


# ## 1) 미디어그룹 × 행위자(국가) 감성 — *어떤 국가를 어떤 톤으로 다루나*

# In[5]:


# 타깃이 명확한 '단독 언급' 문장만 사용
piv=da[da['excl']].pivot_table(index='group',columns='actor',values='senti',aggfunc='mean').reindex(GROUP_ORDER)[['미국','이란','이스라엘']]
cnt=da[da['excl']].pivot_table(index='group',columns='actor',values='senti',aggfunc='size').reindex(GROUP_ORDER)[['미국','이란','이스라엘']]
plt.figure(figsize=(8,5))
annot=piv.round(2).astype(str)+'\n(n='+cnt.fillna(0).astype(int).astype(str)+')'
sns.heatmap(piv,annot=annot,fmt='',cmap='RdYlGn',center=0,linewidths=.5,cbar_kws={'label':'평균 감성(100토큰당)'})
plt.title('미디어그룹 × 행위자(국가) 감성 — 단독 언급 문장',fontsize=12,fontweight='bold'); plt.xlabel(''); plt.ylabel('')
plt.tight_layout(); plt.savefig(OUT/'A01_그룹별_국가_감성.png',dpi=140,bbox_inches='tight'); plt.show()


# **해석** — **미국**은 경제그룹(+0.28)이 가장 우호적, 정치색(−0.17)만 부정. **이란**은 전 그룹 부정이며 통신·보도(−0.50)·정치색(−0.45)이 특히 강함(지상파는 거의 중립).
# *(경제×이스라엘은 n=12로 표본 부족 → 해석 제외)*

# ## 2) 미디어그룹 × 의제(aspect) 감성 — *어떤 의제를 위기로/기회로 보나*

# In[6]:


order_a=['봉쇄·항로','군사·충돌','원유·에너지','외교·제재']
pivp=dp.pivot_table(index='group',columns='aspect',values='senti',aggfunc='mean').reindex(GROUP_ORDER)[order_a]
cntp=dp.pivot_table(index='group',columns='aspect',values='senti',aggfunc='size').reindex(GROUP_ORDER)[order_a]
plt.figure(figsize=(9,5))
annot=pivp.round(2).astype(str)+'\n(n='+cntp.fillna(0).astype(int).astype(str)+')'
sns.heatmap(pivp,annot=annot,fmt='',cmap='RdYlGn',center=0,linewidths=.5,cbar_kws={'label':'평균 감성(100토큰당)'})
plt.title('미디어그룹 × 의제(aspect) 감성',fontsize=12,fontweight='bold'); plt.xlabel(''); plt.ylabel('')
plt.tight_layout(); plt.savefig(OUT/'A02_그룹별_의제_감성.png',dpi=140,bbox_inches='tight'); plt.show()


# **해석** — 봉쇄·항로/군사·충돌은 전 그룹 부정(갈등 소재). **원유·에너지는 경제그룹만 중립(+0.05)** — 타 그룹이 '에너지 위기'로 보는 의제를 경제지는 '시장 변수'로 다룸. 외교·제재는 대체로 긍정(협상 기대)이나 정치색만 부정(−0.18).

# ## 3) 미디어그룹별 행위자 언급 비중 — *누구에게 더 주목하나*

# In[7]:


share=da.pivot_table(index='group',columns='actor',values='senti',aggfunc='size').reindex(GROUP_ORDER)[['미국','이란','이스라엘']]
share_pct=share.div(share.sum(axis=1),axis=0)*100
plt.figure(figsize=(9,5)); bottom=np.zeros(len(share_pct)); colors={'미국':'#3c6382','이란':'#b71540','이스라엘':'#e58e26'}
for a in ['미국','이란','이스라엘']:
    plt.bar(share_pct.index,share_pct[a],bottom=bottom,label=a,color=colors[a])
    for i,v in enumerate(share_pct[a].values):
        if v>3: plt.text(i,bottom[i]+v/2,f'{v:.0f}%',ha='center',va='center',color='white',fontsize=9)
    bottom+=share_pct[a].values
plt.title('미디어그룹별 행위자 언급 비중',fontsize=12,fontweight='bold'); plt.ylabel('언급 문장 비중(%)')
plt.legend(title='행위자',bbox_to_anchor=(1.01,1),loc='upper left')
plt.tight_layout(); plt.savefig(OUT/'A03_그룹별_행위자_언급비중.png',dpi=140,bbox_inches='tight'); plt.show()


# **해석** — 경제그룹은 **미국(51.6%)** 을 이란보다 더 주목(미국발 정책·시장영향 관점), 정치색은 **이스라엘 언급(7.8%)** 이 타 그룹의 2배(중동 분쟁 구도 관심).

# ## 📌 종합: 호르무즈 의제에서 드러난 언론사 그룹별 보도 차이
# 
# | 그룹 | 행위자 톤 | 의제 관점 | 한 줄 요약 |
# |------|----------|-----------|-----------|
# | **경제** | 미국에 가장 우호적(+0.28) | 원유=시장 변수(중립) | 미국·시장 중심의 *경제 영향* 프레임 |
# | **지상파** | 국가별 톤 가장 중립 | 위기 소재도 담담 | 사실 전달 중심의 *균형* 보도 |
# | **통신·보도** | 이란에 가장 부정(−0.50) | 군사·충돌 강조(−1.30) | 속보성 *분쟁/충돌* 중심 |
# | **정치색** | 미국도 부정(−0.17), 이스라엘 주목 | 외교조차 부정 | *분쟁 구도·비판* 프레임 |
# 
# ---
# 
# ## 🔧 더 정밀하게 보려면 (로컬 딥러닝, 무료·설치만 필요)
# 유료 LLM 대신, 사전학습 한국어 모델로 **진짜 스탠스/타깃 감성**을 추정할 수 있습니다.
# ```bash
# pip install transformers sentencepiece    # torch는 이미 설치됨
# ```
# - **Zero-shot NLI** (예: `pongjin/roberta_with_kornli`): 문장 + 가설("이 문장은 이란에 부정적이다")의 *함의(entailment)* 확률로 타깃 스탠스 추정
# - **KoBERT/KLUE 감성 분류기**: 문장 단위 긍/부정을 사전보다 정확히 분류
# 이 방식은 **딥러닝 기반 텍스트분석**이며(=LLM과 같은 계열), 로컬 실행이라 비용·재현성 문제가 없습니다.
