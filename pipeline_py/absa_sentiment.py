#!/usr/bin/env python
# coding: utf-8

# (심화) 대상별 감성분석 — 언론사별 의제 보도 차이
#
# 각 언론사별 보도 차이점 / 의제: 이란–미국 호르무즈 해협 긴장 (2026.05.05~11)
#
# 일반 KNU 감성분석은 기사 전체의 긍/부정만 파악 — 국가·의제 귀속 구분 한계
# 문장 안에 등장한 대상별 감성 점수 분리
#
# 진행 순서
# 1. 기사를 문장 단위로 분리 (Kiwi)
# 2. 각 문장에서 행위자(국가)와 의제 단어 찾기
# 3. 해당 문장의 KNU 감성 점수를 등장한 대상에 붙여서 계산
# 4. 미디어그룹별로 평균 감성 비교
#
# 한 문장에 대상이 여러 개 나오면 감성이 누구에 대한 것인지 애매해서, 대상이 하나만 나온 문장도 따로 집계함
# 이 값은 언론사의 실제 입장보다 대상 주변 단어의 긍정/부정 정도. 더 정확한 해석에는 다른 분석 방법 필요



import json, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt, seaborn as sns
plt.rcParams['axes.unicode_minus']=False  # 그래프에서 마이너스 기호가 깨지지 않게 설정
plt.rcParams['figure.dpi']=110
from kiwipiepy import Kiwi

import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent)); import config as _cfg
QA=_cfg.DATA_DIR
OUT=_cfg.CHART_OUT; OUT.mkdir(parents=True, exist_ok=True)
KNU=_cfg.KNU_PATH
import matplotlib.font_manager as _fm; _fm.fontManager.addfont(str(_cfg.FONT_PATH)); plt.rc('font', family='NanumGothic')  # 그래프 한글 표시용 폰트




# 데이터 + KNU 감성사전
pre=pd.read_csv(QA/'전처리_본문_언론사_260505_260511.csv',encoding='utf-8-sig')
# 날씨·클로징·중복·외국어 기사는 분석에서 제외
pre=pre[~pre['is_weather']&~pre['is_closing']&~pre['is_within_press_dup']&~pre['is_foreign']].copy()
sw=json.load(open(KNU,encoding='utf-8-sig')); dse=pd.DataFrame(sw); dse['polarity']=dse['polarity'].astype(int)
# 감성 점수가 큰 단어를 먼저 사용
POL=(dse.sort_values('polarity',key=lambda x:x.abs(),ascending=False)
       .drop_duplicates('word',keep='first').set_index('word')['polarity'].to_dict())

# 호르무즈/이란 의제 기사만 추출
issue=pre[pre['text'].fillna('').str.contains('호르무즈|이란')].copy().reset_index(drop=True)
print(f'의제 기사: {len(issue)}건 / 미디어그룹별:', dict(issue['media_group'].value_counts()))




# 문장에서 찾을 행위자(국가)와 의제 단어
ACTOR={'미국':['미국','워싱턴','트럼프','백악관','미군','펜타곤','국무부'],
       '이란':['이란','테헤란','혁명수비대','이란군','이란산'],
       '이스라엘':['이스라엘','네타냐후']}
ASPECT={'원유·에너지':['원유','유가','기름','석유','에너지','정유','휘발유'],
        '봉쇄·항로':['봉쇄','호르무즈','해협','항로','통항','차단','선박'],
        '군사·충돌':['공격','교전','미사일','폭격','전쟁','무력','타격','보복','도발','군사'],
        '외교·제재':['협상','외교','회담','제재','핵','합의','대화','중재']}

kiwi=Kiwi()
def sent_senti(sent):
    '''문장을 단어로 분리해 감성 점수와 단어 개수 계산'''
    # 감성 단어가 들어갈 수 있는 품사만 골라 점수 계산
    toks=[t.form for t in kiwi.tokenize(sent) if t.tag.startswith(('N','V','M','I','X'))]
    s=sum(POL.get(w,0) for w in toks); n=len(toks)
    return s, n




# 문장 순회, 대상별 감성 결과 수집 (약 50초 소요)
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
        s100=sc/nt*100                       # 문장 길이 차이를 줄이기 위해 단어 100개 기준 점수로 계산
        for a in actors:  rec_actor.append((grp,a,s100,len(actors)==1))   # 행위자가 하나만 나온 문장인지 표시
        for a in aspects: rec_aspect.append((grp,a,s100))
da=pd.DataFrame(rec_actor,columns=['group','actor','senti','excl'])
dp=pd.DataFrame(rec_aspect,columns=['group','aspect','senti'])
GROUP_ORDER=['지상파','통신·보도','경제','정치색']   # 그래프마다 같은 순서로 표시
print(f'행위자 문장 {len(da):,} / 의제 문장 {len(dp):,}')


# 1) 미디어그룹 × 행위자(국가) 감성 — 국가별 보도 분위기 비교



# 단독 언급 문장만 사용 — 두 행위자가 한 문장에 같이 나오면 어느 쪽 감성인지 불분명
piv=da[da['excl']].pivot_table(index='group',columns='actor',values='senti',aggfunc='mean').reindex(GROUP_ORDER)[['미국','이란','이스라엘']]
cnt=da[da['excl']].pivot_table(index='group',columns='actor',values='senti',aggfunc='size').reindex(GROUP_ORDER)[['미국','이란','이스라엘']]
plt.figure(figsize=(8,5))
annot=piv.round(2).astype(str)+'\n(n='+cnt.fillna(0).astype(int).astype(str)+')'
sns.heatmap(piv,annot=annot,fmt='',cmap='RdYlGn',center=0,linewidths=.5,cbar_kws={'label':'평균 감성(단어 100개 기준)'})
plt.title('미디어그룹 × 행위자(국가) 감성 — 단독 언급 문장',fontsize=12,fontweight='bold'); plt.xlabel(''); plt.ylabel('')
plt.tight_layout(); plt.savefig(OUT/'A01_그룹별_국가_감성.png',dpi=140,bbox_inches='tight'); plt.show()


# 미국 관련 문장은 경제그룹에서 점수가 조금 높고, 정치색 그룹에서는 낮게 나옴
# 이란 관련 문장은 대부분 그룹에서 부정 쪽. 경제×이스라엘은 표본이 적어 해석 제외

# 2) 미디어그룹 × 의제 감성 — 의제별 보도 분위기 비교



order_a=['봉쇄·항로','군사·충돌','원유·에너지','외교·제재']
pivp=dp.pivot_table(index='group',columns='aspect',values='senti',aggfunc='mean').reindex(GROUP_ORDER)[order_a]
cntp=dp.pivot_table(index='group',columns='aspect',values='senti',aggfunc='size').reindex(GROUP_ORDER)[order_a]
plt.figure(figsize=(9,5))
annot=pivp.round(2).astype(str)+'\n(n='+cntp.fillna(0).astype(int).astype(str)+')'
sns.heatmap(pivp,annot=annot,fmt='',cmap='RdYlGn',center=0,linewidths=.5,cbar_kws={'label':'평균 감성(단어 100개 기준)'})
plt.title('미디어그룹 × 의제 감성',fontsize=12,fontweight='bold'); plt.xlabel(''); plt.ylabel('')
plt.tight_layout(); plt.savefig(OUT/'A02_그룹별_의제_감성.png',dpi=140,bbox_inches='tight'); plt.show()


# 봉쇄·항로, 군사·충돌은 거의 모든 그룹에서 부정 쪽으로 나옴
# 원유·에너지는 경제그룹에서만 0에 가까웠고, 경제지는 이 주제를 위기보다 시장 변수에 가깝게 다룬 것으로 해석

# 3) 미디어그룹별 행위자 언급 비중 — 누구에게 더 주목하나



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


# 경제그룹은 이란보다 미국을 더 많이 언급함. 미국 정책이 시장에 미치는 영향을 더 많이 다룬 결과로 해석
# 정치색 그룹은 다른 그룹보다 이스라엘 언급이 조금 더 많았음

# 호르무즈 의제에서 드러난 언론사 그룹별 보도 차이
# 경제 — 미국 관련 문장이 비교적 긍정, 원유·에너지는 시장 변수로 보는 경향
# 지상파 — 국가별 점수가 0에 가까워 사실 전달 중심에 가까움
# 통신·보도 — 이란 관련 문장과 군사·충돌 의제에서 부정 단어가 많이 나타남
# 정치색 — 미국과 외교·제재 의제에서도 부정 단어가 비교적 많이 나타남
