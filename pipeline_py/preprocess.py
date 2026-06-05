# -*- coding: utf-8 -*-
"""
① 전처리 — 통합 본문을 정제하고 분석용 플래그를 붙임
reads : data/news/통합_본문_bs4_언론사_260505_260511.csv (기사 11,990건)
writes: data/news/전처리_본문_언론사_260505_260511.csv (정제본+text 합본, is_weather/is_closing/중복/is_foreign 플래그)
원본  : notebooks/crawling/press/언론사_네이버뉴스_전처리_colab.ipynb (nbconvert, 로직 verbatim)
"""
# Colab에서 실행할 때만 아래 3줄 주석 해제 — 로컬/WSL에서는 그대로 두기
# from google.colab import drive
# drive.mount('/content/drive')

from pathlib import Path
import os
import re
import unicodedata

import pandas as pd

# Colab 기본 경로 — Drive가 마운트돼 있지 않으면 except로 빠져 로컬/WSL 경로 사용
# [PIPELINE patch] 머신고정 try/except 제거 → repo-상대 config (이식성)
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent)); import config as _cfg
PROJECT_DIR = _cfg.PROJECT_DIR
DATA_DIR = _cfg.DATA_DIR

os.chdir(PROJECT_DIR)
print(f'현재 작업 폴더: {Path.cwd()}')

# 언론사 트랙은 news/ 폴더 사용 (통신3사는 notebooks/crawling/data/)

# --- 전처리 정책 (필요 시 여기만 수정) ---
MEDIA_GROUP = '지상파'              # 통합본에 media_group이 없을 때의 기본값
# 통합본에 media_group이 없을 때만 사용하는 press->그룹 매핑 (4그룹 합본이면 통합본 값을 그대로 씀)
MEDIA_GROUP_MAP = {
    'KBS': '지상파', 'MBC': '지상파', 'SBS': '지상파',
    'YTN': '통신·보도', '연합뉴스': '통신·보도',
    '한국경제': '경제', '매일경제': '경제',
    '조선일보': '정치색', '한겨레': '정치색',
}
PERIOD_START = '2026-05-05'         # 분석 기간 시작 (이 날짜 포함)
PERIOD_END = '2026-05-11'           # 분석 기간 끝 (이 날짜 포함)
MIN_BODY_LEN_AFTER_CLEAN = 20       # 태그 제거 후 이 글자 수 미만이면 실질 내용 없음으로 보고 제외
print(f'DATA_DIR: {DATA_DIR}')

# 입력 통합본 찾기 — 한글 파일명 NFD/NFC 차이를 줄이려고 정규화 후 매칭
def normalize_name(path):
    return unicodedata.normalize('NFC', path.name)

# [PIPELINE patch] 자동선택([-1]·mtime 비결정성 제거) → 고정 분석기간 정확 리터럴 파일명
_PINNED = '통합_본문_bs4_언론사_260505_260511.csv'
INPUT_PATH = next(p for p in DATA_DIR.iterdir()
                  if p.is_file() and normalize_name(p) == _PINNED)
print(f'입력 통합본: {INPUT_PATH.name}')

df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')
print(f'원본 행수: {len(df)}')
print('컬럼:', list(df.columns))
print('press별:', df['press'].value_counts().to_dict())

# --- 메타 컬럼 정리 ---

# media_group은 통합본에 있으면 그대로 쓰고, 없을 때만 매핑표/기본값으로 채움
if 'media_group' not in df.columns:
    df['media_group'] = df['press'].map(MEDIA_GROUP_MAP).fillna(MEDIA_GROUP)
# source_file은 통합본에 있으면 그대로, 없으면 press와 기간으로 이름 복원 — 어느 수집 파일에서 왔는지 추적용
if 'source_file' not in df.columns:
    df['source_file'] = '본문_bs4_' + df['press'].astype(str) + '_' + df['source_period'].astype(str) + '.csv'

# category에서 상위분류만 남긴 article_category 생성 (예: '스포츠/kbaseball' -> '스포츠', '생활/문화' -> '생활')
# 원래 값은 article_category_full에 보존, 빈 값은 '미분류'로 채움
df['article_category_full'] = df['category']
df['article_category'] = (
    df['category'].astype(str).str.replace(r'/.*', '', regex=True)
      .where(df['category'].notna(), '미분류')
)

# pubdate를 날짜형(datetime)으로 변환하고 날짜만 뽑은 date 컬럼 추가
df['pubdate'] = pd.to_datetime(df['pubdate'], errors='coerce')
df['date'] = df['pubdate'].dt.date
print('article_category 분포:')
print(df['article_category'].value_counts())


# --- 본문 정제 함수 clean_body ---
# 기사 본문이 아닌 부분(방송 표시, 기자·사진 출처, 제보·저작권 안내 등)만 골라 제거
# 핵심 원칙 — 사진/기자/이메일/제공 같은 단어가 들어있다고 무조건 지우지 않음
#   기사 맨 앞·맨 끝이거나, 정해진 형태(매체명이 든 출처 등)일 때만 제거
#   그래야 발언 인용 [트럼프/미국 대통령 : ...], 본문 속 메일 info@PGSA.ir, <장자(莊子)>, ㎞·한자 같은 실제 내용을 잘못 지우지 않음
#   애매하면 남기고 분석 단계 불용어로 거름

def _strip_dateline(t):
    # 연합뉴스·뉴스1·뉴시스 기사 머리 "(서울=연합뉴스) 김윤구 기자 =" 형태 제거
    # 매체 이름과 '기자 ='가 같이 있을 때만 지우므로 본문 속 일반 (A=B) 괄호는 안 건드림
    # (사진 캡션 문장 뒤에 붙은 경우도 있어 위치 상관없이 제거)
    t = re.sub(r'\(\s*[^()\n]{1,20}=\s*(?:연합뉴스|연합뉴스TV|뉴스1|뉴시스)\)\s*(?:[^\n]{1,30}?\s+)?기자\s*=\s*', ' ', t)
    # 대괄호형 머리 [헤럴드경제=나은정 기자]는 기사 맨 앞일 때만 제거
    t = re.sub(r'^\s*\[[^\]\n]{1,20}=(?:[^\]\n]{1,20}\s+)?[가-힣]{2,5}\s*기자\]\s*', ' ', t)
    return t

def clean_body(text):
    t = str(text)
    # 1. SBS 스포츠 기사 맨 앞 동영상 안내문 제거 ("※ 저작권 관계로 ... [원문에서 영상 보기] https://...")
    t = re.sub(r'^\s*※\s*저작권 관계로 네이버에서 서비스하지 않는 영상입니다\..*?\[원문에서 영상 보기\]\s*https?://\S+\s*', ' ', t, flags=re.S)
    # 2. 방송 코너·지국 이름표 제거 — 기사 맨 앞 [KBS 강릉], [뉴스25], [정치쇼] 등 정해둔 이름만
    t = re.sub(r'^\s*(?:\[KBS\s+[가-힣]{1,12}\]\s*)+', ' ', t)
    t = re.sub(r'^\s*(?:\[(?:뉴스25|뉴스투데이|정오뉴스|5시뉴스|930MBC뉴스|12MBC뉴스|뉴스데스크|뉴스외전)\]\s*)+', ' ', t)
    t = re.sub(r'^\s*(?:\[(?:모닝와이드|12뉴스|뉴스브리핑|주영진\s+뉴스브리핑|정치쇼)\]\s*)+', ' ', t)
    # 3. YTN 라디오 머리말 제거 ([잠시만요], 방송일시·진행·출연자 정보) — 실제 대화(◆◇◎▶▷로 시작)부터 살림
    t = re.sub(r'^\s*(?:\[잠시만요\]|YTN라디오\(FM\s*94\.5\)|\[YTN[^\]]+\]).*?(?=[◆◇◎▶▷□]\s)', ' ', t, flags=re.S)
    t = re.sub(r'□\s*(?:방송일시|진행|출연자)[^◆◇◎▶▷]*', ' ', t)
    # 3-1. YTN 라디오 방송시각 메타 — '□ 방송시각 HH:MM' 형태 추가 제거 (Agent B 보강)
    t = re.sub(r'□\s*방송(?:시각|시간)\s*[:：]\s*\d{1,2}[:：]\d{2}', ' ', t)
    # 4. 방송 대본 화자 표시 제거 (<앵커>, ◀앵커▶, [앵커]/[기자]/[리포트] 등)
    #    단 [트럼프/미국 대통령 : ...]처럼 이름·발언이 든 건 보존
    t = re.sub(r'(?:<\s*(?:앵커|기자|리포트)\s*>|◀\s*(?:앵커|기자|리포트)\s*▶|\[\s*(?:앵커|기자|리포트|답변|녹취|인터뷰)\s*\])', ' ', t)
    t = re.sub(r'\((?:[A-Za-z가-힣]+\s+)?(?:디지털뉴스부|뉴미디어부)\)', ' ', t)   # (SBS 디지털뉴스부) 같은 부서 표기
    # 5. 연합뉴스·뉴스1·뉴시스 기사 머리(데이트라인+기자) 제거 — 위 _strip_dateline 사용
    t = _strip_dateline(t)
    # 6. 사진·자료 출처 표기 제거 — "사진" 단어만으론 안 지우고 '출처/제공/촬영/=' 등 출처 형태일 때만
    #    그래서 본문 인용 [(현금 제공 의혹) CCTV...], [사진 찍었고...] 등은 보존
    t = re.sub(r'\[[^\]\n]{0,20}?(?:사진\s*(?:출처|제공|촬영|=|:|[|ｌlㅣ│])|자료사진|제보사진|캡처|일러스트|그래픽\s*[:=]|(?:로이터|AFP|AP|EPA)=연합뉴스|=\s*(?:연합뉴스|뉴스1|뉴시스)|재판매\s*및\s*DB\s*금지)[^\]\n]{0,80}\]', ' ', t)
    t = re.sub(r'\[[^\]\n]{1,25}\s(?:제공|출처)\]', ' ', t)   # [HMM 제공]처럼 '제공/출처'가 ] 바로 앞에 오는 짧은 출처표기
    t = re.sub(r'\((?:자료)?사진\s*(?:출처|제공|촬영)?\s*[:=][^)]{1,80}\)', ' ', t)
    t = re.sub(r'(?:(?<=\s)|^)(?:자료)?사진\s*[=:|ｌlㅣ│]\s*(?:연합뉴스|뉴스1|뉴시스|게티이미지뱅크|한경\s*DB|EPA|로이터|AFP|AP|EBS|[가-힣A-Za-z]{1,12}\s*(?:제공|DB|SNS|캡처|화면|유튜브))\.?', ' ', t)
    t = re.sub(r'\*기사와 관련 없음', ' ', t)
    t = re.sub(r'(?:사진|이미지|자료\s*사진)\s*\[[^\]\n]{1,30}\]', ' ', t)   # 인라인 사진/이미지 출처 [국립천문대] 등
    # 6에서 맨 앞 사진 출처를 지우면 가려져 있던 기사 머리가 드러날 수 있어 데이트라인 제거를 한 번 더 적용
    t = _strip_dateline(t)
    # 7. 기사 맨 끝 기자 이름·이메일 제거 ("이미아 기자 mia@hankyung.com") — 본문 중간 이메일(info@PGSA.ir)은 그대로
    #    비정상적으로 긴 글에서 정규식이 너무 오래 걸리지 않게 반복 횟수를 최대 6회로 제한
    t = re.sub(r'(?:[가-힣]{2,10}\s*(?:선임\s*|인턴\s*)?(?:기자|특파원|논설위원)(?:\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})?\s*[/·]?\s*){1,6}$', ' ', t)
    t = re.sub(r'(?:[A-Za-z0-9._%+-]+@(?:yna\.co\.kr|hankyung\.com|mk\.co\.kr|hani\.co\.kr|chosun\.com))(?:\s*[A-Za-z0-9._%+-]+@(?:yna\.co\.kr|hankyung\.com|mk\.co\.kr|hani\.co\.kr|chosun\.com))*\s*$', ' ', t)
    t = re.sub(r'\[[가-힣 ]{2,20}(?:스타투데이\s+)?기자\]\s*$', ' ', t)
    # 8. 방송사·신문사별 맨 끝 제보·저작권 안내 블록 제거 — 각 매체가 똑같이 붙이는 정형 문구
    t = re.sub(r'\s*■\s*제보하기.*$', ' ', t, flags=re.S)                                           # KBS
    t = re.sub(r'\s*MBC 뉴스는 24시간 여러분의 제보를 기다립니다\..*$', ' ', t, flags=re.S)          # MBC
    t = re.sub(r'\s*※\s*\W?당신의 제보가 뉴스가 됩니다\W?\[카카오톡\].*?social@ytn\.co\.kr\s*$', ' ', t, flags=re.S)  # YTN
    t = re.sub(r'\s*※\s*자세한 내용은 동영상으로 확인하실 수 있습니다\.?\s*$', ' ', t)                # SBS
    t = re.sub(r'\s*인터뷰 자료의 저작권은 SBS 라디오에 있습니다\..*$', ' ', t, flags=re.S)           # SBS 라디오
    t = re.sub(r'\s*이 기사는 한국경제신문과 금융 AI 전문기업 씽크풀이 공동 개발한 기사 자동생성 알고리즘에 의해 실시간으로 작성된 것입니다\.\s*$', ' ', t)  # 한국경제 자동생성 안내
    t = re.sub(r'\s*당신의 제보가 뉴스로 만들어집니다\..*$', ' ', t, flags=re.S)                      # SBS Biz
    # 9. 저작권 표시 블록 제거 (<저작권자 ⓒ ...>, [Copyright (c) ...], [본 기사는 ...])
    t = re.sub(r'<저작권자\s*ⓒ[^>]+>|\[Copyright\s*\(c\)[^\]]+\]|\[본 기사는 [^\]]+\]', ' ', t)
    # 9-1. 본문 중간에 녹아든 기자 바이라인+이메일 제거 — 5개 매체 도메인만(연합/한경/매경/한겨레/조선), info@PGSA.ir·gmail 등은 보존
    t = re.sub(r'(?:/\s*)?(?:사진\s*=\s*)?[가-힣]{2,5}\s*(?:한경닷컴\s*|닷컴\s*)?기자\s*[A-Za-z0-9._%+-]+@(?:yna\.co\.kr|hankyung\.com|mk\.co\.kr|hani\.co\.kr|chosun\.com)', ' ', t)
    t = re.sub(r'[A-Za-z0-9._%+-]+@(?:yna\.co\.kr|hankyung\.com|mk\.co\.kr|hani\.co\.kr|chosun\.com)', ' ', t)
    # 10. 기사 맨 끝 제작진 표기 제거 ("영상편집:홍길동 그래픽:김철수") — 끝에 있을 때만 지우므로 본문 중간 제목·인용(관계의 기술:, 재구성:)은 보존
    #     값에 공백을 안 넣고 반복 횟수를 8회로 제한해 정규식 속도 확보
    t = re.sub(r'(?:\s*(?<![가-힣])(?:출연|진행|구성|영상취재|영상편집|영상|디자인|촬영|촬영기자|연출|제작|제작진|현장진행|기술감독|자막|그래픽|CG|책임\s*프로듀서|작가)\s*[:：]\s*[가-힣A-Za-z·,]{1,30}){1,8}\s*$', ' ', t)
    t = re.sub(r'\s*\((?:구성|영상편집|연출|제작|촬영기자)\s*:\s*[^()]{1,80}\)\s*$', ' ', t)
    # 11. 맨 앞 구분선(---) 제거
    t = re.sub(r'^\s*---\s*', ' ', t)
    # 11-1. 장식·목록 마커 글로벌 제거 (◆◇▲■◎ 등 — 순수 기호라 본문 손실 없음). ·ㅣ│ⓒ%·…는 본문 정보라 보존
    t = re.sub(r'[■□◇◆◎▲▽△▷◁▶◀☞☜※★☆◈➤●]+', ' ', t)
    # 12. 줄바꿈·연속 공백을 한 칸으로 정리 (이 단계에서만 공백 정리)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# 원본 title/body는 그대로 두고 정제본(title_cleaned/body_cleaned)과 합본(text)을 새 컬럼으로 생성
# title_cleaned는 NFC 정규화와 공백 정리만 — NFKC를 쓰면 ㎞->km 처럼 글자가 바뀌어 정보가 사라지므로 NFC 사용
df['title_cleaned'] = df['title'].astype(str).map(
    lambda s: re.sub(r'\s+', ' ', unicodedata.normalize('NFC', s)).strip())
df['body_cleaned'] = df['body'].map(clean_body)
df['text'] = (df['title_cleaned'] + ' ' + df['body_cleaned']).str.strip()   # LDA·워드클라우드용 합본 텍스트
df['body_length'] = df['body_cleaned'].str.len()                            # 정제 후 본문 글자 수
# 외국어(영문·일문) 우세 본문 식별용 한글비율 — 본문은 안 건드리고 수치만 산출(삭제 아님)
# body_cleaned는 NFC 정규화 전이라 분자·분모 둘 다 같은 NFC 문자열로 계산(분모 어긋나면 비율이 낮게 나와 오제거 위험)
_body_nfc = df['body_cleaned'].astype(str).map(lambda s: unicodedata.normalize('NFC', s))
df['hangul_chars'] = _body_nfc.map(lambda s: len(re.findall(r'[가-힣]', s)))
df['hangul_ratio'] = (df['hangul_chars'] / _body_nfc.str.len().clip(lower=1)).round(4)
# 외국어 전용 기사 플래그(팀 결정: 파이프라인에서 제외) — 한글 0자 기준이라 한글이 1자라도 있으면 절대 안 걸림
# 실측(260505_260511): 연합뉴스 영문52·일문30 = 82건 전부 한글 0자, 보존 기사 최저 한글비율 0.2444 — 비율 임계 대신 0자 기준이 over-removal 구조적 차단
df['is_foreign'] = df['hangul_chars'] == 0
print('정제 후 본문 길이 분포:')
print(df['body_length'].describe())


# --- 플래그 생성 — 기사를 제거하지 않고 표시만 해서 분석 단계에서 골라 쓰게 함 ---

# 방송 비기사 표시
df['is_weather'] = df['title'].astype(str).str.contains(r'\[날씨\]', regex=True, na=False)   # [날씨] 단신
df['is_closing'] = (
    (df['title_cleaned'] == '클로징')
    | df['body'].astype(str).map(lambda b: '뉴스 마칩니다' in b.rstrip()[-30:])   # 본문 끝의 마무리 멘트
)

# 중복 판정용으로 제목을 간단히 정규화 — 저장하지 않는 임시 컬럼
# 말머리([속보] 등)·따옴표·말줄임표·끝의 시각(- HH:MM)을 떼어 같은 기사끼리 잘 묶이게 함, 한자는 보존
def _title_dup_norm(s):
    s = re.sub(r'^\[(?:속보|1보|2보|3보|단독|종합|포토|사진|영상|모닝와이드|12뉴스|뉴스브리핑|정치쇼|잠시만요)\]\s*', '', s)
    s = re.sub(r'[“”"‘’\']', '', s)
    s = s.replace('…', ' ').replace('...', ' ')
    s = re.sub(r'\s*-\s*\d{1,2}:\d{2}\s*$', '', s)
    return re.sub(r'\s+', ' ', s).strip(' .,:;')

df['_tdn'] = df['title_cleaned'].map(_title_dup_norm)
df['_d10'] = df['pubdate'].dt.strftime('%Y-%m-%d')   # 날짜(YYYY-MM-DD)만
_nonempty = df['_tdn'].str.len() > 0
# 같은 매체에서 같은 날 같은 제목이 2건 이상이면 같은 기사 반복 송고로 추정
df['is_within_press_dup'] = df.duplicated(['press', '_d10', '_tdn'], keep=False) & _nonempty
# 같은 날 같은 제목이 2개 이상 매체에 걸치면 연합뉴스 등 재전송으로 추정
_cross_n = df.groupby(['_d10', '_tdn'])['press'].transform('nunique')
df['is_cross_press_dup'] = (_cross_n >= 2) & _nonempty

print('is_weather:', int(df['is_weather'].sum()), '/ is_closing:', int(df['is_closing'].sum()))
print('is_within_press_dup:', int(df['is_within_press_dup'].sum()),
      '/ is_cross_press_dup:', int(df['is_cross_press_dup'].sum()))
# 참고 — 날짜가 자정을 넘거나(어제 23시 vs 오늘 0시) '클로징'처럼 매일 같은 제목이면 잘못 묶일 수 있음
#        삭제가 아니라 표시일 뿐이므로 분석 단계에서 필요할 때만 골라 제외하면 됨
print('  (자정 경계·반복 제목은 잘못 묶일 수 있음 — 삭제 아닌 표시라 분석 때 선택 활용)')


# --- 검수·필터 — 실제로 행을 지우는 건 두 경우뿐, 중복·비기사는 위에서 표시만 했으니 안 지움 ---
report = {'원본': len(df)}

# 1) 분석 기간(2026-05-05~05-11) 밖이거나 날짜를 못 읽은(NaT) 기사 제외
start = pd.Timestamp(PERIOD_START)
end = pd.Timestamp(PERIOD_END) + pd.Timedelta(days=1)   # 끝 날짜 당일까지 포함
in_range = df['pubdate'].notna() & (df['pubdate'] >= start) & (df['pubdate'] < end)
report['날짜 제외'] = int((~in_range).sum())
df = df[in_range].copy()

# 2) 정제 후 본문이 너무 짧은(20자 미만) 기사 제외 — 사진만 있던 단신 등 실질 내용 없는 것
before = len(df)
df = df[df['body_length'] >= MIN_BODY_LEN_AFTER_CLEAN].copy()
report['짧은 본문 제외'] = before - len(df)

df = df.reset_index(drop=True)
report['최종'] = len(df)
print('필터 리포트:', report)
print('보존 표시(삭제 안 함):',
      {'is_weather': int(df['is_weather'].sum()), 'is_closing': int(df['is_closing'].sum()),
       'within_dup': int(df['is_within_press_dup'].sum()), 'cross_dup': int(df['is_cross_press_dup'].sum())})


# --- 컬럼 순서 정리와 저장 ---
# 매체그룹 -> 매체 -> 발행시각 순으로 정렬한 뒤 article_id(고유 번호)를 1부터 부여
df = df.sort_values(['media_group', 'press', 'pubdate']).reset_index(drop=True)
df['article_id'] = range(1, len(df) + 1)

col_order = [
    'article_id', 'link', 'pubdate', 'date',
    'category', 'article_category', 'article_category_full',
    'press', 'media_group', 'source_period', 'source_file',
    'title', 'title_cleaned', 'body', 'body_cleaned', 'text', 'body_length', 'hangul_chars', 'hangul_ratio',
    'is_weather', 'is_closing', 'is_within_press_dup', 'is_cross_press_dup', 'is_foreign',
]
df = df[col_order]   # 중복 판정용 임시 컬럼(_tdn/_d10)은 여기서 자동으로 빠짐

# 출력 파일명 기간(YYMMDD_YYMMDD)은 입력 통합본 파일명에서 그대로 가져옴
period = re.search(r'(\d{6}_\d{6})', normalize_name(INPUT_PATH)).group(1)
OUTPUT_PATH = DATA_DIR / f'전처리_본문_언론사_{period}.csv'
df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
print(f'저장 완료: {OUTPUT_PATH}')
print(f'최종 행수: {len(df)} / 컬럼: {list(df.columns)}')


# --- 데이터 현황 표 — 보고서 '데이터 수집 개요'에 그대로 쓸 수 있음 ---
print('=== 매체그룹별 기사 수 ===')
print(df['media_group'].value_counts())
print('\n=== 매체별 기사 수 ===')
print(df['press'].value_counts())
print('\n=== 날짜별 × 매체그룹별 기사 수 ===')
pivot = df.pivot_table(index='date', columns='media_group', values='article_id', aggfunc='count', fill_value=0)
pivot['합계'] = pivot.sum(axis=1)
print(pivot)
print('\n=== 표시(플래그) 합계 ===')
print({c: int(df[c].sum()) for c in ['is_weather', 'is_closing', 'is_within_press_dup', 'is_cross_press_dup', 'is_foreign']})
df.head(3)

