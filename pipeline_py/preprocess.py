# -*- coding: utf-8 -*-
"""
① 전처리 — 통합 본문을 정제하고 분석용 표시 컬럼을 붙임
읽는 파일: data/news/통합_본문_bs4_언론사_260505_260511.csv (기사 11,990건)
저장 파일: data/news/전처리_본문_언론사_260505_260511.csv (정제본+text 합본, is_weather/is_closing/중복/is_foreign 표시 컬럼)
"""
# Colab에서 실행할 때만 아래 3줄 주석 해제 — 로컬/WSL에서는 그대로 두기
# from google.colab import drive
# drive.mount('/content/drive')

from pathlib import Path
import os
import re
import unicodedata

import pandas as pd

# 같은 폴더의 설정 파일 불러오기
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent)); import config as _cfg
PROJECT_DIR = _cfg.PROJECT_DIR
DATA_DIR = _cfg.DATA_DIR

os.chdir(PROJECT_DIR)   # 프로젝트 폴더를 기준으로 파일을 읽고 저장
print(f'현재 작업 폴더: {Path.cwd()}')

# 전처리 기준값 — 필요하면 여기만 수정
MEDIA_GROUP = '지상파'              # 통합본에 media_group이 없을 때의 기본값
# 통합본에 그룹 정보가 없을 때 언론사 이름을 보고 채우는 표
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

# 입력 통합본 찾기
# 한글 파일명이 환경에 따라 다르게 저장될 수 있어 이름을 한 번 정리해서 비교
def normalize_name(path):
    return unicodedata.normalize('NFC', path.name)

# 이번 분석 기간에 해당하는 통합본만 사용
_PINNED = '통합_본문_bs4_언론사_260505_260511.csv'
INPUT_PATH = next(p for p in DATA_DIR.iterdir()
                  if p.is_file() and normalize_name(p) == _PINNED)
print(f'입력 통합본: {INPUT_PATH.name}')

df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')   # 한글 CSV 파일 불러오기
print(f'원본 행수: {len(df)}')
print('press별:', df['press'].value_counts().to_dict())

# 출처와 날짜 관련 컬럼 정리

# 그룹 정보는 통합본에 있으면 그대로 쓰고, 없을 때만 위 표나 기본값으로 채움
if 'media_group' not in df.columns:
    df['media_group'] = df['press'].map(MEDIA_GROUP_MAP).fillna(MEDIA_GROUP)
# source_file은 통합본에 있으면 그대로, 없으면 press와 기간으로 파일 이름을 다시 만들어 둠
if 'source_file' not in df.columns:
    df['source_file'] = '본문_bs4_' + df['press'].astype(str) + '_' + df['source_period'].astype(str) + '.csv'

# category에서 상위분류만 남긴 article_category 생성 (예: '스포츠/kbaseball' -> '스포츠', '생활/문화' -> '생활')
# 원래 값은 article_category_full에 남기고, 빈 값은 '미분류'로 채움
df['article_category_full'] = df['category']
df['article_category'] = (
    df['category'].astype(str).str.replace(r'/.*', '', regex=True)
      .where(df['category'].notna(), '미분류')
)

# pubdate를 날짜로 바꾸고, 날짜 부분만 date 컬럼에 따로 저장
df['pubdate'] = pd.to_datetime(df['pubdate'], errors='coerce')   # 날짜로 바꾸지 못한 값은 아래 필터에서 제외
df['date'] = df['pubdate'].dt.date
print('article_category 분포:')
print(df['article_category'].value_counts())


# 기사 본문이 아닌 안내문, 기자·사진 출처, 제보·저작권 문구 등을 골라 제거
# 핵심 원칙 — 사진/기자/이메일/제공 같은 단어가 들어있다고 무조건 지우지 않음
# 기사 맨 앞·맨 끝이거나, 정해진 형태(매체명이 든 출처 등)일 때만 제거
# 발언 인용, 본문 속 이메일, 한자 표기처럼 기사 내용에 해당하는 부분은 남김
# 애매하면 남기고 분석 단계 불용어로 거름

def _strip_dateline(t):
    # 연합뉴스·뉴스1·뉴시스 기사 머리 "(서울=연합뉴스) 김윤구 기자 =" 형태 제거
    # 매체 이름과 '기자 ='가 같이 있을 때만 지우므로 본문 속 일반 (A=B) 괄호는 안 건드림
    # 사진 캡션 문장 뒤에 붙은 경우도 있어 위치 상관없이 제거
    t = re.sub(r'\(\s*[^()\n]{1,20}=\s*(?:연합뉴스|연합뉴스TV|뉴스1|뉴시스)\)\s*(?:[^\n]{1,30}?\s+)?기자\s*=\s*', ' ', t)
    # 대괄호형 머리 [헤럴드경제=나은정 기자]는 기사 맨 앞일 때만 제거
    t = re.sub(r'^\s*\[[^\]\n]{1,20}=(?:[^\]\n]{1,20}\s+)?[가-힣]{2,5}\s*기자\]\s*', ' ', t)
    return t

def clean_body(text):
    t = str(text)
    # 1. SBS 스포츠 기사 맨 앞 동영상 안내문 제거 ("※ 저작권 관계로 ... [원문에서 영상 보기] https://...")
    # 안내문이 여러 줄로 들어와도 한 번에 제거
    t = re.sub(r'^\s*※\s*저작권 관계로 네이버에서 서비스하지 않는 영상입니다\..*?\[원문에서 영상 보기\]\s*https?://\S+\s*', ' ', t, flags=re.S)
    # 2. 방송 코너·지국 이름표 제거 — 기사 맨 앞 [KBS 강릉], [뉴스25], [정치쇼] 등 정해둔 이름만
    t = re.sub(r'^\s*(?:\[KBS\s+[가-힣]{1,12}\]\s*)+', ' ', t)
    t = re.sub(r'^\s*(?:\[(?:뉴스25|뉴스투데이|정오뉴스|5시뉴스|930MBC뉴스|12MBC뉴스|뉴스데스크|뉴스외전)\]\s*)+', ' ', t)
    t = re.sub(r'^\s*(?:\[(?:모닝와이드|12뉴스|뉴스브리핑|주영진\s+뉴스브리핑|정치쇼)\]\s*)+', ' ', t)
    # 3. YTN 라디오 머리말 제거 ([잠시만요], 방송일시·진행·출연자 정보) — 실제 대화(◆◇◎▶▷로 시작)는 남김
    t = re.sub(r'^\s*(?:\[잠시만요\]|YTN라디오\(FM\s*94\.5\)|\[YTN[^\]]+\]).*?(?=[◆◇◎▶▷□]\s)', ' ', t, flags=re.S)
    t = re.sub(r'□\s*(?:방송일시|진행|출연자)[^◆◇◎▶▷]*', ' ', t)
    # 3-1. YTN 라디오 방송시각 안내문 제거 — '□ 방송시각 HH:MM' 형태도 지움
    t = re.sub(r'□\s*방송(?:시각|시간)\s*[:：]\s*\d{1,2}[:：]\d{2}', ' ', t)
    # 4. 방송 대본 화자 표시 제거 (<앵커>·◀앵커▶·[앵커/기자/리포트] 등) — [트럼프/미국 대통령 : ...] 형태처럼 이름·발언이 든 건 남김
    t = re.sub(r'(?:<\s*(?:앵커|기자|리포트)\s*>|◀\s*(?:앵커|기자|리포트)\s*▶|\[\s*(?:앵커|기자|리포트|답변|녹취|인터뷰)\s*\])', ' ', t)
    t = re.sub(r'\((?:[A-Za-z가-힣]+\s+)?(?:디지털뉴스부|뉴미디어부)\)', ' ', t)   # (SBS 디지털뉴스부) 같은 부서 표기
    # 5. 연합뉴스·뉴스1·뉴시스 기사 머리(지역·매체·기자 표시) 제거 — 위 _strip_dateline 사용
    t = _strip_dateline(t)
    # 6. 사진·자료 출처 표기 제거 — "사진" 단어만 보고 지우지는 않음
    t = re.sub(r'\[[^\]\n]{0,20}?(?:사진\s*(?:출처|제공|촬영|=|:|[|ｌlㅣ│])|자료사진|제보사진|캡처|일러스트|그래픽\s*[:=]|(?:로이터|AFP|AP|EPA)=연합뉴스|=\s*(?:연합뉴스|뉴스1|뉴시스)|재판매\s*및\s*DB\s*금지)[^\]\n]{0,80}\]', ' ', t)
    t = re.sub(r'\[[^\]\n]{1,25}\s(?:제공|출처)\]', ' ', t)   # [HMM 제공]처럼 '제공/출처'가 ] 바로 앞에 오는 짧은 출처표기
    t = re.sub(r'\((?:자료)?사진\s*(?:출처|제공|촬영)?\s*[:=][^)]{1,80}\)', ' ', t)   # (사진출처=연합뉴스) 괄호형 출처 표기
    t = re.sub(r'(?:(?<=\s)|^)(?:자료)?사진\s*[=:|ｌlㅣ│]\s*(?:연합뉴스|뉴스1|뉴시스|게티이미지뱅크|한경\s*DB|EPA|로이터|AFP|AP|EBS|[가-힣A-Za-z]{1,12}\s*(?:제공|DB|SNS|캡처|화면|유튜브))\.?', ' ', t)   # 사진=연합뉴스 형태의 출처 표기
    t = re.sub(r'\*기사와 관련 없음', ' ', t)   # 게티이미지·스톡 사진 삽입 시 붙는 안내 문구
    t = re.sub(r'(?:사진|이미지|자료\s*사진)\s*\[[^\]\n]{1,30}\]', ' ', t)   # 인라인 사진/이미지 출처 [국립천문대] 등
    # 6에서 맨 앞 사진 출처를 지우면 가려져 있던 기사 머리가 드러날 수 있어 지역·매체·기자 표시 제거를 한 번 더 적용
    t = _strip_dateline(t)
    # 7. 기사 맨 끝 기자 이름·이메일 제거 ("이미아 기자 mia@hankyung.com") — 본문 중간 이메일(info@PGSA.ir)은 그대로
    # 기자 이름이 여러 명 붙어 있는 경우까지 제거
    t = re.sub(r'(?:[가-힣]{2,10}\s*(?:선임\s*|인턴\s*)?(?:기자|특파원|논설위원)(?:\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})?\s*[/·]?\s*){1,6}$', ' ', t)
    t = re.sub(r'(?:[A-Za-z0-9._%+-]+@(?:yna\.co\.kr|hankyung\.com|mk\.co\.kr|hani\.co\.kr|chosun\.com))(?:\s*[A-Za-z0-9._%+-]+@(?:yna\.co\.kr|hankyung\.com|mk\.co\.kr|hani\.co\.kr|chosun\.com))*\s*$', ' ', t)
    t = re.sub(r'\[[가-힣 ]{2,20}(?:스타투데이\s+)?기자\]\s*$', ' ', t)
    # 8. 방송사·신문사별 맨 끝 제보·저작권 안내 블록 제거 — 각 매체가 늘 붙이는 문구
    t = re.sub(r'\s*■\s*제보하기.*$', ' ', t, flags=re.S)                                           # KBS
    t = re.sub(r'\s*MBC 뉴스는 24시간 여러분의 제보를 기다립니다\..*$', ' ', t, flags=re.S)          # MBC
    t = re.sub(r'\s*※\s*\W?당신의 제보가 뉴스가 됩니다\W?\[카카오톡\].*?social@ytn\.co\.kr\s*$', ' ', t, flags=re.S)  # YTN
    t = re.sub(r'\s*※\s*자세한 내용은 동영상으로 확인하실 수 있습니다\.?\s*$', ' ', t)                # SBS
    t = re.sub(r'\s*인터뷰 자료의 저작권은 SBS 라디오에 있습니다\..*$', ' ', t, flags=re.S)           # SBS 라디오
    t = re.sub(r'\s*이 기사는 한국경제신문과 금융 AI 전문기업 씽크풀이 공동 개발한 기사 자동생성 알고리즘에 의해 실시간으로 작성된 것입니다\.\s*$', ' ', t)  # 한국경제 자동생성 안내
    t = re.sub(r'\s*당신의 제보가 뉴스로 만들어집니다\..*$', ' ', t, flags=re.S)                      # SBS Biz
    # 9. 저작권 표시 블록 제거 (<저작권자 ⓒ ...>, [Copyright (c) ...], [본 기사는 ...])
    t = re.sub(r'<저작권자\s*ⓒ[^>]+>|\[Copyright\s*\(c\)[^\]]+\]|\[본 기사는 [^\]]+\]', ' ', t)
    # 9-1. 본문 중간에 녹아든 기자 이름과 이메일 제거 — 5개 매체 주소만(연합/한경/매경/한겨레/조선), info@PGSA.ir·gmail 등은 남김
    t = re.sub(r'(?:/\s*)?(?:사진\s*=\s*)?[가-힣]{2,5}\s*(?:한경닷컴\s*|닷컴\s*)?기자\s*[A-Za-z0-9._%+-]+@(?:yna\.co\.kr|hankyung\.com|mk\.co\.kr|hani\.co\.kr|chosun\.com)', ' ', t)
    t = re.sub(r'[A-Za-z0-9._%+-]+@(?:yna\.co\.kr|hankyung\.com|mk\.co\.kr|hani\.co\.kr|chosun\.com)', ' ', t)
    # 10. 기사 맨 끝 제작진 표기 제거 ("영상편집:홍길동 그래픽:김철수") — 끝에 있을 때만 지우므로 본문 중간 제목·인용(관계의 기술:, 재구성:)은 남김
    # 제작진 이름이 여러 명 붙어 있는 경우까지 제거
    t = re.sub(r'(?:\s*(?<![가-힣])(?:출연|진행|구성|영상취재|영상편집|영상|디자인|촬영|촬영기자|연출|제작|제작진|현장진행|기술감독|자막|그래픽|CG|책임\s*프로듀서|작가)\s*[:：]\s*[가-힣A-Za-z·,]{1,30}){1,8}\s*$', ' ', t)
    t = re.sub(r'\s*\((?:구성|영상편집|연출|제작|촬영기자)\s*:\s*[^()]{1,80}\)\s*$', ' ', t)
    # 11. 기사 맨 앞의 구분선 제거
    t = re.sub(r'^\s*---\s*', ' ', t)
    # 11-1. 본문 전체에서 장식·목록 기호 제거 (◆◇▲■◎ 등 — 순수 기호라 본문 손실 없음). ·ㅣ│ⓒ%·…는 본문 정보라 남김
    t = re.sub(r'[■□◇◆◎▲▽△▷◁▶◀☞☜※★☆◈➤●]+', ' ', t)
    # 12. 줄바꿈·연속 공백을 한 칸으로 정리 (이 단계에서만 공백 정리)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# 원본 title/body는 그대로 두고 정제본(title_cleaned/body_cleaned)과 합본(text)을 새 컬럼으로 생성
# 제목은 글자를 바꾸지 않고 공백만 정리
df['title_cleaned'] = df['title'].astype(str).map(
    lambda s: re.sub(r'\s+', ' ', unicodedata.normalize('NFC', s)).strip())
df['body_cleaned'] = df['body'].map(clean_body)
df['text'] = (df['title_cleaned'] + ' ' + df['body_cleaned']).str.strip()   # LDA·워드클라우드용 합본 텍스트
df['body_length'] = df['body_cleaned'].str.len()                            # 정제 후 본문 글자 수
# 영어·일본어가 대부분인 기사를 찾기 위해 한글 비율 계산 — 본문은 지우지 않고 수치만 저장
# 정리된 본문에서 한글 글자 수와 한글 비율 계산
_body_nfc = df['body_cleaned'].astype(str).map(lambda s: unicodedata.normalize('NFC', s))
df['hangul_chars'] = _body_nfc.map(lambda s: len(re.findall(r'[가-힣]', s)))
df['hangul_ratio'] = (df['hangul_chars'] / _body_nfc.str.len().clip(lower=1)).round(4)
# 한글이 하나도 없는 외국어 기사는 분석에서 제외하기 위해 표시
df['is_foreign'] = df['hangul_chars'] == 0
print('정제 후 본문 길이 분포:')
print(df['body_length'].describe())


# 표시 컬럼 생성 — 기사를 제거하지 않고 표시만 해서 분석 단계에서 골라 씀

# 방송 비기사 표시
df['is_weather'] = df['title'].astype(str).str.contains(r'\[날씨\]', regex=True, na=False)   # [날씨] 단신
df['is_closing'] = (
    (df['title_cleaned'] == '클로징')
    | df['body'].astype(str).map(lambda b: '뉴스 마칩니다' in b.rstrip()[-30:])   # 기사 끝의 마무리 멘트 확인
)

# 중복 판정용으로 제목을 비교하기 쉽게 간단히 정리 — 저장하지 않고 잠깐 쓰는 컬럼
# 말머리([속보] 등)·따옴표·말줄임표·끝의 시각(- HH:MM)을 떼어 같은 기사끼리 잘 묶이게 함
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
# 같은 날 같은 제목이 2개 이상 매체에 걸치면 다른 매체가 같은 기사를 받은 경우일 수 있음
_cross_n = df.groupby(['_d10', '_tdn'])['press'].transform('nunique')
df['is_cross_press_dup'] = (_cross_n >= 2) & _nonempty

print('is_weather:', int(df['is_weather'].sum()), '/ is_closing:', int(df['is_closing'].sum()))
print('is_within_press_dup:', int(df['is_within_press_dup'].sum()),
      '/ is_cross_press_dup:', int(df['is_cross_press_dup'].sum()))
# 날짜가 자정을 넘거나 '클로징'처럼 매일 같은 제목이면 잘못 묶일 수 있음 — 삭제하지 않고 표시만 하므로 분석 때 골라 사용
print('  (자정 경계·반복 제목은 잘못 묶일 수 있음 — 삭제하지 않고 표시만 하므로 분석 때 골라 사용)')


# 검수·필터 — 실제로 행을 지우는 건 두 경우뿐, 중복·비기사는 위에서 표시만 했으니 안 지움
report = {'원본': len(df)}

# 1) 분석 기간(2026-05-05~05-11) 밖이거나 날짜를 못 읽은 기사 제외
start = pd.Timestamp(PERIOD_START)
end = pd.Timestamp(PERIOD_END) + pd.Timedelta(days=1)   # 끝 날짜 하루 뒤보다 작은 기사까지 포함
in_range = df['pubdate'].notna() & (df['pubdate'] >= start) & (df['pubdate'] < end)
report['날짜 제외'] = int((~in_range).sum())
df = df[in_range].copy()

# 2) 정제 후 본문이 너무 짧은(20자 미만) 기사 제외 — 사진만 있던 단신 등 실질 내용 없는 것
before = len(df)
df = df[df['body_length'] >= MIN_BODY_LEN_AFTER_CLEAN].copy()
report['짧은 본문 제외'] = before - len(df)

df = df.reset_index(drop=True)
report['최종'] = len(df)
print('제외 결과:', report)
print('남김 표시(삭제 안 함):',
      {'is_weather': int(df['is_weather'].sum()), 'is_closing': int(df['is_closing'].sum()),
       'within_dup': int(df['is_within_press_dup'].sum()), 'cross_dup': int(df['is_cross_press_dup'].sum())})


# 컬럼 순서 정리와 저장
# 매체그룹 -> 매체 -> 발행시각 순으로 정렬하고 기사 번호를 1부터 붙임
df = df.sort_values(['media_group', 'press', 'pubdate']).reset_index(drop=True)
df['article_id'] = range(1, len(df) + 1)

col_order = [
    'article_id', 'link', 'pubdate', 'date',
    'category', 'article_category', 'article_category_full',
    'press', 'media_group', 'source_period', 'source_file',
    'title', 'title_cleaned', 'body', 'body_cleaned', 'text', 'body_length', 'hangul_chars', 'hangul_ratio',
    'is_weather', 'is_closing', 'is_within_press_dup', 'is_cross_press_dup', 'is_foreign',
]
df = df[col_order]

# 출력 파일명 기간(YYMMDD_YYMMDD)은 입력 통합본 파일명에서 그대로 가져옴
period = re.search(r'(\d{6}_\d{6})', normalize_name(INPUT_PATH)).group(1)
OUTPUT_PATH = DATA_DIR / f'전처리_본문_언론사_{period}.csv'
df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
print(f'저장 완료: {OUTPUT_PATH}')
print(f'최종 행수: {len(df)}')


# 데이터 현황 표
print('=== 매체그룹별 기사 수 ===')
print(df['media_group'].value_counts())
print('\n=== 매체별 기사 수 ===')
print(df['press'].value_counts())
print('\n=== 날짜별 × 매체그룹별 기사 수 ===')
pivot = df.pivot_table(index='date', columns='media_group', values='article_id', aggfunc='count', fill_value=0)
pivot['합계'] = pivot.sum(axis=1)
print(pivot)
print('\n=== 표시 컬럼 합계 ===')
print({c: int(df[c].sum()) for c in ['is_weather', 'is_closing', 'is_within_press_dup', 'is_cross_press_dup', 'is_foreign']})
