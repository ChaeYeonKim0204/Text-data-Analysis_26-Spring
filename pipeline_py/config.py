# -*- coding: utf-8 -*-
"""
공통 경로 설정
각 분석 파일에서 같은 입력·출력 폴더를 쓰기 위해 한 곳에 모아둠
"""
from pathlib import Path
import unicodedata

# pipeline_py의 부모 폴더를 프로젝트 폴더로 사용
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR    = PROJECT_DIR / 'data' / 'news'          # 통합 데이터, 전처리 결과, 분석 토큰, LDA 결과 저장 폴더
LDA_OUT     = DATA_DIR                                # 팀 LDA 결과는 DATA_DIR 저장
CHART_OUT   = PROJECT_DIR / 'pipeline_py' / '산출물_차트'   # fulltext F* 차트와 absa A* 차트 png 저장 폴더

# 분석에 필요한 사전과 한글 폰트
KNU_PATH    = PROJECT_DIR / 'resources' / 'SentiWord_info.json'   # KNU 감성사전
FONT_PATH   = PROJECT_DIR / 'resources' / 'NanumGothic.ttf'       # 한글 폰트(차트·워드클라우드, PNG 전용)

# 호르무즈 의제 분석 결과 폴더
DL_DIR  = PROJECT_DIR / 'pipeline_py' / '딥러닝_논조분석_산출물'
NLI_DIR = PROJECT_DIR / 'pipeline_py' / '제로샷NLI_스탠스분석_산출물'

PERIOD = '260505_260511'

def nfc_find(directory: Path, exact_name: str) -> Path:
    """맥과 윈도우에서 한글 파일명이 다르게 잡힐 때 같은 파일을 찾음"""
    target = unicodedata.normalize('NFC', exact_name)
    for p in directory.iterdir():
        if p.is_file() and unicodedata.normalize('NFC', p.name) == target:
            return p
    raise FileNotFoundError(f'{exact_name} not found in {directory}')

for _d in (CHART_OUT, DL_DIR, NLI_DIR):
    _d.mkdir(parents=True, exist_ok=True)
