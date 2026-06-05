# -*- coding: utf-8 -*-
"""
중앙 경로 설정 — 절대 디렉토리만, repo-상대로 도출(이식성). 계산·로직 없음(PIPELINE_PLAN §4).
PROJECT_DIR을 이 파일 위치에서 도출하므로 scp/이동해도 그대로 동작.
period 리터럴·알고리즘은 각 모듈 verbatim, 여기선 디렉토리 앵커만 제공.
고정 분석기간 260505_260511 재현이 목표.
"""
from pathlib import Path
import unicodedata

# pipeline_py/ 의 부모 = repo 루트 (이동·scp 후에도 자동 도출)
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR    = PROJECT_DIR / 'data' / 'news'          # 통합/전처리/분석토큰/lda결과 공용 입출력
LDA_OUT     = DATA_DIR                                # 팀 LDA는 DATA_DIR에 결과 csv 저장
CHART_OUT   = PROJECT_DIR / 'pipeline_py' / '산출물_차트'   # fulltext F* / absa A* png

# 번들 리소스(repo/resources/) — scp로 함께 이동
KNU_PATH    = PROJECT_DIR / 'resources' / 'SentiWord_info.json'   # KNU 감성사전
FONT_PATH   = PROJECT_DIR / 'resources' / 'NanumGothic.ttf'       # 한글 폰트(차트·워드클라우드, PNG 전용)

# 호르무즈 추론 입출력 (불변식: nli.DL_OUT == dl_tone.OUT_DIR)
DL_DIR  = PROJECT_DIR / 'pipeline_py' / '딥러닝_논조분석_산출물'
NLI_DIR = PROJECT_DIR / 'pipeline_py' / '제로샷NLI_스탠스분석_산출물'

PERIOD = '260505_260511'   # 고정 분석기간(파일명 리터럴은 각 모듈 verbatim, 여기선 참고)

def nfc_find(directory: Path, exact_name: str) -> Path:
    """exact_name(NFC)과 일치하는 파일을 directory에서 찾아 실제 경로 반환(Drive NFD 대비)."""
    target = unicodedata.normalize('NFC', exact_name)
    for p in directory.iterdir():
        if p.is_file() and unicodedata.normalize('NFC', p.name) == target:
            return p
    raise FileNotFoundError(f'{exact_name} not found in {directory}')

for _d in (CHART_OUT, DL_DIR, NLI_DIR):
    _d.mkdir(parents=True, exist_ok=True)
