# -*- coding: utf-8 -*-
"""
분석 파이프라인 전체 실행
앞 단계에서 만든 파일을 지우고 처음부터 순서대로 다시 실행
순서: 전처리(preprocess) → 토큰화(tokenize_kiwi) → LDA/전체분석/감성분석 → 딥러닝 논조 → NLI 스탠스 분석
"""
import subprocess, sys, shutil
from pathlib import Path
import config as cfg

HERE = Path(__file__).resolve().parent
PY = sys.executable

# 다시 실행하기 전에 지울 이전 결과 파일과 실행할 파일 목록
PERIOD = cfg.PERIOD
STAGES = [
    ("preprocess.py",        [cfg.DATA_DIR / f"전처리_본문_언론사_{PERIOD}.csv"]),
    ("tokenize_kiwi.py",     [cfg.DATA_DIR / f"분석토큰_언론사_{PERIOD}.csv"]),
    ("lda_topics.py",        [cfg.DATA_DIR / "analysis_언론사_lda결과_재수집.csv"]),
    ("fulltext_analysis.py", [cfg.CHART_OUT]),   # 전체 분석 차트
    ("absa_sentiment.py",    [cfg.CHART_OUT]),   # 같은 차트 폴더에 ABSA 차트 추가
    ("dl_tone.py",           [cfg.DL_DIR]),      # 딥러닝 논조 분석 결과
    ("nli_stance.py",        [cfg.NLI_DIR]),     # NLI 스탠스 분석 결과
]

def clean(targets):
    for t in targets:
        t = Path(t)
        if t.is_dir():
            shutil.rmtree(t, ignore_errors=True); t.mkdir(parents=True, exist_ok=True)
        elif t.exists():
            t.unlink()

def main():
    # fulltext와 absa는 같은 차트 폴더를 사용하므로 처음 한 번만 비운다
    chart_cleaned = False
    for script, outs in STAGES:
        if Path(cfg.CHART_OUT) in [Path(o) for o in outs]:
            if not chart_cleaned:
                clean([cfg.CHART_OUT]); chart_cleaned = True
        else:
            clean(outs)
        print(f"\n===== {script} 실행 =====", flush=True)
        r = subprocess.run([PY, str(HERE / script)], cwd=str(HERE))
        if r.returncode != 0:
            print(f"!! {script} 실행 실패(종료 코드 {r.returncode}) - 여기서 멈춤", flush=True)
            sys.exit(r.returncode)
    print("\n파이프라인 완료.")

if __name__ == "__main__":
    main()
