# -*- coding: utf-8 -*-
"""
파이프라인 오케스트레이터 — 각 모듈을 '외부 서브프로세스'로 순차 실행(os.chdir/전역 격리).
실행 전 각 모듈의 '출력 파일/폴더만' 제거(clean-before-run) — 공유 DATA_DIR 통째 삭제 금지.
순서: preprocess → tokenize_kiwi → {lda_topics, fulltext_analysis, absa_sentiment} → dl_tone → nli_stance
PIPELINE_PLAN.md §1/§6 준수. dl_tone/nli_stance는 transformers+torch+HF모델 필요(없으면 그 단계만 실패).
"""
import subprocess, sys, shutil
from pathlib import Path
import config as cfg

HERE = Path(__file__).resolve().parent
PY = sys.executable

# 모듈별 (스크립트, 실행 전 제거할 출력) — 공유 DATA_DIR은 '파일명 단위'만, 전용 폴더는 통째
PERIOD = cfg.PERIOD
STAGES = [
    ("preprocess.py",        [cfg.DATA_DIR / f"전처리_본문_언론사_{PERIOD}.csv"]),
    ("tokenize_kiwi.py",     [cfg.DATA_DIR / f"분석토큰_언론사_{PERIOD}.csv"]),
    ("lda_topics.py",        [cfg.DATA_DIR / "analysis_언론사_lda결과_재수집.csv"]),
    ("fulltext_analysis.py", [cfg.CHART_OUT]),   # 전용 차트 폴더(통째)
    ("absa_sentiment.py",    [cfg.CHART_OUT]),    # 같은 차트 폴더에 A01~03 추가(통째 삭제는 fulltext 전에만; 아래 처리)
    ("dl_tone.py",           [cfg.DL_DIR]),        # 전용 산출 폴더(통째) — NLI glob 단일성 보장
    ("nli_stance.py",        [cfg.NLI_DIR]),       # 전용 산출 폴더(통째)
]

def clean(targets):
    for t in targets:
        t = Path(t)
        if t.is_dir():
            shutil.rmtree(t, ignore_errors=True); t.mkdir(parents=True, exist_ok=True)
        elif t.exists():
            t.unlink()

def main():
    # CHART_OUT은 fulltext/absa가 공유 → fulltext 전에 한 번만 통째 비우고, absa는 추가만(삭제 안 함)
    chart_cleaned = False
    for script, outs in STAGES:
        if Path(cfg.CHART_OUT) in [Path(o) for o in outs]:
            if not chart_cleaned:
                clean([cfg.CHART_OUT]); chart_cleaned = True
        else:
            clean(outs)
        print(f"\n===== RUN {script} =====", flush=True)
        r = subprocess.run([PY, str(HERE / script)], cwd=str(HERE))
        if r.returncode != 0:
            print(f"!! {script} 실패(returncode {r.returncode}) — 중단", flush=True)
            sys.exit(r.returncode)
    print("\n파이프라인 완료.")

if __name__ == "__main__":
    main()
