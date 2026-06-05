#!/usr/bin/env bash
# 최종 제출물 조립 — 1조.zip 골격(수집한 데이터/코드/발표PPT)에 맞춰 폴더+zip 생성.
# repo는 '소스'(코드·입력·리소스)만 두고, 데이터/차트/추론/PPT는 여기서 '조립'만 함(repo 비대화 방지).
# 사용: bash build_submission.sh [조이름]   예) bash build_submission.sh 텍데분_3조
# 전제: 먼저 pipeline_py/run_all.py 실행(전처리·토큰·LDA·차트 재생성) + dl/nli(학교 GPU) 완료 + PPT 준비.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
NAME="${1:-텍데분_제출}"
OUT="$ROOT/제출_build/$NAME"
PERIOD="260505_260511"
PPT_SRC="${PPT_PATH:-}"   # 환경변수 PPT_PATH로 발표 pdf 경로 지정 가능

rm -rf "$OUT"; mkdir -p "$OUT"
warn(){ echo "  ⚠ 없음(스킵): $1"; }
cp_if(){ if [ -e "$1" ]; then mkdir -p "$2"; cp -r "$1" "$2/"; echo "  + $(basename "$1")"; else warn "$1"; fi; }

echo "[1] 수집한 데이터/"
D="$OUT/수집한 데이터"
cp_if "$ROOT/data/news/통합_본문_bs4_언론사_${PERIOD}.csv"      "$D/1. 크롤링 결과"
cp_if "$ROOT/data/news/전처리_본문_언론사_${PERIOD}.csv"         "$D/2. 전처리·토큰화 결과"
cp_if "$ROOT/data/news/분석토큰_언론사_${PERIOD}.csv"           "$D/2. 전처리·토큰화 결과"
cp_if "$ROOT/data/news/analysis_언론사_lda결과_재수집.csv"       "$D/3. 분석 결과/LDA"
cp_if "$ROOT/outputs/감성_문서별_${PERIOD}.csv"                 "$D/3. 분석 결과/감성"
cp_if "$ROOT/pipeline_py/딥러닝_논조분석_산출물"                 "$D/3. 분석 결과/논조(DL)"
cp_if "$ROOT/pipeline_py/제로샷NLI_스탠스분석_산출물"            "$D/3. 분석 결과/스탠스(NLI)"
cp_if "$ROOT/pipeline_py/산출물_차트"                           "$D/3. 분석 결과/차트"
cp_if "$ROOT/resources/SentiWord_info.json"                    "$D/사전"

echo "[2] 코드/"
C="$OUT/코드"
cp_if "$ROOT/notebooks/crawling/press"     "$C/노트북"
cp_if "$ROOT/notebooks/crawling/analysis"  "$C/노트북"
mkdir -p "$C/pipeline_py"
cp "$ROOT/pipeline_py/"*.py "$ROOT/pipeline_py/requirements.txt" "$ROOT/pipeline_py/README.md" "$C/pipeline_py/" 2>/dev/null && echo "  + pipeline_py/*.py"
cp_if "$ROOT/PIPELINE_PLAN.md" "$C"

echo "[3] 발표 PPT + README"
[ -n "$PPT_SRC" ] && cp_if "$PPT_SRC" "$OUT" || warn "발표 PDF (PPT_PATH 미지정 — 수동으로 $OUT 에 넣기)"
cp "$ROOT/제출_README.txt" "$OUT/README.txt" 2>/dev/null && echo "  + README.txt"

echo "[4] zip 생성"
( cd "$ROOT/제출_build" && zip -rq "$NAME.zip" "$NAME" ) && echo "완료 → 제출_build/$NAME.zip ($(du -sh "$OUT" | cut -f1))"
echo
echo "체크: DL/NLI·PPT가 '없음'이면 학교 작업 후 다시 실행하거나 수동 추가."
