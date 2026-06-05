# Repo cleanup + transfer-to-school plan (review for delete-safety & completeness)

Repo: /home/carol/Text-data-Analysis_26-Spring (git remote: github.com/ChaeYeonKim0204/Text-data-Analysis_26-Spring). data/news & outputs are gitignored. Goal: clean repo (free ~600MB), transfer to a SCHOOL machine (GPU, fresh env) so the new `pipeline_py/` runs there.

## KEEP (required to run pipeline_py on school)
- `pipeline_py/` (8 modules + config.py + run_all.py + requirements.txt + README.md) — the code.
- `resources/SentiWord_info.json` (KNU dict), `resources/NanumGothic.ttf` (font) — bundled inputs.
- `data/news/통합_본문_bs4_언론사_260505_260511.csv` (29MB) — the pipeline's ONLY required external input (preprocess reads it; everything else is regenerated).
- `PIPELINE_PLAN.md`, `README.md`, `CLAUDE.md` — docs.
- `notebooks/` (1.7MB) — source notebooks the .py were derived from (reference).

## DELETE (local; per user decision)
- `data/crawling/` (535MB — old telecom SKT/KT/LG phase, unrelated). NOTE: some small JSONs here ARE git-tracked → deletion shows in `git status`; recoverable via remote.
- `data/news/` intermediates (gitignored → local-only delete): `전처리_본문_언론사_260505_260511.csv`(87M)+`.bak_before_hangul`, `분석토큰_언론사_260505_260511.csv`(15M), `analysis_언론사_lda결과_재수집.csv`(15M), `본문_bs4_*`(per-press), `본문_통합_260505_260511.csv`, link/체크포인트 JSONs. **KEEP `통합_본문_bs4_언론사_260505_260511.csv`.**
- `outputs/` (37MB, gitignored — my separate 01–05 analysis track).

## TRANSFER to school
1. **Code/resources/docs via git**: `git add pipeline_py resources PIPELINE_PLAN.md` → commit → push to GitHub. On school: `git clone` recreates them (+ history). (resources ~4.3MB committed.)
2. **Input data via scp** (통합본문 is gitignored → can't go via git): `scp` `data/news/통합_본문_bs4_언론사_260505_260511.csv` → school repo `data/news/`.
3. **dl/nli verification reference** (optional): scp the team's shipped `DL_문장별_논조분석.csv` / `NLI_문장별_스탠스분석.csv` (from team zip) for value-compare on school.
4. School setup: python3.11 venv → `pip install -r pipeline_py/requirements.txt` + torch/transformers (GPU) + fetch 2 HF models → `python pipeline_py/run_all.py`.

## Review questions
1. **Delete-safety**: does any pipeline_py module READ a file in the DELETE list (other than ones it regenerates)? Confirm 통합본문 + resources are the only external inputs, so deleting intermediates/crawling/outputs cannot break a from-scratch run.
2. **Completeness**: after `git clone` + scp 통합본문, does school have everything to run run_all.py end-to-end (preprocess→…→absa; dl/nli pending models)? Any needed file neither in git nor scp'd?
3. **resources via git**: are KNU `SentiWord_info.json` + `NanumGothic.ttf` OK to commit (license/size)? config.py references them repo-relative (PROJECT_DIR/resources) — does that resolve on school after clone?
4. Any risk in deleting `data/crawling/` git-tracked JSONs (recoverable via remote)? Any non-regenerable artifact about to be lost (not on remote/Drive)?
