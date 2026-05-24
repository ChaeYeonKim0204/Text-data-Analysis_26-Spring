# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Jupyter-notebook pipeline that crawls Naver News articles by keyword and month, then aggregates the results for text analysis. Keywords target Korean telecom carriers (SKT/SK텔레콤, KT, LG U+/LG유플러스) — each carrier is searched under both its abbreviation and full name as two separate queries.

There is no application code, no build, no test suite. All work lives in `notebook/crawling/*.ipynb` and writes to `notebook/crawling/data/`. The `result/` directory is currently empty (reserved for analysis output).

## Pipeline architecture

The notebooks form a linear, file-based pipeline. Each stage reads the previous stage's output from `notebook/crawling/data/` and writes new files into the same directory. There is no orchestrator — the user runs notebooks in order.

```
url_수집(_colab)        →  링크_{query}_{YYMMDD}_{YYMMDD}.json        (Selenium, Naver search)
본문_수집_bs4_colab     →  본문_bs4_{query}_{period}.csv              (requests + BS4, preferred)
본문_수집(_colab)        →  본문_{query}_{period}.csv                  (Selenium, fallback)
본문_재시도_selenium    →  merges retries back into 본문_bs4_*.csv     (rescues BS4 failures)
본문_통합_colab          →  통합_본문_bs4_{query}.csv                  (per-query merge)
url_통계 / 본문_통계     →  통계_*.csv                                  (diagnostics, no pipeline output)
```

CSV schema for body files: `link, pubdate, category, title, body` (+ `source_query`, `source_period`, `source_start_ym` in the merged files).

### Conventions every notebook shares

- **`query_ranges` block at the top** — list of `{query, start_ym, end_ym}` dicts. `build_monthly_jobs()` expands each into one job per calendar month (end-of-month is computed via `calendar.monthrange`, not hardcoded). To run a different keyword/range, edit this list and re-run; do not parametrize externally.
- **Filename = identity.** Stage, query, and `YYMMDD_YYMMDD` period are the only keys. Reruns are idempotent because `SKIP_COMPLETED=True` checks for the final output file and short-circuits.
- **Checkpoint + resume.** Long jobs write `체크포인트_*.json` (URL stage) or `체크포인트_본문_*.json` (body stage) every N items (`CHECKPOINT_INTERVAL`). On restart, the loop reads `next_i` and resumes mid-month. Checkpoints are deleted on clean completion; they persist if any URL failed retry so the failure list isn't lost.
- **Two-pass failure handling.** Errors during the main loop go into `err_idx`. After the month finishes, the loop retries `err_idx` once. Anything still failing is written to `본문_bs4_재실패_{query}_{period}.json` for the Selenium retry notebook to pick up.
- **Polite scraping.** Every notebook uses the same hardcoded `USER_AGENT`, random sleeps between requests (`ARTICLE_PAUSE_RANGE_SEC`) and between months (`JOB_PAUSE_RANGE_SEC`), and removes the `navigator.webdriver` flag. Don't lower these without reason — they're tuned to avoid Naver rate-limiting.
- **Dual environment (Colab + local).** Files ending in `_colab.ipynb` mount Google Drive and assume `PROJECT_DIR=/content/drive/MyDrive/Text-data-Analysis_26-Spring`. Local-only files (no `_colab` suffix) detect `Path.cwd()` and fall back to `/home/carol/Text-data-Analysis_26-Spring`. Both write to the same `data/` directory layout, so files round-trip between the two environments via the synced Drive folder.
- **Korean filename NFC/NFD.** Files synced via Google Drive may have NFD-normalized Korean names while local Linux uses NFC. All stats/merge notebooks call `unicodedata.normalize('NFC', name)` before matching. Preserve this when adding new file discovery code.
- **BS4 vs Selenium for body extraction.** BS4 is the default — Naver embeds article body in the initial HTML, so `requests` works and is ~10x faster. Selenium is reserved for the retry pass (`본문_재시도_selenium_local.ipynb`) on URLs that BS4 couldn't parse (typically JS-rendered or transient errors). The Selenium retry merges successes back into the existing BS4 CSV (with `.csv.bak_before_selenium_retry` backup) rather than producing a parallel file.

### Same carrier, two queries

SKT and "SK텔레콤" are run as separate queries (same for LG U+ / LG유플러스) and produce separate output files; the merge stage keeps them separate (one `통합_본문_bs4_{query}.csv` per query string). If you need a single per-carrier file, set `MAKE_ALL_COMBINED=True` in `본문_통합_colab.ipynb` — but be aware it dedupes by `link` only, so the same article surfaced by both query variants will be kept once.

## Running things

There are no CLI commands — open the relevant notebook in Jupyter/VS Code/Colab and run cells top-to-bottom. Required packages: `selenium`, `beautifulsoup4`, `requests`, `pandas`. The Selenium notebooks rely on Selenium Manager to auto-fetch ChromeDriver, so a working Chrome install is enough.

For local Selenium on WSL: `HEADLESS=False` is the committed default (so you can watch the browser). Switch to `True` for unattended runs.

## When you're asked to make changes

- Editing a notebook? Most logic is duplicated across the local and `_colab` variants (URL collection, body collection, Selenium retry). If you change the extraction logic or filename convention in one, check whether the other variant needs the same change — they are kept deliberately in sync.
- Adding a new query/period: edit `query_ranges` in the relevant notebook, don't add a new notebook.
- Don't commit the large CSVs in `data/` unless explicitly asked — some files are >70 MB. The committed `notebook/crawling/data/통합_본문_bs4_*.csv` files are tracked as data artifacts; treat them as such, not as code.
