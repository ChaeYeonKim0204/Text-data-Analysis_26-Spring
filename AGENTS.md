# Repository Guidelines

## Project Structure & Module Organization
This repository is a notebook-based news crawling and text-analysis pipeline, not a packaged application. Core workflow notebooks live in `notebook/crawling/`. Raw and intermediate crawl artifacts are written to `notebook/crawling/data/`, weekly press-analysis inputs live in `news/`, reusable resources such as sentiment lexicons live in `resources/`, and generated charts/tables belong in `result/`. Keep top-level notes such as `README.md`, `분석_진행상황.md`, and method guides in Markdown.

## Build, Test, and Development Commands
There is no build step or CLI entrypoint. Run notebooks top-to-bottom in Jupyter or Colab:

- `jupyter lab notebook/crawling` opens the local notebook workspace.
- `git status --short` checks notebook and data-file diffs before committing.
- `rg "query_ranges|KEEP_TERMS|NEWS_STOP" notebook/crawling` finds the main configuration blocks quickly.

Required packages depend on the notebook: crawling uses `selenium`, `beautifulsoup4`, `requests`, and `pandas`; analysis notebooks also use tools such as `kiwipiepy`, `gensim`, `matplotlib`, and `wordcloud`.

## Coding Style & Naming Conventions
Use 4-space indentation in notebook code cells and prefer `snake_case` for Python names. Preserve the existing Korean stage-based filenames and the `_colab.ipynb` suffix for Google Drive variants. Output filenames are part of the pipeline contract: follow patterns like `링크_{query}_{YYMMDD}_{YYMMDD}.json`, `본문_bs4_{query}_{period}.csv`, and `통합_본문_bs4_{query}.csv`. Keep shared logic synchronized between local and `_colab` notebook pairs.

## Testing Guidelines
There is no automated test suite or coverage gate. Validate changes by rerunning the affected notebook on a narrow date range or one query, then confirm output schema, row counts, and resume files in `notebook/crawling/data/`. For preprocessing or tokenization edits, spot-check columns such as `body_cleaned`, `tokens`, `source_period`, and duplicate flags before trusting downstream analysis.

## Commit & Pull Request Guidelines
Recent history uses short, direct summaries such as `add`, `colab version code`, and `data collection finish`. Keep commit subjects brief, present tense, and scope-specific, for example `sync bs4 body extraction with colab notebook`. In pull requests, state which notebook stage changed, list affected input/output files, note any required reruns, and include screenshots only when a chart or notebook output changed materially. Avoid committing large generated CSV/PNG artifacts unless the data refresh itself is the purpose of the change.
