---
name: ep-yf-report-skill
description: >-
  Build SVG stock charts and a detailed markdown report by extracting stock tickers
  from a local episode transcript (.md), downloading recent prices from Yahoo Finance,
  and matching episode investment themes to market performance. Use when the user provides
  `transcript_md_path` and asks for "Yahoo Finance charts" or
  an "episode觀點/走勢綜合報告".
---

# ep-yf-report-skill
## Installation (local run)
1. Prerequisites: Python 3 (and internet access to Yahoo Finance).
2. From the repo root, create a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install yfinance pandas matplotlib pyyaml
   ```
4. Note: This skill uses **local** `transcript_md_path` only, so `youtube-transcript-api` is not required.

## Quick Start
1. Ask for the episode transcript source:
   - Required: `transcript_md_path` (a local `ep*.md` file with the episode transcript)
2. Extract stock candidates (company names and tickers) from the transcript.
3. Map candidates to Yahoo Finance tickers using `reference.md` mapping rules.
4. Download recent price data from Yahoo Finance (`period`/`interval`), and generate:
   - `*.svg` per ticker
   - a normalized comparison chart `ep<episode>-market-compare-normalized.svg`
5. Generate a detailed report markdown that combines:
   - the episode viewpoints/themes (from the transcript)
   - the computed market metrics (return, max drawdown, volatility)
   - inline embedding of the generated SVG files

## Inputs
- `transcript_md_path` (required): Local transcript file path (example: `ep645-260320.md`).
- `period` (optional, default `3mo`): Yahoo Finance lookback window.
- `interval` (optional, default `1d`): Yahoo Finance bar interval.
- `output_dir` (optional): Where to save SVGs and the report.
- `episode_prefix` (optional): Example `ep645`. Used when `output_dir` cannot be inferred.

## Output
- SVG charts:
  - `output_dir/<ticker>.svg` for each resolved Yahoo Finance ticker
  - `output_dir/<episode_prefix>-market-compare-normalized.svg` for the normalized comparison chart
- Report markdown:
  - If `transcript_md_path` is provided: `output_dir/<transcript_filename>-report.md`
  - Otherwise: `output_dir/<episode_prefix>-report.md` (with a timestamp if needed)
- Mapping failures:
  - `output_dir/unresolved.txt` listing transcript candidates that could not be mapped

## Failure Handling
- If reading `transcript_md_path` fails:
  - Ask the user to provide a valid `transcript_md_path` (local `ep*.md`) or paste the transcript text.
- If a candidate ticker mapping fails:
  - Continue processing the other tickers.
  - Record unresolved items to `unresolved.txt`.
- If Yahoo Finance download fails for a ticker:
  - Record the ticker to `unresolved.txt` with the error summary.

## Usage (local run)
Run the script directly:
```bash
python scripts/run_yf_report.py \
  --transcript_md_path "/path/to/ep645-260320.md" \
  --output_dir "/path/to/output/ep645" \
  --period 3mo \
  --interval 1d
```

### When `episode_prefix` is needed
If your transcript filename cannot be used to infer `ep<digits>` (for example, the name does not contain `ep123...`), pass:
```bash
--episode_prefix ep645
```

## Usage (calling the skill in chat)
Provide `transcript_md_path`, then ask for one of the supported outputs, for example:
- `請使用 transcript_md_path=/path/to/ep645-260320.md，產出 Yahoo Finance charts 與 episode 走勢綜合報告。`

## Implementation Notes (for the agent)
- Prefer generating charts with English ticker labels to avoid missing CJK glyphs.
- Ensure report image links use relative paths that work from the markdown file location.
- Keep the report structure consistent so it can be compared across episodes.

