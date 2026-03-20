---
name: ep-yf-report-skill
description: >-
  Build SVG stock charts and a detailed markdown report by extracting stock tickers
  from an episode transcript (via YouTube URL), downloading recent prices from Yahoo Finance,
  and matching episode investment themes to market performance. Use when the user provides
  a YouTube URL (or a local transcript .md file) and asks for "Yahoo Finance charts" or
  an "episode观点/走勢綜合報告".
---

# ep-yf-report-skill

## Quick Start
1. Ask for (or read) the episode transcript source:
   - Required: `youtube_url` (or ask for it if missing)
   - Optional: `transcript_md_path` (a local `ep*.md` file with the episode transcript)
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
- `youtube_url` (required): YouTube link for the episode audio/video.
- `transcript_md_path` (optional): Local transcript file path (example: `ep645-260320.md`).
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
- If YouTube transcript extraction fails:
  - Ask the user to provide `transcript_md_path` (local `ep*.md`) or paste the transcript text.
- If a candidate ticker mapping fails:
  - Continue processing the other tickers.
  - Record unresolved items to `unresolved.txt`.
- If Yahoo Finance download fails for a ticker:
  - Record the ticker to `unresolved.txt` with the error summary.

## Implementation Notes (for the agent)
- Prefer generating charts with English ticker labels to avoid missing CJK glyphs.
- Ensure report image links use relative paths that work from the markdown file location.
- Keep the report structure consistent so it can be compared across episodes.

