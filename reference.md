# ep-yf-report-skill Reference

This skill generates:

1. `*.svg` price charts (one per resolved Yahoo Finance ticker)
2. `ep<episode>-market-compare-normalized.svg` (a normalized comparison chart)
3. A detailed markdown report that combines episode themes (from transcript) and market metrics

## How tickers are extracted

The agent takes a transcript (from `youtube_url` or `transcript_md_path`), normalizes it, and then resolves tickers using:

- `scripts/ticker_map.yaml`
- `scripts/ticker_mapping.py` (pattern matching + optional aliases)

### Transcript normalization

Normalization is whitespace-robust:
- remove all whitespace
- lowercase
- apply a small set of known transcript noise fixes (e.g. `invidia` -> `nvidia`)

This approach matches transcripts where characters are split by spaces.

## Mapping maintenance

Ticker mapping file:

- `scripts/ticker_map.yaml`

You can safely extend `mappings.<ticker>.patterns` with additional patterns (company names, localized names, or common transcript spellings).

## Unresolved handling

The pipeline produces:
- `unresolved.txt` for tickers that could not be mapped or could not be downloaded/validated from Yahoo Finance.

The report generation continues with whatever tickers are available.

## Dependencies

The runtime script `scripts/run_yf_report.py` needs:
- `yfinance`
- `pandas`
- `matplotlib`
- `PyYAML` (for reading `ticker_map.yaml`)
- `youtube-transcript-api` (only when using `youtube_url`)

If `youtube-transcript-api` is missing and you only provide `transcript_md_path`, the skill still works without it.

## Local execution (debug)

Example (using an existing transcript file):

```bash
python scripts/run_yf_report.py \
  --transcript_md_path /path/to/ep645-260320.md \
  --output_dir /path/to/output/ep645 \
  --period 3mo \
  --interval 1d
```

Example (using YouTube URL):

```bash
python scripts/run_yf_report.py \
  --youtube_url "https://www.youtube.com/watch?v=..." \
  --episode_prefix ep645 \
  --output_dir /path/to/output/ep645 \
  --period 3mo \
  --interval 1d
```

