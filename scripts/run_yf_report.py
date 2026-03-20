from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yfinance as yf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ticker_mapping import build_unresolved_report, resolve_tickers_from_transcript, validate_tickers


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _infer_episode_prefix_from_transcript_md(transcript_md_path: str) -> Optional[str]:
    basename = os.path.basename(transcript_md_path)
    # Expected: ep645-260320.md or ep645-something.md
    match = re.match(r"(ep\d+)[-_].*\.md$", basename, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # Fallback: any "ep<digits>" token in file name.
    match = re.search(r"(ep\d+)", basename, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def _infer_output_dir(
    transcript_md_path: Optional[str],
    output_dir: Optional[str],
    episode_prefix: Optional[str],
) -> Tuple[Path, str]:
    """
    Decide output directory and episode prefix.
    """

    if output_dir:
        out = Path(output_dir).expanduser().resolve()
        if episode_prefix is None and transcript_md_path:
            episode_prefix = _infer_episode_prefix_from_transcript_md(transcript_md_path)
        if episode_prefix is None:
            episode_prefix = "ep"
        return out, episode_prefix

    if transcript_md_path:
        transcript_path = Path(transcript_md_path).expanduser().resolve()
        inferred_prefix = _infer_episode_prefix_from_transcript_md(transcript_md_path)
        if inferred_prefix is None:
            raise ValueError("Cannot infer episode_prefix from transcript filename. Please pass episode_prefix or output_dir.")
        out = transcript_path.parent / inferred_prefix
        return out, inferred_prefix

    if episode_prefix is None:
        raise ValueError("Either transcript_md_path or episode_prefix/output_dir must be provided.")

    # If no transcript file exists, the caller must provide output_dir explicitly.
    raise ValueError("output_dir must be provided when transcript_md_path is not available.")


def _select_close_column(df: pd.DataFrame) -> str:
    for col in ["Adj Close", "Close"]:
        if col in df.columns:
            return col
    # For some tickers, yfinance may return fewer fields.
    raise KeyError("No Close/Adj Close column found.")


def fetch_transcript_text(
    transcript_md_path: Optional[str],
) -> str:
    """
    Get transcript text (local file only).
    """

    if not transcript_md_path:
        raise ValueError(
            "transcript_md_path is required. youtube_url transcript extraction is disabled in this skill."
        )

    md_path = Path(transcript_md_path).expanduser().resolve()
    return md_path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class MarketMetrics:
    ticker: str
    label: str
    start_price: float
    end_price: float
    total_return_pct: float
    max_drawdown_pct: float
    volatility_3m: float


def compute_metrics_from_close_series(ticker: str, label: str, close: pd.Series) -> MarketMetrics:
    close = close.dropna()
    if len(close) < 5:
        raise ValueError(f"Not enough close points for {ticker}.")

    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])
    total_return_pct = (end_price / start_price - 1.0) * 100.0

    running_max = close.cummax()
    drawdown = close / running_max - 1.0
    max_drawdown_pct = float(drawdown.min() * 100.0)

    daily_returns = close.pct_change().dropna()
    volatility_3m = float(daily_returns.std() * 100.0)

    return MarketMetrics(
        ticker=ticker,
        label=label,
        start_price=start_price,
        end_price=end_price,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        volatility_3m=volatility_3m,
    )


def cache_paths(output_dir: Path, ticker: str, period: str, interval: str) -> Path:
    cache_dir = output_dir / "cache"
    _safe_mkdir(cache_dir)
    filename = f"{ticker}__{period}__{interval}__close.csv"
    return cache_dir / filename


def save_close_to_cache(path: Path, close: pd.Series) -> None:
    df = pd.DataFrame({"Date": close.index, "Close": close.values})
    df.to_csv(path, index=False, encoding="utf-8")


def load_close_from_cache(path: Path) -> Optional[pd.Series]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "Date" not in df.columns or "Close" not in df.columns:
        return None
    close = pd.Series(df["Close"].values, index=pd.to_datetime(df["Date"]))
    close = close.sort_index()
    return close


def yfinance_download_close_series(
    tickers: Sequence[str],
    period: str,
    interval: str,
) -> Dict[str, pd.Series]:
    """
    Download close/adj close series for multiple tickers in one call when possible.
    """

    if not tickers:
        return {}

    tickers_str = " ".join(tickers)
    df = yf.download(
        tickers_str,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        actions=False,
        group_by="ticker",
        threads=True,
    )
    if df is None or df.empty:
        raise RuntimeError("Yahoo Finance returned empty data.")

    out: Dict[str, pd.Series] = {}
    if isinstance(df.columns, pd.MultiIndex):
        # columns levels: [Ticker, Price]
        for ticker in tickers:
            # Determine close column
            close_col = "Adj Close" if (ticker, "Adj Close") in df.columns else "Close"
            if (ticker, close_col) not in df.columns:
                # Fallback: try Close
                if (ticker, "Close") in df.columns:
                    close_col = "Close"
                else:
                    continue
            series = df[(ticker, close_col)].dropna()
            out[ticker] = series
    else:
        # Single ticker mode: columns are price fields.
        # In this case, tickers length should be 1.
        only = tickers[0]
        close_col = _select_close_column(df)
        out[only] = df[close_col].dropna()

    return out


def format_money(x: float) -> str:
    return f"{x:,.2f}"


def format_pct(x: float) -> str:
    return f"{x:+.2f}%"


def format_pct_abs(x: float) -> str:
    return f"{x:.2f}%"


def ticker_to_label(ticker: str) -> str:
    mapping: Dict[str, str] = {
        "TSM": "台積電 TSMC (TSM)",
        "TSLA": "特斯拉 Tesla (TSLA)",
        "NVDA": "輝達 NVIDIA (NVDA)",
        "AAPL": "蘋果 Apple (AAPL)",
        "MSFT": "微軟 Microsoft (MSFT)",
        "ORCL": "甲骨文 Oracle (ORCL)",
        "AMZN": "亞馬遜 Amazon (AMZN)",
        "GOOGL": "Google (GOOGL)",
        "AMD": "超微 AMD (AMD)",
        "005930.KS": "三星 Samsung (005930.KS)",
        "MU": "美光 Micron (MU)",
        "2345.TW": "華邦 Winbond (2345.TW)",
        "2454.TW": "聯發科 MediaTek (2454.TW)",
        "AVGO": "Broadcom (AVGO)",
    }
    return mapping.get(ticker, ticker)


def extract_theme_keywords(transcript_text: str) -> List[str]:
    """
    Extract high-level themes from transcript text using keyword matching.
    """

    transcript_norm = transcript_text.lower().replace("invidia", "nvidia")

    themes: List[str] = []
    checks: List[Tuple[str, List[str]]] = [
        ("AI工廠與平台化", ["nvidia", "gtc", "spectrum", "lpu", "gpu", "token", "ai factory", "workshop"]),
        ("先進製程與供應鏈", ["tsmc", "台積電", "samsung", "三星", "order", "supply"]),
        ("事件驅動：Apple發表與CPO節奏", ["apple", "cpo", "發表", "earnings", "財報"]),
        ("記憶體與計算記憶體", ["micron", "美光", "winbond", "華邦", "cube", "sram", "hbm", "memory computing", "unchip"]),
        ("預期落差與追價風險", ["sellside", "上修", "priced", "eps", "股價"]),
        ("OpenAI/雲端與企業軟體題材", ["open ai", "openai", "oracle", "microsoft", "google", "azure"]),
    ]

    for theme, keywords in checks:
        if any(k in transcript_norm.replace(" ", "") for k in keywords):
            themes.append(theme)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for t in themes:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def theme_to_ticker_snippet(ticker: str, themes: List[str]) -> str:
    """
    Map themes to ticker for report narrative. This is heuristic.
    """

    mapping: Dict[str, List[str]] = {
        "TSM": ["先進製程與供應鏈"],
        "NVDA": ["AI工廠與平台化"],
        "AMD": ["AI工廠與平台化"],
        "AAPL": ["事件驅動：Apple發表與CPO節奏"],
        "MSFT": ["OpenAI/雲端與企業軟體題材", "預期落差與追價風險"],
        "ORCL": ["OpenAI/雲端與企業軟體題材"],
        "AMZN": ["AI工廠與平台化", "預期落差與追價風險"],
        "GOOGL": ["OpenAI/雲端與企業軟體題材"],
        "MU": ["記憶體與計算記憶體"],
        "2345.TW": ["記憶體與計算記憶體"],
        "2454.TW": ["記憶體與計算記憶體", "先進製程與供應鏈"],
        "005930.KS": ["記憶體與計算記憶體", "先進製程與供應鏈"],
        "AVGO": ["記憶體與計算記憶體", "AI工廠與平台化"],
        "TSLA": ["先進製程與供應鏈"],
    }
    wants = mapping.get(ticker, [])
    selected = [w for w in wants if w in themes]
    if not selected:
        selected = wants[:2]
    return "、".join(selected[:2])


def plot_ticker_close_svg(output_dir: Path, ticker: str, close: pd.Series, period: str) -> None:
    plt.figure(figsize=(6.2, 3.6), dpi=150)
    plt.plot(close.index, close.values, linewidth=1.6)
    plt.title(f"{ticker} - {period} Close", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path = output_dir / f"{ticker}.svg"
    plt.savefig(out_path, format="svg")
    plt.close()


def plot_normalized_comparison_svg(
    output_dir: Path,
    episode_prefix: str,
    series_by_ticker: Dict[str, pd.Series],
    period: str,
) -> None:
    common_index: Optional[pd.DatetimeIndex] = None
    for s in series_by_ticker.values():
        idx = s.index
        common_index = idx if common_index is None else common_index.intersection(idx)

    if common_index is None or len(common_index) < 5:
        return

    plt.figure(figsize=(7.8, 4.8), dpi=160)
    for ticker, s in series_by_ticker.items():
        s2 = s.loc[common_index]
        if s2.empty:
            continue
        base = float(s2.iloc[0])
        norm = s2 / base * 100.0
        plt.plot(norm.index, norm.values, linewidth=1.5, label=ticker)

    plt.title(f"Normalized close comparison (Start=100), {period}", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("Index (Start=100)")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_path = output_dir / f"{episode_prefix}-market-compare-normalized.svg"
    plt.savefig(out_path, format="svg")
    plt.close()


def generate_report_md(
    output_dir: Path,
    report_md_path: Path,
    episode_prefix: str,
    transcript_text: str,
    tickers: List[str],
    series_by_ticker: Dict[str, pd.Series],
    metrics_by_ticker: Dict[str, MarketMetrics],
    period: str,
    unresolved: List[str],
) -> None:
    themes = extract_theme_keywords(transcript_text)

    def sort_key(m: MarketMetrics) -> float:
        return m.total_return_pct

    sorted_metrics = sorted(metrics_by_ticker.values(), key=sort_key, reverse=True)

    lines: List[str] = []
    lines.append(f"# {episode_prefix} 走勢與觀點綜合報告")
    lines.append("")
    lines.append(
        f"本報告把節目逐字稿（`{report_md_path.name}` 對應上游檔案來源）中反覆出現的投資框架，與 `Yahoo Finance` 近 {period} 行情走勢（SVG）做對照，用於檢視「敘事是否被市場定價」與「事件後的落差」可能如何體現在價格上。"
    )
    lines.append("")

    # Executive summary
    lines.append("## Executive summary")
    if sorted_metrics:
        top = sorted_metrics[:3]
        bottom = sorted_metrics[-3:]
        top_labels = ", ".join([m.label.split(" (")[0] for m in top])
        bottom_labels = ", ".join([m.label.split(" (")[0] for m in bottom])
        lines.append(f"- 近 {period} 報酬表現靠前：{top_labels}。")
        lines.append(f"- 近 {period} 相對落後：{bottom_labels}。")
        lines.append("- 節目強調的「預期落差、後續驗證、供應鏈節奏」更適合用價格落點是否同步來追蹤。")
    else:
        lines.append("- 未能取得足夠行情資料，報告只包含部分內容。")
    lines.append("")

    lines.append("## Data scope")
    lines.append(f"- 節目逐字稿：由 skill 取得（來源：本地 transcript）。")
    lines.append(f"- 行情週期：近 {period}（日頻 1d，收盤價）")
    lines.append(f"- 圖表來源：`output_dir/*.svg` 與 `{episode_prefix}-market-compare-normalized.svg`.")
    lines.append("")

    # Themes
    lines.append("## 節目觀點重點（對照市場題材）")
    if themes:
        for t in themes:
            lines.append(f"- {t}")
    else:
        lines.append("- （未能穩定擷取關鍵字；仍會以整體節目脈絡生成對照。）")
    lines.append("")

    lines.append("## Market overview")
    lines.append("")
    lines.append("| 標的 | 近期間報酬 | 最大回撤 | 起點價 | 期末價 | 波動（每日報酬標準差, %） |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in sorted_metrics:
        lines.append(
            f"| {m.label} | {format_pct_abs(m.total_return_pct)} | {format_pct_abs(m.max_drawdown_pct)} | {format_money(m.start_price)} | {format_money(m.end_price)} | {m.volatility_3m:.2f} |"
        )
    lines.append("")

    normalized_chart_name = f"{episode_prefix}-market-compare-normalized.svg"
    normalized_chart_path = output_dir / normalized_chart_name
    if normalized_chart_path.exists():
        lines.append("## Normalized comparison chart")
        lines.append("")
        lines.append(f"![]({normalized_chart_name})")
        lines.append("")

    # Per-ticker
    lines.append("## 逐一標的：節目觀點 vs 近況走勢")
    lines.append("")
    for m in sorted_metrics:
        lines.append(f"### {m.label}")
        lines.append("")
        lines.append(f"- 近 {period} 報酬：{format_pct_abs(m.total_return_pct)}；最大回撤：{format_pct_abs(m.max_drawdown_pct)}。")
        lines.append(f"- 與節目觀點的對照：{theme_to_ticker_snippet(m.ticker, themes)}。")
        # Short inference (heuristic)
        if m.total_return_pct >= 0:
            lines.append("- 可能代表市場在這段期間更願意把「後續驗證/供需節奏」反映到價格，或風險溢酬下降。")
        else:
            lines.append("- 可能代表這段期間市場在「預期落差/估值重估」上仍偏保守，需等待下一輪驗證訊號。")
        lines.append("")
        if (output_dir / f"{m.ticker}.svg").exists():
            lines.append(f"![]({m.ticker}.svg)")
            lines.append("")
        lines.append("---")
        lines.append("")

    # Unresolved
    if unresolved:
        lines.append("## Unresolved mapping")
        lines.append("")
        lines.append("下列項目無法完整解析或下載失敗（已寫入 `unresolved.txt`，供你後續人工修正映射/標的）：")
        lines.append("")
        lines.append(build_unresolved_report(unresolved))
        lines.append("")

    report_md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Yahoo Finance charts and an episode report.")
    parser.add_argument("--youtube_url", type=str, default="")
    parser.add_argument("--transcript_md_path", type=str, default="")
    parser.add_argument("--period", type=str, default="3mo")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--episode_prefix", type=str, default="")
    parser.add_argument("--language_code", type=str, default="")
    parser.add_argument("--ticker_map_path", type=str, default="")
    args = parser.parse_args()

    youtube_url = args.youtube_url.strip() or None
    transcript_md_path = args.transcript_md_path.strip() or None
    period = args.period.strip() or "3mo"
    interval = args.interval.strip() or "1d"
    output_dir_arg = args.output_dir.strip() or None
    episode_prefix = args.episode_prefix.strip() or None
    language_code = args.language_code.strip() or None

    if transcript_md_path is None:
        raise ValueError("transcript_md_path is required. youtube_url transcript extraction is disabled in this skill.")

    # Default mapping file
    if args.ticker_map_path:
        ticker_map_path = args.ticker_map_path
    else:
        ticker_map_path = str(Path(__file__).resolve().parent / "ticker_map.yaml")

    out_dir, resolved_episode_prefix = _infer_output_dir(
        transcript_md_path=transcript_md_path,
        output_dir=output_dir_arg,
        episode_prefix=episode_prefix,
    )
    out_dir = out_dir.resolve()
    _safe_mkdir(out_dir)

    # Report name
    if transcript_md_path:
        transcript_path = Path(transcript_md_path).expanduser().resolve()
        base = transcript_path.stem
        report_md_path = out_dir / f"{base}-report.md"
    else:
        ts = datetime.now().strftime("%y%m%d-%H%M%S")
        report_md_path = out_dir / f"{resolved_episode_prefix}-{ts}-report.md"

    # 1) transcript text
    transcript_text = fetch_transcript_text(
        transcript_md_path=transcript_md_path,
    )

    # 2) resolve tickers from transcript
    resolved = resolve_tickers_from_transcript(transcript_text, ticker_map_path)
    resolved_tickers = resolved.resolved_tickers

    # 3) cache and download close series for resolved tickers
    series_by_ticker: Dict[str, pd.Series] = {}
    missing: List[str] = []

    for ticker in resolved_tickers:
        cpath = cache_paths(out_dir, ticker=ticker, period=period, interval=interval)
        cached = load_close_from_cache(cpath)
        if cached is not None and len(cached) > 5:
            series_by_ticker[ticker] = cached
        else:
            missing.append(ticker)

    # Batch download missing tickers
    unresolved: List[str] = []
    if missing:
        try:
            downloaded = yfinance_download_close_series(missing, period=period, interval=interval)
        except Exception:
            downloaded = {}

        for ticker in missing:
            series = downloaded.get(ticker)
            if series is None or series.empty or len(series) < 5:
                unresolved.append(ticker)
                continue
            series_by_ticker[ticker] = series
            cpath = cache_paths(out_dir, ticker=ticker, period=period, interval=interval)
            save_close_to_cache(cpath, series)

    # validate via download_ok (best effort)
    def download_ok_fn(t: str) -> bool:
        cpath = cache_paths(out_dir, ticker=t, period=period, interval=interval)
        s = load_close_from_cache(cpath)
        return s is not None and len(s) > 5

    unresolved_from_validate, reason_map = validate_tickers(resolved_tickers, download_ok_fn)
    # Merge unresolved lists but keep deterministic order.
    unresolved_set = set(unresolved)
    unresolved_set.update(unresolved_from_validate)
    unresolved = [t for t in resolved_tickers if t in unresolved_set]

    # Write unresolved.txt
    if unresolved:
        unresolved_path = out_dir / "unresolved.txt"
        unresolved_path.write_text(build_unresolved_report(unresolved, reason_map), encoding="utf-8")

    # 4) compute metrics and export charts
    metrics_by_ticker: Dict[str, MarketMetrics] = {}
    for ticker, close in series_by_ticker.items():
        metrics_by_ticker[ticker] = compute_metrics_from_close_series(
            ticker=ticker,
            label=ticker_to_label(ticker),
            close=close,
        )
        plot_ticker_close_svg(out_dir, ticker=ticker, close=close, period=period)

    # Normalized comparison chart
    plot_normalized_comparison_svg(out_dir, episode_prefix=resolved_episode_prefix, series_by_ticker=series_by_ticker, period=period)

    # 5) generate report
    generate_report_md(
        output_dir=out_dir,
        report_md_path=report_md_path,
        episode_prefix=resolved_episode_prefix,
        transcript_text=transcript_text,
        tickers=resolved_tickers,
        series_by_ticker=series_by_ticker,
        metrics_by_ticker=metrics_by_ticker,
        period=period,
        unresolved=unresolved,
    )

    print(f"Report: {report_md_path}")
    if unresolved:
        print(f"Unresolved: {len(unresolved)} tickers -> {out_dir / 'unresolved.txt'}")


if __name__ == "__main__":
    main()

