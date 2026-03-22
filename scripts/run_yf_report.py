from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yfinance as yf

import matplotlib
import matplotlib.dates as mdates

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ticker_mapping import build_unresolved_report, resolve_tickers_from_transcript, validate_tickers
from llm_gemini_report import (
    GeminiConfig,
    GeminiReportWriter,
    get_gemini_api_key,
    load_dotenv_if_present,
)

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore


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
    for col in ["Close", "Adj Close"]:
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
            close_col = "Close" if (ticker, "Close") in df.columns else "Adj Close"
            if (ticker, close_col) not in df.columns:
                # Fallback: try adjusted close when close is unavailable.
                if (ticker, "Adj Close") in df.columns:
                    close_col = "Adj Close"
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


def format_period_cn(period: str) -> str:
    """
    Convert yfinance-style period (e.g. "3mo") into Chinese (e.g. "3個月").
    """
    match = re.match(r"^(\d+)mo$", period.strip(), flags=re.IGNORECASE)
    if match:
        return f"{int(match.group(1))}個月"
    return period


def format_pct(x: float) -> str:
    return f"{x:+.2f}%"


def format_pct_abs(x: float) -> str:
    return f"{x:.2f}%"


def format_narrative_score(raw_score: Optional[float]) -> str:
    if not isinstance(raw_score, (int, float)):
        return "?"

    score = float(raw_score)
    if 0.0 <= score <= 1.0:
        score *= 10.0
    score = max(0.0, min(score, 10.0))
    return f"{score:.1f}/10"


def _risk_band_from_max_drawdown(max_drawdown_pct: float) -> str:
    """
    Classify risk using max drawdown magnitude.
    """
    if max_drawdown_pct <= -25.0:
        return "風險明顯偏高"
    if max_drawdown_pct <= -15.0:
        return "回撤中等"
    return "回撤相對受控"


def _volatility_band_from_3m_volatility(volatility_3m: float) -> str:
    """
    Classify volatility band for short-term narrative.
    """
    if volatility_3m >= 4.0:
        return "波動偏高"
    if volatility_3m >= 2.5:
        return "波動中等"
    return "波動相對溫和"


def _theme_focus_phrase(theme_snippet: str) -> str:
    """
    Convert a short theme snippet into a more explainable focus phrase.
    """
    if "先進製程與供應鏈" in theme_snippet:
        return "供應鏈與資本支出（產能/節奏）"
    if "AI工廠與平台化" in theme_snippet:
        return "模型擴張與設備/平台需求敘事"
    if "記憶體與計算記憶體" in theme_snippet:
        return "記憶體供需與計算效率（如 HBM 路線）"
    if "事件驅動：Apple發表與CPO節奏" in theme_snippet:
        return "產品發表後的資本支出與供應鏈反應"
    if "預期落差與追價風險" in theme_snippet:
        return "估值與 EPS/上修路徑的風險偏好"
    if "OpenAI/雲端與企業軟體題材" in theme_snippet:
        return "雲端成長與企業軟體續航敘事"
    return "節目敘事主軸"


def _build_per_ticker_interpretation(
    period: str,
    theme_snippet: str,
    metrics: MarketMetrics,
) -> str:
    """
    Build a conclusion-like interpretation sentence for one ticker.
    """
    risk_band = _risk_band_from_max_drawdown(metrics.max_drawdown_pct)
    volatility_band = _volatility_band_from_3m_volatility(metrics.volatility_3m)
    focus_phrase = _theme_focus_phrase(theme_snippet)

    if metrics.total_return_pct >= 0:
        return (
            f"在「{theme_snippet}」的敘事脈絡下，價格近 {period} 呈現正向定價；"
            f"其中 {focus_phrase} 被市場階段性接受，但 {risk_band}、{volatility_band} 也意味著仍存在驗證/風險溢酬調整。"
        )

    return (
        f"在「{theme_snippet}」的敘事脈絡下，價格近 {period} 走弱；"
        f"對 {focus_phrase} 的短期定價仍偏保守，且 {risk_band}、{volatility_band} 反映出市場尚在消化落差與不確定性。"
    )


def ticker_to_label(ticker: str) -> str:
    config = _load_report_config()
    return config.ticker_labels.get(ticker, ticker)


@dataclass(frozen=True)
class ReportThemeKeyword:
    theme: str
    keywords: List[str]


@dataclass(frozen=True)
class ReportConfig:
    ticker_labels: Dict[str, str]
    theme_keywords: List[ReportThemeKeyword]
    theme_to_ticker_snippet: Dict[str, List[str]]


_REPORT_CONFIG_CACHE: Optional[ReportConfig] = None


def _load_report_config() -> ReportConfig:
    """
    Load theme/label configuration from YAML to avoid hardcoded report text.
    """

    global _REPORT_CONFIG_CACHE
    if _REPORT_CONFIG_CACHE is not None:
        return _REPORT_CONFIG_CACHE

    if yaml is None:
        raise ModuleNotFoundError(
            "PyYAML is required to load report_config.yaml. Please install it (pip install pyyaml)."
        )

    config_path = Path(__file__).resolve().parent / "report_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("report_config.yaml must parse into a dict at the top level.")

    ticker_labels_raw = data.get("ticker_labels", {})
    if not isinstance(ticker_labels_raw, dict):
        raise ValueError("report_config.yaml: 'ticker_labels' must be a dict.")

    theme_keywords_raw = data.get("theme_keywords", [])
    if not isinstance(theme_keywords_raw, list):
        raise ValueError("report_config.yaml: 'theme_keywords' must be a list.")

    theme_keywords: List[ReportThemeKeyword] = []
    for item in theme_keywords_raw:
        if not isinstance(item, dict):
            continue
        theme = item.get("theme")
        keywords = item.get("keywords")
        if not isinstance(theme, str) or not isinstance(keywords, list):
            continue
        keyword_list = [k for k in keywords if isinstance(k, str)]
        theme_keywords.append(ReportThemeKeyword(theme=theme, keywords=keyword_list))

    theme_to_snippet_raw = data.get("theme_to_ticker_snippet", {})
    if not isinstance(theme_to_snippet_raw, dict):
        raise ValueError("report_config.yaml: 'theme_to_ticker_snippet' must be a dict.")

    theme_to_ticker_snippet: Dict[str, List[str]] = {}
    for ticker, wants in theme_to_snippet_raw.items():
        if not isinstance(ticker, str) or not isinstance(wants, list):
            continue
        theme_to_ticker_snippet[ticker] = [w for w in wants if isinstance(w, str)]

    _REPORT_CONFIG_CACHE = ReportConfig(
        ticker_labels={str(k): str(v) for k, v in ticker_labels_raw.items()},
        theme_keywords=theme_keywords,
        theme_to_ticker_snippet=theme_to_ticker_snippet,
    )

    return _REPORT_CONFIG_CACHE


def extract_theme_keywords(transcript_text: str) -> List[str]:
    """
    Extract high-level themes from transcript text using keyword matching.
    """

    transcript_norm = transcript_text.lower().replace("invidia", "nvidia")
    config = _load_report_config()

    matched_themes: List[str] = []
    transcript_compact = transcript_norm.replace(" ", "")
    for rule in config.theme_keywords:
        if any(k in transcript_compact for k in rule.keywords):
            matched_themes.append(rule.theme)

    # Deduplicate while preserving order
    seen: set[str] = set()
    out: List[str] = []
    for theme in matched_themes:
        if theme not in seen:
            out.append(theme)
            seen.add(theme)
    return out


def theme_to_ticker_snippet(ticker: str, themes: List[str]) -> str:
    """
    Map themes to ticker for report narrative. This is heuristic.
    """

    config = _load_report_config()
    wants = config.theme_to_ticker_snippet.get(ticker, [])
    selected = [w for w in wants if w in themes]
    if not selected:
        selected = wants[:2]
    return "、".join(selected[:2])


def plot_ticker_close_svg(output_dir: Path, ticker: str, close: pd.Series, period: str) -> None:
    plt.figure(figsize=(6.2, 3.6), dpi=150)
    ax = plt.gca()
    plt.plot(close.index, close.values, linewidth=1.6)
    plt.title(f"{ticker} - {period} Close", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.25)

    # Reduce x-axis label crowding for dense date ranges.
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

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
    ax = plt.gca()
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

    # Reduce x-axis label crowding for dense date ranges.
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

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
    use_llm: bool,
    llm_model: str,
    llm_temperature: float,
    llm_max_output_tokens: int,
    llm_transcript_max_chars: int,
) -> None:
    themes = extract_theme_keywords(transcript_text)

    def sort_key(m: MarketMetrics) -> float:
        return m.total_return_pct

    sorted_metrics = sorted(metrics_by_ticker.values(), key=sort_key, reverse=True)

    # Prepare LLM writer (best effort). If something fails, the report falls back to deterministic text.
    llm_writer: Optional[GeminiReportWriter] = None
    transcript_for_llm = transcript_text
    if len(transcript_for_llm) > llm_transcript_max_chars:
        transcript_for_llm = transcript_for_llm[:llm_transcript_max_chars]

    if use_llm:
        api_key = get_gemini_api_key()
        if api_key:
            try:
                llm_writer = GeminiReportWriter(
                    output_dir=output_dir,
                    config=GeminiConfig(
                        model=llm_model,
                        temperature=llm_temperature,
                        max_output_tokens=llm_max_output_tokens,
                    ),
                    api_key=api_key,
                )
            except Exception:
                llm_writer = None

    # Print LLM errors at most a few times to avoid flooding console output.
    llm_error_printed = 0

    lines: List[str] = []
    lines.append(f"# {episode_prefix} 走勢與觀點綜合報告")
    lines.append("")
    lines.append(
        f"本報告把節目逐字稿（`{report_md_path.name}` 對應上游檔案來源）中反覆出現的投資框架，與 `Yahoo Finance` 近{format_period_cn(period)}行情走勢（SVG）做對照，用於檢視「敘事是否被市場定價」與「事件後的落差」可能如何體現在價格上。"
    )
    lines.append("")

    # Executive summary
    lines.append("## Executive summary")
    if sorted_metrics:
        top = sorted_metrics[:3]
        bottom = sorted_metrics[-3:]
        if llm_writer:
            try:
                assets_payload = [
                    {
                        "ticker_id": m.ticker,
                        "name": m.label,
                        "total_return_pct": m.total_return_pct,
                        "max_drawdown_pct": m.max_drawdown_pct,
                        "volatility_3m": m.volatility_3m,
                        "start_price": m.start_price,
                        "end_price": m.end_price,
                    }
                    for m in sorted_metrics
                ]
                market_summary_payload = {"assets": assets_payload}

                llm_exec = llm_writer.generate_executive_summary(
                    episode_prefix=episode_prefix,
                    period=period,
                    themes=themes,
                    transcript_text=transcript_for_llm,
                    market_metrics_summary=market_summary_payload,
                )

                if llm_exec:
                    market_sentiment = llm_exec.get("market_sentiment")
                    if isinstance(market_sentiment, dict):
                        trend_phase = market_sentiment.get("trend_phase")
                        risk_score = market_sentiment.get("risk_score")
                        if isinstance(trend_phase, str) and trend_phase.strip():
                            lines.append(f"**趨勢階段：** {trend_phase.strip()}")
                        if isinstance(risk_score, (int, float)):
                            lines.append(f"**風險分數：** {risk_score}/10。")

                        inflection_analysis = market_sentiment.get("inflection_analysis")
                        if isinstance(inflection_analysis, dict):
                            leading_indicators = inflection_analysis.get("leading_indicators")
                            lagging_indicators = inflection_analysis.get("lagging_indicators")
                            trigger_condition = inflection_analysis.get("trigger_condition")
                            if isinstance(leading_indicators, str) and leading_indicators.strip():
                                lines.append(f"**領先指標：** {leading_indicators.strip()}")
                            if isinstance(lagging_indicators, str) and lagging_indicators.strip():
                                lines.append(f"**滯後指標：** {lagging_indicators.strip()}")
                            if isinstance(trigger_condition, str) and trigger_condition.strip():
                                lines.append(f"**觸發條件：** {trigger_condition.strip()}")

                    asset_analysis = llm_exec.get("asset_analysis")
                    if isinstance(asset_analysis, list) and asset_analysis:
                        scored_assets: List[Tuple[float, Dict[str, Any]]] = []
                        for item in asset_analysis:
                            if not isinstance(item, dict):
                                continue
                            correlation_analysis = item.get("correlation_analysis")
                            score_val: Optional[float] = None
                            if isinstance(correlation_analysis, dict):
                                raw_score = correlation_analysis.get("score")
                                if isinstance(raw_score, (int, float)):
                                    score_val = float(raw_score)
                            if score_val is None:
                                continue
                            scored_assets.append((score_val, item))

                        if scored_assets:
                            scored_assets.sort(key=lambda x: x[0], reverse=True)
                            top_assets = [pair[1] for pair in scored_assets[:3]]
                        else:
                            top_assets = asset_analysis[:3]

                        for item in top_assets:
                            if not isinstance(item, dict):
                                continue
                            name = item.get("name")
                            temporal_role = item.get("temporal_role")
                            action_plan = item.get("action_plan")
                            correlation_analysis = item.get("correlation_analysis")
                            score_text = "?"
                            classification = None
                            logic = None
                            if isinstance(correlation_analysis, dict):
                                raw_score = correlation_analysis.get("score")
                                score_text = format_narrative_score(raw_score)
                                classification = correlation_analysis.get("classification")
                                logic = correlation_analysis.get("logic")

                            if isinstance(name, str) and name.strip() and isinstance(temporal_role, str) and temporal_role.strip():
                                lines.append("")
                                lines.append(f"**{name.strip()}**")
                                lines.append(
                                    f"敘事評價：{score_text}（{classification or '—'}）；時序角色：{temporal_role.strip()}"
                                )
                                if isinstance(action_plan, str) and action_plan.strip():
                                    lines.append(f"操作建議：{action_plan.strip()}")
                                if isinstance(logic, str) and logic.strip():
                                    lines.append(f"邏輯：{logic.strip()}")
                else:
                    raise RuntimeError("Gemini executive summary returned no payload.")
            except Exception as exc:
                if llm_error_printed == 0 and llm_writer and llm_writer.last_error:
                    print(f"[LLM] Executive summary failed: {llm_writer.last_error}")
                    llm_error_printed = 1
                top_labels = ", ".join([m.label.split(" (")[0] for m in top])
                bottom_labels = ", ".join([m.label.split(" (")[0] for m in bottom])
                lines.append(f"- 近 {period} 報酬表現靠前：{top_labels}。")
                lines.append(f"- 近 {period} 相對落後：{bottom_labels}。")
                lines.append("- 節目強調的「預期落差、後續驗證、供應鏈節奏」更適合用價格落點是否同步來追蹤。")
        else:
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
    lines.append(f"- 行情週期：近{format_period_cn(period)}（日頻 1d，收盤價）")
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
    lines.append("| 標的 | 近期間報酬 | 最大回撤 | 起點價 | 最後收盤價 | 波動（每日報酬標準差, %） |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in sorted_metrics:
        lines.append(
            f"| {m.label} | {format_pct_abs(m.total_return_pct)} | {format_pct_abs(m.max_drawdown_pct)} | {format_money(m.start_price)} | {format_money(m.end_price)} | {m.volatility_3m:.2f} |"
        )
    lines.append("")

    if llm_writer:
        try:
            assets_payload = [
                {
                    "ticker_id": m.ticker,
                    "name": m.label,
                    "total_return_pct": m.total_return_pct,
                    "max_drawdown_pct": m.max_drawdown_pct,
                    "volatility_3m": m.volatility_3m,
                    "start_price": m.start_price,
                    "end_price": m.end_price,
                }
                for m in sorted_metrics
            ]

            market_summary: Dict[str, Any] = {"assets": assets_payload, "ticker_count": {"value": len(sorted_metrics)}}
            overview_payload = llm_writer.generate_market_overview_narrative(
                period=period,
                themes=themes,
                transcript_text=transcript_for_llm,
                market_metrics_summary=market_summary,
            )
            if isinstance(overview_payload, dict):
                market_sentiment = overview_payload.get("market_sentiment")
                if isinstance(market_sentiment, dict):
                    trend_phase = market_sentiment.get("trend_phase")
                    risk_score = market_sentiment.get("risk_score")
                    inflection_analysis = market_sentiment.get("inflection_analysis")
                    if isinstance(trend_phase, str) and trend_phase.strip():
                            lines.append(f"**市場解讀：** 趨勢階段 {trend_phase.strip()}")
                    if isinstance(risk_score, (int, float)):
                            lines.append(f"**風險分數：** {risk_score}/10。")
                    if isinstance(inflection_analysis, dict):
                        leading_indicators = inflection_analysis.get("leading_indicators")
                        lagging_indicators = inflection_analysis.get("lagging_indicators")
                        trigger_condition = inflection_analysis.get("trigger_condition")
                        if isinstance(leading_indicators, str) and leading_indicators.strip():
                                lines.append(f"**領先指標：** {leading_indicators.strip()}")
                        if isinstance(lagging_indicators, str) and lagging_indicators.strip():
                                lines.append(f"**滯後指標：** {lagging_indicators.strip()}")
                        if isinstance(trigger_condition, str) and trigger_condition.strip():
                                lines.append(f"**觸發條件：** {trigger_condition.strip()}")

                asset_analysis = overview_payload.get("asset_analysis")
                if isinstance(asset_analysis, list) and asset_analysis:
                    for item in asset_analysis:
                        if not isinstance(item, dict):
                            continue
                        name = item.get("name")
                        temporal_role = item.get("temporal_role")
                        action_plan = item.get("action_plan")
                        correlation_analysis = item.get("correlation_analysis")
                        score_val = None
                        classification = None
                        logic = None
                        if isinstance(correlation_analysis, dict):
                            raw_score = correlation_analysis.get("score")
                            if isinstance(raw_score, (int, float)):
                                score_val = float(raw_score)
                            classification = correlation_analysis.get("classification")
                            logic = correlation_analysis.get("logic")
                        if isinstance(name, str) and name.strip() and isinstance(temporal_role, str) and temporal_role.strip():
                            score_text = format_narrative_score(score_val)
                            cls_text = classification.strip() if isinstance(classification, str) and classification.strip() else "—"
                            action_text = action_plan.strip() if isinstance(action_plan, str) and action_plan.strip() else ""
                            logic_text = logic.strip() if isinstance(logic, str) and logic.strip() else ""
                            lines.append("")
                            lines.append(f"**{name.strip()}**")
                            lines.append(
                                f"敘事評價：{score_text}（{cls_text}）；時序角色：{temporal_role.strip()}"
                            )
                            if action_text:
                                lines.append(f"操作建議：{action_text}")
                            if logic_text:
                                lines.append(f"邏輯：{logic_text}")
                lines.append("")
            elif llm_error_printed == 0 and llm_writer and llm_writer.last_error:
                print(f"[LLM] Market overview narrative failed (first occurrence): {llm_writer.last_error}")
                llm_error_printed = 1
        except Exception:
            pass

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
        theme_snippet = theme_to_ticker_snippet(m.ticker, themes)
        lines.append(f"**節目題材：** {theme_snippet}。")
        analysis: Optional[str] = None
        if llm_writer:
            try:
                metrics_payload = {
                    "start_price": m.start_price,
                    "end_price": m.end_price,
                    "total_return_pct": m.total_return_pct,
                    "max_drawdown_pct": m.max_drawdown_pct,
                    "volatility_3m": m.volatility_3m,
                }
                analysis = llm_writer.generate_per_ticker_analysis(
                    period=period,
                    ticker_label=m.label,
                    ticker_id=m.ticker,
                    theme_snippet=theme_snippet,
                    metrics=metrics_payload,
                    transcript_text=transcript_for_llm,
                )
            except Exception:
                analysis = None

        if analysis is None and llm_error_printed == 0 and llm_writer and llm_writer.last_error:
            print(f"[LLM] Per-ticker analysis failed (first occurrence): {llm_writer.last_error}")
            llm_error_printed = 1

        if analysis:
            lines.append(
                f"**近{format_period_cn(period)}市場結果：** 報酬 {format_pct_abs(m.total_return_pct)}；最大回撤 {format_pct_abs(m.max_drawdown_pct)}。"
            )
            lines.append("")
            lines.append("**解讀：**")
            lines.append("")
            lines.append(analysis)
        else:
            interpretation = _build_per_ticker_interpretation(
                period=period,
                theme_snippet=theme_snippet,
                metrics=m,
            )
            lines.append(
                f"**近{format_period_cn(period)}市場結果：** 報酬 {format_pct_abs(m.total_return_pct)}；最大回撤 {format_pct_abs(m.max_drawdown_pct)}。"
            )
            lines.append("")
            lines.append("**解讀：**")
            lines.append("")
            lines.append(interpretation)
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
    load_dotenv_if_present(SCRIPT_DIR.parent)

    parser = argparse.ArgumentParser(description="Generate Yahoo Finance charts and an episode report.")
    parser.add_argument("--youtube_url", type=str, default="")
    parser.add_argument("--transcript_md_path", type=str, default="")
    parser.add_argument("--period", type=str, default="3mo")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--episode_prefix", type=str, default="")
    parser.add_argument("--language_code", type=str, default="")
    parser.add_argument("--ticker_map_path", type=str, default="")
    parser.add_argument("--no_llm", action="store_true", help="Disable Gemini LLM generation; use deterministic templates.")
    parser.add_argument("--llm_model", type=str, default="", help="Gemini model name (overrides GEMINI_MODEL when provided).")
    parser.add_argument("--llm_temperature", type=float, default=0.3)
    parser.add_argument("--llm_max_output_tokens", type=int, default=900)
    parser.add_argument("--llm_transcript_max_chars", type=int, default=8000)
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

    # Prefer env-configured model unless user explicitly passes --llm_model.
    llm_model_effective = args.llm_model.strip() or os.environ.get("GEMINI_MODEL") or "models/gemini-2.5-pro"

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
        use_llm=(not args.no_llm) and (get_gemini_api_key() is not None),
        llm_model=llm_model_effective,
        llm_temperature=args.llm_temperature,
        llm_max_output_tokens=args.llm_max_output_tokens,
        llm_transcript_max_chars=args.llm_transcript_max_chars,
    )

    print(f"Report: {report_md_path}")
    if unresolved:
        print(f"Unresolved: {len(unresolved)} tickers -> {out_dir / 'unresolved.txt'}")


if __name__ == "__main__":
    main()

