from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


try:
    from google import genai  # type: ignore
    from google.genai.types import GenerateContentConfig  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    genai = None  # type: ignore
    GenerateContentConfig = None  # type: ignore


CORRELATION_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "market_sentiment": {
            "type": "object",
            "properties": {
                "trend_phase": {"type": "string"},
                "risk_score": {"type": "number"},
                "inflection_analysis": {
                    "type": "object",
                    "properties": {
                        "leading_indicators": {"type": "string"},
                        "lagging_indicators": {"type": "string"},
                        "trigger_condition": {"type": "string"},
                    },
                    "required": ["leading_indicators", "lagging_indicators", "trigger_condition"],
                },
            },
            "required": ["trend_phase", "risk_score", "inflection_analysis"],
        },
        "asset_analysis": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "correlation_analysis": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "number"},
                            "classification": {"type": "string"},
                            "logic": {"type": "string"},
                        },
                        "required": ["score", "classification", "logic"],
                    },
                    "temporal_role": {"type": "string"},
                    "action_plan": {"type": "string"},
                },
                "required": ["name", "correlation_analysis", "temporal_role", "action_plan"],
            },
        },
    },
    "required": ["market_sentiment", "asset_analysis"],
}


@dataclass(frozen=True)
class GeminiConfig:
    model: str
    temperature: float
    max_output_tokens: int


def load_dotenv_if_present(project_root: Path) -> None:
    dotenv_path = project_root / ".env"
    if not dotenv_path.exists():
        return

    try:
        content = dotenv_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # If the file is not UTF-8, keep existing environment variables.
        return

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ and value:
            os.environ[key] = value


def get_gemini_api_key() -> Optional[str]:
    for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"]:
        api_key = os.environ.get(key)
        if api_key:
            return api_key
    return None


def _stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cached_text(cache_file: Path) -> Optional[str]:
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict) and isinstance(data.get("response_text"), str):
        return data["response_text"]
    return None


def _write_cached_text(cache_file: Path, response_text: str) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"response_text": response_text}, ensure_ascii=False), encoding="utf-8")


def _extract_json_object(text: str) -> Optional[str]:
    """
    Extract the first JSON object found in text.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    return match.group(0)


def _parse_json_from_response_text(response_text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    extracted = _extract_json_object(response_text)
    if extracted:
        parsed2 = json.loads(extracted)
        if isinstance(parsed2, dict):
            return parsed2
    raise ValueError("Gemini response did not contain a JSON object.")


def _build_transcript_excerpt(transcript_text: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", transcript_text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars]


def build_executive_prompt(
    *,
    episode_prefix: str,
    period: str,
    themes: List[str],
    transcript_text: str,
    market_metrics_summary: Dict[str, Any],
) -> str:
    transcript_excerpt = _build_transcript_excerpt(transcript_text, max_chars=6000)

    schema_hint = """{
  "market_sentiment": {
    "trend_phase": "趨勢階段",
    "risk_score": 6,
    "inflection_analysis": {
      "leading_indicators": "預測未來轉折的關鍵數據（如：美光波動率、CPO 研發進度）",
      "lagging_indicators": "用於確認趨勢結束的數據（如：軟體巨頭財報營收、終端出貨量）",
      "trigger_condition": "觸發策略變更的具體數值門檻"
    }
  },
  "asset_analysis": [
    {
      "name": "標的名稱",
      "correlation_analysis": {
        "score": 0.85,
        "classification": "真 AI 受惠 / 蹭熱度 / 邊緣受惠",
        "logic": "解釋數據與敘事為何高度/低度相關（例如：股價領先敘事爆發，代表資訊優勢已定價）"
      },
      "temporal_role": "領先/隨行/滯後",
      "action_plan": "基於相關性的操作建議"
    }
  ]
}"""

    return (
        "# Role\n"
        "你是一位精通「量化策略」與「產業基本面」的資深首席分析師。你擅長透過數據與敘事的相關性分析（Correlation Analysis），區分市場的「真價值」與「假題材」。\n\n"
        "# Task\n"
        "分析【節目逐字稿】與【市場數據】，產出一份具備「相關係數分析」與「時序預警功能」的深度報告。\n\n"
        "# Input Data\n"
        f"1. Themes (目前敘事主軸): {themes}\n"
        f"2. Market Summary (市場指標): {json.dumps(market_metrics_summary, ensure_ascii=False)}\n"
        f"3. Transcript Excerpt (逐字稿節選): {transcript_excerpt}\n"
        f"4. Period (觀察區間): {period}\n\n"
        f"EpisodePrefix: {episode_prefix}\n\n"
        "# New Specific Requirements\n"
        "1. AI 敘事相關性分析 (Narrative Correlation):\n"
        " - 為每個標的計算一個「敘事得分 (0.0 - 1.0)」。\n"
        " - 高分 (0.7-1.0): 數據走勢與節目 AI 核心題材高度同步，且具備實質報酬支撐，定義為「真 AI 受惠股」。\n"
        " - 低分 (0.0-0.4): 僅有短期噴發但回撤巨大，或敘事熱烈但數據呈現負報酬，定義為「蹭熱度 (Hype-rider)」。\n\n"
        "2. 時序敏感度與指標分類 (Temporal Sensitivity):\n"
        " - 在轉折點分析中，明確標示「領先指標 (Leading)」與「滯後指標 (Lagging)」。\n"
        " - 範例：記憶體波動率 (Volatility) 視為領先指標；終端產品售價或營收確認視為滯後指標。\n\n"
        "3. 專業要求:\n"
        " - 使用繁體中文，語法冷靜且邏輯嚴密。\n"
        " - 嚴禁臆測數據，僅能從提供資料中歸納。\n"
        " - 輸出純 JSON，不含 Markdown 標籤。\n\n"
        "# Output JSON Schema\n"
        f"{schema_hint}\n"
    )


def build_market_overview_prompt(
    *,
    period: str,
    themes: List[str],
    transcript_text: str,
    market_metrics_summary: Dict[str, Any],
) -> str:
    transcript_excerpt = _build_transcript_excerpt(transcript_text, max_chars=5000)
    # Reuse the same correlation prompt schema for this section.
    return build_executive_prompt(
        episode_prefix="",
        period=period,
        themes=themes,
        transcript_text=transcript_excerpt,
        market_metrics_summary=market_metrics_summary,
    )


def build_per_ticker_prompt(
    *,
    period: str,
    ticker_label: str,
    ticker_id: str,
    theme_snippet: str,
    metrics: Dict[str, Any],
    transcript_text: str,
) -> str:
    transcript_excerpt = _build_transcript_excerpt(transcript_text, max_chars=3000)
    return (
        "你是一位資深股票/產業研究分析師。"
        "請針對單一標的，撰寫一段繁體中文的「更具體解讀」。"
        "你只能使用我提供的數字與敘事線索，不得臆測數字。"
        "\n\n"
        "輸出 JSON（只輸出一個物件），格式如下："
        '{'
        '"analysis":string'
        '}' 
        "\n\n"
        f"Period: {period}\n"
        f"TickerLabel: {ticker_label}\n"
        f"TickerId: {ticker_id}\n"
        f"ThemeSnippet: {theme_snippet}\n"
        f"Metrics: {json.dumps(metrics, ensure_ascii=False)}\n"
        "TranscriptExcerpt:\n"
        f"{transcript_excerpt}\n"
        "\n\n"
        "analysis 建議包含："
        "1) 用節目題材（ThemeSnippet）解釋這段期間的價格定價邏輯"
        "2) 用報酬、最大回撤、波動解釋市場風險偏好與驗證節奏"
        "3) 提出你認為後續需要觀察的 1-2 個指標（不要用外部資料捏造，只能用一般性的觀察點）"
    )


class GeminiReportWriter:
    def __init__(self, output_dir: Path, config: GeminiConfig, api_key: str):
        if genai is None or GenerateContentConfig is None:  # pragma: no cover
            raise ModuleNotFoundError("google-genai is required for LLM report generation.")
        self._output_dir = output_dir
        self._config = config
        self._api_key = api_key
        self._cache_dir = output_dir / "llm_cache"
        self._client = genai.Client(api_key=api_key)
        self._last_error: Optional[str] = None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def _cache_file(self, purpose: str, payload: Dict[str, Any], prompt_digest: str) -> Path:
        digest = _stable_hash(
            {"purpose": purpose, "payload": payload, "model": self._config.model, "prompt_digest": prompt_digest}
        )
        return self._cache_dir / f"{purpose}_{digest}.json"

    def _generate_json(self, purpose: str, prompt: str, payload_for_cache: Dict[str, Any]) -> Dict[str, Any]:
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_file = self._cache_file(purpose, payload_for_cache, prompt_digest=prompt_digest)
        cached = _read_cached_text(cache_file)
        if cached is not None:
            try:
                return _parse_json_from_response_text(cached)
            except Exception:
                # If cached response is not valid JSON, fall back to re-generation.
                pass

        config = GenerateContentConfig(
            temperature=self._config.temperature,
            maxOutputTokens=self._config.max_output_tokens,
            responseMimeType="application/json",
            responseSchema=CORRELATION_RESPONSE_SCHEMA if purpose in ["executive", "market_overview"] else None,
        )
        response = self._client.models.generate_content(model=self._config.model, contents=[prompt], config=config)
        response_text = str(getattr(response, "text", "") or "")
        _write_cached_text(cache_file, response_text)
        return _parse_json_from_response_text(response_text)

    def generate_executive_summary(
        self,
        *,
        episode_prefix: str,
        period: str,
        themes: List[str],
        transcript_text: str,
        market_metrics_summary: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        self._last_error = None
        prompt = build_executive_prompt(
            episode_prefix=episode_prefix,
            period=period,
            themes=themes,
            transcript_text=transcript_text,
            market_metrics_summary=market_metrics_summary,
        )
        try:
            return self._generate_json(
                purpose="executive",
                prompt=prompt,
                payload_for_cache={
                    "episode_prefix": episode_prefix,
                    "period": period,
                    "themes": themes,
                    "market_metrics_summary": market_metrics_summary,
                },
            )
        except Exception as exc:
            self._last_error = str(exc)
            return None

    def generate_market_overview_narrative(
        self,
        *,
        period: str,
        themes: List[str],
        transcript_text: str,
        market_metrics_summary: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        self._last_error = None
        prompt = build_market_overview_prompt(
            period=period,
            themes=themes,
            transcript_text=transcript_text,
            market_metrics_summary=market_metrics_summary,
        )
        try:
            payload = self._generate_json(
                purpose="market_overview",
                prompt=prompt,
                payload_for_cache={"period": period, "themes": themes, "summary": market_metrics_summary},
            )
            if isinstance(payload, dict):
                return payload
            return None
        except Exception as exc:
            self._last_error = str(exc)
            return None

    def generate_per_ticker_analysis(
        self,
        *,
        period: str,
        ticker_label: str,
        ticker_id: str,
        theme_snippet: str,
        metrics: Dict[str, Any],
        transcript_text: str,
    ) -> Optional[str]:
        self._last_error = None
        prompt = build_per_ticker_prompt(
            period=period,
            ticker_label=ticker_label,
            ticker_id=ticker_id,
            theme_snippet=theme_snippet,
            metrics=metrics,
            transcript_text=transcript_text,
        )
        try:
            payload = self._generate_json(
                purpose=f"per_ticker_{ticker_id}",
                prompt=prompt,
                payload_for_cache={"period": period, "ticker_id": ticker_id, "theme_snippet": theme_snippet, "metrics": metrics},
            )
            analysis = payload.get("analysis")
            if isinstance(analysis, str):
                return analysis.strip()
            return None
        except Exception as exc:
            self._last_error = str(exc)
            return None

