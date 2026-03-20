from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import typing

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore


@dataclass(frozen=True)
class TickerMappingResult:
    resolved_tickers: List[str]
    matched_patterns: Dict[str, List[str]]


def normalize_transcript_text(text: str) -> str:
    """
    Normalize transcript text for pattern matching.

    This function is intentionally conservative: it focuses on whitespace removal and
    a few common transcript noise fixes, because the input often arrives with
    characters split by spaces.
    """

    # Remove all whitespace (including full-width whitespace).
    normalized = re.sub(r"\s+", "", text, flags=re.UNICODE).lower()

    # Common ASR/typo fix observed in the episode transcript.
    normalized = normalized.replace("invidia", "nvidia")

    return normalized


def _normalize_mapping_key(key: str) -> str:
    # For alias keys like "nvidi a", we normalize them the same way as transcript text.
    key_norm = re.sub(r"\s+", "", key, flags=re.UNICODE).lower()
    return key_norm


def load_ticker_map(map_path: str) -> Dict[str, object]:
    """
    Load ticker mapping from YAML.
    """

    if yaml is None:
        raise ModuleNotFoundError(
            "PyYAML is required to load ticker_map.yaml. Please install it (pip install pyyaml)."
        )

    with open(map_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("ticker_map.yaml must parse into a dict at the top level.")
    return data


def resolve_tickers_from_transcript(transcript_text: str, map_path: str) -> TickerMappingResult:
    """
    Resolve Yahoo Finance tickers by matching transcript text against mapping patterns.
    """

    data = load_ticker_map(map_path)
    mappings_raw = data.get("mappings", {})
    aliases_raw = data.get("aliases", {})

    if not isinstance(mappings_raw, dict) or not isinstance(aliases_raw, dict):
        raise ValueError("Invalid ticker_map.yaml structure: expected 'mappings' and 'aliases'.")

    transcript_norm = normalize_transcript_text(transcript_text)

    resolved: Set[str] = set()
    matched_patterns: Dict[str, List[str]] = {}

    for ticker, cfg in mappings_raw.items():
        if not isinstance(cfg, dict):
            continue
        patterns = cfg.get("patterns", [])
        if not isinstance(patterns, list):
            continue

        hit: List[str] = []
        for p in patterns:
            if not isinstance(p, str):
                continue
            p_norm = _normalize_mapping_key(p)
            if p_norm and p_norm in transcript_norm:
                hit.append(p)

        if hit:
            resolved.add(str(ticker))
            matched_patterns[str(ticker)] = hit

    # Apply aliases (useful for shorthand or broken spellings).
    for alias_key, ticker in aliases_raw.items():
        if not isinstance(alias_key, str) or not isinstance(ticker, str):
            continue
        alias_norm = _normalize_mapping_key(alias_key)
        if alias_norm and alias_norm in transcript_norm:
            resolved.add(ticker)
            matched_patterns.setdefault(ticker, []).append(alias_key)

    return TickerMappingResult(
        resolved_tickers=sorted(resolved),
        matched_patterns={k: sorted(set(v)) for k, v in matched_patterns.items()},
    )


def build_unresolved_report(
    unresolved_candidates: List[str],
    unresolved_reason_map: Dict[str, str] | None = None,
) -> str:
    """
    Build a human-readable unresolved report line format.
    """

    unresolved_reason_map = unresolved_reason_map or {}
    lines: List[str] = []
    for item in unresolved_candidates:
        reason = unresolved_reason_map.get(item)
        if reason:
            lines.append(f"- {item}: {reason}")
        else:
            lines.append(f"- {item}")
    return "\n".join(lines) + ("\n" if lines else "")


def validate_tickers(
    tickers: List[str],
    download_ok: typing.Callable[[str], bool],
) -> Tuple[List[str], Dict[str, str]]:
    """
    Validate that each ticker is downloadable.

    The caller is responsible for implementing `download_ok` (e.g., via yfinance),
    so this utility stays independent from network and plotting layers.
    """

    resolved: List[str] = []
    unresolved: List[str] = []
    reason_map: Dict[str, str] = {}

    for ticker in tickers:
        try:
            ok = bool(download_ok(ticker))
        except Exception as exc:  # pragma: no cover
            ok = False
            reason_map[ticker] = f"download/check failed: {exc}"

        if ok:
            resolved.append(ticker)
        else:
            unresolved.append(ticker)
            reason_map.setdefault(ticker, "download/check returned empty or error")

    return unresolved, reason_map

