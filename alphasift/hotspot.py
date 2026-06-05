# -*- coding: utf-8 -*-
"""Topic-first hotspot discovery and detail helpers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from alphasift.industry import (
    _board_heat_score,
    _normalize_code,
    _safe_float,
    _safe_text,
    load_board_heat_trends,
)


HOTSPOT_STAGES = (
    "初次异动",
    "确认扩散",
    "加速主升",
    "分歧放量",
    "降温退潮",
)


@dataclass
class HotspotSummary:
    topic: str
    name: str = ""
    source: str = ""
    rank: int | None = None
    change_pct: float | None = None
    heat_score: float = 50.0
    trend_score: float | None = None
    persistence_score: float | None = None
    cooling_score: float | None = None
    observations: int = 0
    state: str = ""
    stage: str = "初次异动"
    sample_stock_count: int = 0
    leaders: list[str] = field(default_factory=list)
    provider_used: str = ""
    fallback_used: bool = False
    source_errors: list[str] = field(default_factory=list)
    stale: bool = False
    stale_age_hours: float | None = None


@dataclass
class HotspotStock:
    code: str
    name: str = ""
    change_pct: float | None = None
    amount: float | None = None
    turnover_rate: float | None = None
    volume_ratio: float | None = None
    net_inflow: float | None = None
    is_limit_up: bool = False
    active_days: int = 0
    evidence_count: int = 0
    role: str = ""
    hot_stock_score: float = 0.0


@dataclass
class TimelineEvent:
    date: str
    source: str
    title: str
    event_type: str = "news"
    impact_score: float = 0.0
    related_codes: list[str] = field(default_factory=list)


@dataclass
class HotspotDetail:
    summary: HotspotSummary
    stocks: list[HotspotStock] = field(default_factory=list)
    timeline: list[TimelineEvent] = field(default_factory=list)


class HotspotResults(list[HotspotSummary]):
    """List-compatible hotspot result with degradation metadata."""

    def __init__(
        self,
        items: list[HotspotSummary] | None = None,
        *,
        provider_used: str = "",
        fallback_used: bool = False,
        source_errors: list[str] | None = None,
        stale: bool = False,
        stale_age_hours: float | None = None,
    ) -> None:
        super().__init__(items or [])
        self.provider_used = provider_used
        self.fallback_used = fallback_used
        self.source_errors = _dedupe_errors(source_errors or [])
        self.stale = stale
        self.stale_age_hours = stale_age_hours


def compute_hotspot_heat_score(change_pct: float | None, rank: float | None) -> float:
    """Compute board heat using the industry cache semantics."""
    return _board_heat_score(change_pct=change_pct, rank=rank)


def classify_hotspot_stage(
    *,
    state: str = "",
    trend_score: float | None = None,
    cooling_score: float | None = None,
    persistence_score: float | None = None,
    latest_score: float | None = None,
    observations: int | None = None,
) -> str:
    """Classify a hotspot into a coarse lifecycle stage."""
    state_text = _safe_text(state).lower()
    trend = _safe_float(trend_score) or 0.0
    cooling = _safe_float(cooling_score) or 0.0
    persistence = _safe_float(persistence_score) or 0.0
    latest = _safe_float(latest_score) or 0.0
    obs = int(_safe_float(observations) or 0)

    if state_text in {"weakening", "cooling"} and (latest < 60 or trend <= -5):
        return "降温退潮"
    if cooling >= 8 and (latest < 60 or trend <= -5):
        return "降温退潮"
    if cooling >= 5:
        return "分歧放量"
    if latest >= 75 and trend >= 8 and persistence >= 50:
        return "加速主升"
    if state_text == "persistent_hot" or persistence >= 66.6667:
        return "确认扩散"
    if trend >= 5 and obs >= 2:
        return "确认扩散"
    return "初次异动"


def score_hotspot_stock(row: dict[str, Any] | pd.Series) -> float:
    """Score a constituent stock for hotspot leadership strength."""
    change = _safe_float(_row_value(row, ["change_pct", "涨跌幅", "涨幅"])) or 0.0
    amount = _safe_float(_row_value(row, ["amount", "成交额", "成交金额"])) or 0.0
    turnover = _safe_float(_row_value(row, ["turnover_rate", "换手率"])) or 0.0
    volume_ratio = _safe_float(_row_value(row, ["volume_ratio", "量比"])) or 0.0
    net_inflow = _safe_float(_row_value(row, ["net_inflow", "主力净流入", "主力净流入-净额"])) or 0.0
    is_limit_up = _safe_bool(_row_value(row, ["is_limit_up", "涨停", "是否涨停"]))
    active_days = int(_safe_float(_row_value(row, ["active_days", "连续活跃", "活跃天数"])) or 0)
    evidence_count = int(_safe_float(_row_value(row, ["evidence_count", "证据数", "线索数"])) or 0)

    amount_score = 0.0
    if amount > 0:
        amount_score = _clamp((math.log10(amount) - 6.0) / 4.0 * 18.0, 0.0, 18.0)

    inflow_score = 0.0
    if net_inflow > 0:
        inflow_score = _clamp((math.log10(net_inflow) - 5.0) / 4.0 * 12.0, 0.0, 12.0)
    elif net_inflow < 0:
        inflow_score = -_clamp((math.log10(abs(net_inflow)) - 5.0) / 4.0 * 8.0, 0.0, 8.0)

    score = 35.0
    score += _clamp(change * 2.7, -18.0, 32.0)
    score += amount_score
    score += _clamp(turnover * 1.1, 0.0, 14.0)
    score += _clamp(volume_ratio * 3.0, 0.0, 12.0)
    score += inflow_score
    score += 8.0 if is_limit_up else 0.0
    score += _clamp(active_days * 2.5, 0.0, 8.0)
    score += _clamp(evidence_count * 2.0, 0.0, 8.0)
    return round(_clamp(score, 0.0, 100.0), 4)


def assign_stock_roles(scored_rows: list[dict[str, Any] | HotspotStock]) -> list[HotspotStock]:
    """Sort scored constituents and assign hotspot roles."""
    stocks = [_coerce_hotspot_stock(item) for item in scored_rows]
    stocks = sorted(
        stocks,
        key=lambda item: (
            item.hot_stock_score,
            item.change_pct if item.change_pct is not None else -999.0,
            item.amount if item.amount is not None else -1.0,
            item.code,
        ),
        reverse=True,
    )
    if not stocks:
        return []

    top_score = stocks[0].hot_stock_score
    for idx, stock in enumerate(stocks):
        change = stock.change_pct or 0.0
        if idx == 0 and stock.hot_stock_score >= 70:
            role = "核心龙头"
        elif idx <= 2 and stock.hot_stock_score >= max(68.0, top_score - 8.0) and change >= 5.0:
            role = "核心龙头"
        elif stock.hot_stock_score >= 62.0 and change >= 3.0:
            role = "助攻"
        elif stock.hot_stock_score >= 48.0 and change >= 0:
            role = "补涨"
        elif stock.hot_stock_score >= 38.0:
            role = "后排"
        else:
            role = "掉队"
        stock.role = role
    return stocks


def discover_hotspots(
    *,
    provider: str | object = "akshare",
    max_boards: int = 80,
    history_path: str | Path | None = None,
    fallback_cache_path: str | Path | None = None,
    top: int = 20,
) -> HotspotResults:
    """Discover ranked concept/industry hotspots from a provider."""
    source_errors: list[str] = []
    provider_chain = _resolve_provider_chain(provider, source_errors)
    provider_used = ""
    rows: list[dict[str, Any]] = []

    for label, provider_obj in provider_chain:
        provider_used = label or provider_used
        if provider_obj is None:
            continue
        error_count = len(source_errors)
        rows = _load_board_summaries(
            provider_obj,
            max_boards=max_boards,
            source_errors=source_errors,
            provider_label=label,
        )
        if rows:
            provider_used = label
            break
        if len(source_errors) == error_count:
            source_errors.append(f"{label}: returned no hotspot rows")

    if not rows:
        if fallback_cache_path and not source_errors and provider_used == "none":
            source_errors.append("none: no live provider requested")
        fallback = _load_fallback_hotspots(
            fallback_cache_path,
            source_errors=source_errors,
            top=top,
        )
        if fallback is not None:
            return fallback
        return HotspotResults(
            [],
            provider_used=provider_used,
            fallback_used=False,
            source_errors=source_errors,
        )

    trends = _load_history_trends(history_path)
    summaries: list[HotspotSummary] = []
    for row in rows:
        topic = row["topic"]
        trend = trends.get(topic, {})
        latest_score = _safe_float(trend.get("board_heat_latest_score"))
        trend_score = _safe_float(trend.get("board_heat_trend_score"))
        persistence_score = _safe_float(trend.get("board_heat_persistence_score"))
        cooling_score = _safe_float(trend.get("board_heat_cooling_score"))
        observations = int(_safe_float(trend.get("board_heat_observations")) or 0)
        state = _safe_text(trend.get("board_heat_state"))
        heat_score = latest_score if latest_score is not None else float(row["heat_score"])
        stage = classify_hotspot_stage(
            state=state,
            trend_score=trend_score,
            cooling_score=cooling_score,
            persistence_score=persistence_score,
            latest_score=heat_score,
            observations=observations,
        )
        summaries.append(HotspotSummary(
            topic=topic,
            name=topic,
            source=row.get("source", ""),
            rank=row.get("rank"),
            change_pct=row.get("change_pct"),
            heat_score=round(heat_score, 4),
            trend_score=trend_score,
            persistence_score=persistence_score,
            cooling_score=cooling_score,
            observations=observations,
            state=state,
            stage=stage,
        ))

    ranked = sorted(summaries, key=_hotspot_sort_key, reverse=True)[:max(int(top), 0)]
    for summary in ranked:
        provider_obj = next((obj for label, obj in provider_chain if label == provider_used), None)
        stocks = _load_scored_constituents(
            provider_obj,
            summary.topic,
            source=summary.source,
            source_errors=source_errors,
            provider_label=provider_used,
        ) if provider_obj is not None else []
        summary.sample_stock_count = len(stocks)
        summary.leaders = [stock.name or stock.code for stock in stocks if stock.role == "核心龙头"][:3]
        if not summary.leaders:
            summary.leaders = [stock.name or stock.code for stock in stocks[:3]]
    return _with_result_metadata(
        ranked,
        provider_used=provider_used,
        fallback_used=False,
        source_errors=source_errors,
        stale=False,
        stale_age_hours=None,
    )


def get_hotspot_detail(
    topic: str,
    *,
    provider: str | object = "akshare",
    top_stocks: int = 10,
    timeline_path: str | Path | None = None,
    history_path: str | Path | None = None,
    fallback_cache_path: str | Path | None = None,
) -> HotspotDetail:
    """Return one hotspot detail view with constituent stock roles and timeline."""
    topic_text = _safe_text(topic)
    source_errors: list[str] = []
    provider_chain = _resolve_provider_chain(provider, source_errors)
    provider_used = ""
    summary = HotspotSummary(topic=topic_text, name=topic_text, source="")
    stocks: list[HotspotStock] = []

    if topic_text:
        for label, provider_obj in provider_chain:
            provider_used = label or provider_used
            if provider_obj is None:
                continue
            row = _find_board_summary(
                provider_obj,
                topic_text,
                source_errors=source_errors,
                provider_label=label,
            )
            if row:
                row_heat = _safe_float(row.get("heat_score"))
                summary = HotspotSummary(
                    topic=topic_text,
                    name=topic_text,
                    source=row.get("source", ""),
                    rank=row.get("rank"),
                    change_pct=row.get("change_pct"),
                    heat_score=row_heat if row_heat is not None else 50.0,
                )
            source = summary.source or "concept"
            stocks = _load_scored_constituents(
                provider_obj,
                topic_text,
                source=source,
                source_errors=source_errors,
                provider_label=label,
            )
            if not stocks and source != "industry":
                stocks = _load_scored_constituents(
                    provider_obj,
                    topic_text,
                    source="industry",
                    source_errors=source_errors,
                    provider_label=label,
                )
                if stocks:
                    summary.source = "industry"
            if row or stocks:
                provider_used = label
                break

    if not stocks and not summary.source:
        fallback_summary = _find_fallback_hotspot(topic_text, fallback_cache_path, source_errors=source_errors)
        if fallback_summary is not None:
            stale_age = _cache_stale_age_hours(fallback_cache_path)
            summary = fallback_summary
            _apply_summary_metadata(
                summary,
                provider_used="last_good_cache",
                fallback_used=True,
                source_errors=source_errors or ["none: no live detail rows"],
                stale=True,
                stale_age_hours=stale_age,
            )

    trends = _load_history_trends(history_path)
    trend = trends.get(topic_text, {})
    latest_score = _safe_float(trend.get("board_heat_latest_score"))
    summary.heat_score = round(latest_score if latest_score is not None else summary.heat_score, 4)
    summary.trend_score = _safe_float(trend.get("board_heat_trend_score"))
    summary.persistence_score = _safe_float(trend.get("board_heat_persistence_score"))
    summary.cooling_score = _safe_float(trend.get("board_heat_cooling_score"))
    summary.observations = int(_safe_float(trend.get("board_heat_observations")) or 0)
    summary.state = _safe_text(trend.get("board_heat_state"))
    summary.stage = classify_hotspot_stage(
        state=summary.state,
        trend_score=summary.trend_score,
        cooling_score=summary.cooling_score,
        persistence_score=summary.persistence_score,
        latest_score=summary.heat_score,
        observations=summary.observations,
    )
    if not summary.fallback_used or stocks:
        summary.sample_stock_count = len(stocks)
        summary.leaders = [stock.name or stock.code for stock in stocks if stock.role == "核心龙头"][:3]
        if not summary.leaders:
            summary.leaders = [stock.name or stock.code for stock in stocks[:3]]
    if not summary.provider_used:
        _apply_summary_metadata(
            summary,
            provider_used=provider_used,
            fallback_used=False,
            source_errors=source_errors,
            stale=False,
            stale_age_hours=None,
        )

    timeline = load_hotspot_timeline(timeline_path, topic=topic_text) if timeline_path else []
    return HotspotDetail(summary=summary, stocks=stocks[:max(int(top_stocks), 0)], timeline=timeline)


def load_hotspot_history(path_like: str | Path) -> list[dict[str, Any]]:
    """Load hotspot history JSONL records, skipping malformed lines."""
    path = Path(path_like)
    if not path.is_file():
        raise FileNotFoundError(f"Hotspot history file not found: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        topic = _safe_text(item.get("topic") or item.get("board"))
        heat = _safe_float(_row_value(item, ["heat_score", "max_board_heat_score"]))
        generated_at = _safe_text(item.get("generated_at"))
        if not topic or heat is None or heat < 0 or heat > 100:
            continue
        item["topic"] = topic
        item["board"] = _safe_text(item.get("board")) or topic
        item["heat_score"] = heat
        item["max_board_heat_score"] = heat
        item["generated_at"] = generated_at
        rows.append(item)
    return sorted(rows, key=lambda item: (str(item.get("generated_at", "")), str(item.get("topic", ""))))


def append_hotspot_history(
    path_like: str | Path,
    hotspots: list[HotspotSummary | dict[str, Any]],
    *,
    generated_at: str,
) -> Path:
    """Append hotspot summaries to a trend-compatible JSONL history file."""
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for hotspot in hotspots:
            item = asdict(hotspot) if isinstance(hotspot, HotspotSummary) else dict(hotspot)
            topic = _safe_text(item.get("topic") or item.get("name"))
            heat = _safe_float(item.get("heat_score"))
            if not topic or heat is None:
                continue
            record = {
                "generated_at": generated_at,
                "topic": topic,
                "board": topic,
                "source": _safe_text(item.get("source")),
                "rank": item.get("rank"),
                "change_pct": item.get("change_pct"),
                "heat_score": heat,
                "max_board_heat_score": heat,
                "sample_stock_count": int(_safe_float(item.get("sample_stock_count")) or 0),
                "leaders": item.get("leaders") if isinstance(item.get("leaders"), list) else [],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_hotspot_timeline(path_like: str | Path, topic: str | None = None) -> list[TimelineEvent]:
    """Load timeline JSONL records, sorted by date, skipping malformed rows."""
    path = Path(path_like)
    if not path.is_file():
        raise FileNotFoundError(f"Hotspot timeline file not found: {path}")
    topic_text = _safe_text(topic)
    events: list[TimelineEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        if topic_text and not _timeline_matches_topic(item, topic_text):
            continue
        date = _safe_text(item.get("date") or item.get("generated_at") or item.get("time"))
        source = _safe_text(item.get("source"))
        title = _safe_text(item.get("title"))
        if not date or not source or not title:
            continue
        events.append(TimelineEvent(
            date=date,
            source=source,
            title=title,
            event_type=_safe_text(item.get("event_type")) or "news",
            impact_score=round(_safe_float(item.get("impact_score")) or 0.0, 4),
            related_codes=_normalize_related_codes(item.get("related_codes")),
        ))
    return sorted(events, key=lambda item: (item.date, item.source, item.title))


def load_hotspots_json(path_like: str | Path) -> list[HotspotSummary]:
    """Load a last-good hotspot cache, skipping malformed rows."""
    path = Path(path_like)
    if not path.is_file():
        raise FileNotFoundError(f"Hotspot cache file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if isinstance(payload, dict):
        raw_rows = (
            payload.get("hotspots")
            or payload.get("rows")
            or payload.get("items")
            or payload.get("data")
            or []
        )
    else:
        raw_rows = payload
    if not isinstance(raw_rows, list):
        return []

    rows: list[HotspotSummary] = []
    for item in raw_rows:
        summary = _coerce_hotspot_summary(item)
        if summary is not None:
            rows.append(summary)
    return rows


def save_hotspots_json(path_like: str | Path, hotspots: list[HotspotSummary]) -> Path:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(item) for item in hotspots], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def hotspot_detail_to_dict(detail: HotspotDetail) -> dict[str, Any]:
    return {
        "summary": asdict(detail.summary),
        "stocks": [asdict(item) for item in detail.stocks],
        "timeline": [asdict(item) for item in detail.timeline],
    }


def _resolve_provider(provider: str | object) -> object | None:
    source_errors: list[str] = []
    for _, provider_obj in _resolve_provider_chain(provider, source_errors):
        if provider_obj is not None:
            return provider_obj
    return None


def _resolve_provider_chain(
    provider: str | object,
    source_errors: list[str],
) -> list[tuple[str, object | None]]:
    if not isinstance(provider, str):
        return [(_provider_label(provider), provider)]

    names = [part.strip().lower() for part in provider.split(",") if part.strip()]
    if not names:
        names = ["none"]

    resolved: list[tuple[str, object | None]] = []
    for name in names:
        if name in {"none", "off", "false"}:
            resolved.append(("none", None))
            continue
        if name == "akshare":
            try:
                import akshare as ak
            except Exception as exc:  # noqa: BLE001 - provider import is optional.
                source_errors.append(f"akshare: {exc}")
                continue
            resolved.append(("akshare", ak))
            continue
        source_errors.append(f"unknown provider '{name}'")
    return resolved


def _load_board_summaries(
    provider: object,
    *,
    max_boards: int,
    source_errors: list[str] | None = None,
    provider_label: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        ("concept", "stock_board_concept_name_em"),
        ("industry", "stock_board_industry_name_em"),
    ]
    limit = max(int(max_boards), 1)
    for source, method_name in specs:
        frame = _call_provider_frame(
            provider,
            method_name,
            source_errors=source_errors,
            provider_label=provider_label,
        )
        if frame is None:
            frame = _mapping_provider_frame(provider, f"{source}_boards")
        if frame is None:
            continue
        rows.extend(_normalize_board_rows(frame, source=source)[:limit])
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        topic = row["topic"]
        existing = deduped.get(topic)
        if existing is None or _board_row_rank_key(row) > _board_row_rank_key(existing):
            deduped[topic] = row
    return sorted(deduped.values(), key=_board_row_rank_key, reverse=True)


def _normalize_board_rows(frame: pd.DataFrame, *, source: str) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for idx, row in frame.iterrows():
        topic = _safe_text(_row_value(row, [
            "topic",
            "board",
            "板块名称",
            "概念名称",
            "行业名称",
            "名称",
            "name",
        ]))
        if not topic:
            continue
        rank = _safe_float(_row_value(row, ["rank", "排名", "序号"]))
        if rank is None:
            rank = float(idx + 1)
        change_pct = _safe_float(_row_value(row, ["change_pct", "涨跌幅", "涨幅"]))
        heat_score = compute_hotspot_heat_score(change_pct, rank)
        rows.append({
            "topic": topic,
            "source": source,
            "rank": int(rank) if rank is not None else None,
            "change_pct": change_pct,
            "heat_score": heat_score,
        })
    return rows


def _find_board_summary(
    provider: object,
    topic: str,
    *,
    source_errors: list[str] | None = None,
    provider_label: str = "",
) -> dict[str, Any] | None:
    for row in _load_board_summaries(
        provider,
        max_boards=500,
        source_errors=source_errors,
        provider_label=provider_label,
    ):
        if row["topic"] == topic:
            return row
    return None


def _load_scored_constituents(
    provider: object,
    topic: str,
    *,
    source: str,
    source_errors: list[str] | None = None,
    provider_label: str = "",
) -> list[HotspotStock]:
    method_names = []
    if source == "industry":
        method_names.append("stock_board_industry_cons_em")
    else:
        method_names.append("stock_board_concept_cons_em")
        method_names.append("stock_board_industry_cons_em")
    frame = None
    for method_name in method_names:
        frame = _call_provider_frame(
            provider,
            method_name,
            source_errors=source_errors,
            provider_label=provider_label,
            symbol=topic,
        )
        if frame is not None:
            break
    if frame is None:
        frame = _mapping_constituents_frame(provider, topic, source=source)
    if frame is None:
        return []
    rows = [asdict(stock) for stock in _normalize_stock_rows(frame)]
    return assign_stock_roles(rows)


def _normalize_stock_rows(frame: pd.DataFrame) -> list[HotspotStock]:
    if frame is None or frame.empty:
        return []
    stocks: list[HotspotStock] = []
    for _, row in frame.iterrows():
        code = _normalize_code(_row_value(row, ["code", "代码", "证券代码"]))
        if not code:
            continue
        change_pct = _safe_float(_row_value(row, ["change_pct", "涨跌幅", "涨幅"]))
        amount = _safe_float(_row_value(row, ["amount", "成交额", "成交金额"]))
        turnover_rate = _safe_float(_row_value(row, ["turnover_rate", "换手率"]))
        volume_ratio = _safe_float(_row_value(row, ["volume_ratio", "量比"]))
        net_inflow = _safe_float(_row_value(row, ["net_inflow", "主力净流入", "主力净流入-净额"]))
        stock = HotspotStock(
            code=code,
            name=_safe_text(_row_value(row, ["name", "名称", "股票名称"])),
            change_pct=change_pct,
            amount=amount,
            turnover_rate=turnover_rate,
            volume_ratio=volume_ratio,
            net_inflow=net_inflow,
            is_limit_up=_safe_bool(_row_value(row, ["is_limit_up", "涨停", "是否涨停"])) or (change_pct or 0) >= 9.8,
            active_days=int(_safe_float(_row_value(row, ["active_days", "连续活跃", "活跃天数"])) or 0),
            evidence_count=int(_safe_float(_row_value(row, ["evidence_count", "证据数", "线索数"])) or 0),
        )
        stock.hot_stock_score = score_hotspot_stock(asdict(stock))
        stocks.append(stock)
    return stocks


def _coerce_hotspot_stock(item: dict[str, Any] | HotspotStock) -> HotspotStock:
    if isinstance(item, HotspotStock):
        stock = item
    else:
        stock = HotspotStock(
            code=_normalize_code(_row_value(item, ["code", "代码"])),
            name=_safe_text(_row_value(item, ["name", "名称"])),
            change_pct=_safe_float(_row_value(item, ["change_pct", "涨跌幅"])),
            amount=_safe_float(_row_value(item, ["amount", "成交额"])),
            turnover_rate=_safe_float(_row_value(item, ["turnover_rate", "换手率"])),
            volume_ratio=_safe_float(_row_value(item, ["volume_ratio", "量比"])),
            net_inflow=_safe_float(_row_value(item, ["net_inflow", "主力净流入"])),
            is_limit_up=_safe_bool(_row_value(item, ["is_limit_up", "涨停"])),
            active_days=int(_safe_float(_row_value(item, ["active_days", "连续活跃"])) or 0),
            evidence_count=int(_safe_float(_row_value(item, ["evidence_count", "证据数"])) or 0),
            hot_stock_score=_safe_float(item.get("hot_stock_score")) or 0.0,
        )
    if stock.hot_stock_score <= 0:
        stock.hot_stock_score = score_hotspot_stock(asdict(stock))
    return stock


def _coerce_hotspot_summary(item: object) -> HotspotSummary | None:
    if isinstance(item, HotspotSummary):
        return item
    if not isinstance(item, dict):
        return None
    topic = _safe_text(_row_value(item, ["topic", "name", "board", "hotspot"]))
    if not topic:
        return None
    heat_score = _safe_float(_row_value(item, ["heat_score", "max_board_heat_score"]))
    if heat_score is None:
        heat_score = 50.0
    if heat_score < 0 or heat_score > 100:
        return None
    rank = _safe_float(item.get("rank"))
    leaders = item.get("leaders")
    source_errors = item.get("source_errors")
    return HotspotSummary(
        topic=topic,
        name=_safe_text(item.get("name")) or topic,
        source=_safe_text(item.get("source")),
        rank=int(rank) if rank is not None else None,
        change_pct=_safe_float(item.get("change_pct")),
        heat_score=round(float(heat_score), 4),
        trend_score=_safe_float(item.get("trend_score")),
        persistence_score=_safe_float(item.get("persistence_score")),
        cooling_score=_safe_float(item.get("cooling_score")),
        observations=int(_safe_float(item.get("observations")) or 0),
        state=_safe_text(item.get("state")),
        stage=_safe_text(item.get("stage")) or "初次异动",
        sample_stock_count=int(_safe_float(item.get("sample_stock_count")) or 0),
        leaders=[_safe_text(value) for value in leaders if _safe_text(value)] if isinstance(leaders, list) else [],
        provider_used=_safe_text(item.get("provider_used")),
        fallback_used=_safe_bool(item.get("fallback_used")),
        source_errors=[
            _safe_text(value)
            for value in source_errors
            if _safe_text(value)
        ] if isinstance(source_errors, list) else [],
        stale=_safe_bool(item.get("stale")),
        stale_age_hours=_safe_float(item.get("stale_age_hours")),
    )


def _load_fallback_hotspots(
    fallback_cache_path: str | Path | None,
    *,
    source_errors: list[str],
    top: int,
) -> HotspotResults | None:
    if not fallback_cache_path:
        return None
    try:
        hotspots = load_hotspots_json(fallback_cache_path)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 - malformed fallback cache should not crash live flow.
        source_errors.append(f"last_good_cache: {exc}")
        return None
    if not hotspots:
        return None
    stale_age = _cache_stale_age_hours(fallback_cache_path)
    return _with_result_metadata(
        hotspots[:max(int(top), 0)],
        provider_used="last_good_cache",
        fallback_used=True,
        source_errors=source_errors or ["none: no live hotspot rows"],
        stale=True,
        stale_age_hours=stale_age,
    )


def _find_fallback_hotspot(
    topic: str,
    fallback_cache_path: str | Path | None,
    *,
    source_errors: list[str],
) -> HotspotSummary | None:
    if not fallback_cache_path or not topic:
        return None
    try:
        hotspots = load_hotspots_json(fallback_cache_path)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 - malformed fallback cache should not crash live flow.
        source_errors.append(f"last_good_cache: {exc}")
        return None
    for hotspot in hotspots:
        if hotspot.topic == topic or hotspot.name == topic:
            return hotspot
    return None


def _with_result_metadata(
    hotspots: list[HotspotSummary],
    *,
    provider_used: str,
    fallback_used: bool,
    source_errors: list[str],
    stale: bool,
    stale_age_hours: float | None,
) -> HotspotResults:
    errors = _dedupe_errors(source_errors)
    for hotspot in hotspots:
        _apply_summary_metadata(
            hotspot,
            provider_used=provider_used,
            fallback_used=fallback_used,
            source_errors=errors,
            stale=stale,
            stale_age_hours=stale_age_hours,
        )
    return HotspotResults(
        hotspots,
        provider_used=provider_used,
        fallback_used=fallback_used,
        source_errors=errors,
        stale=stale,
        stale_age_hours=stale_age_hours,
    )


def _apply_summary_metadata(
    summary: HotspotSummary,
    *,
    provider_used: str,
    fallback_used: bool,
    source_errors: list[str],
    stale: bool,
    stale_age_hours: float | None,
) -> None:
    summary.provider_used = provider_used
    summary.fallback_used = fallback_used
    summary.source_errors = _dedupe_errors(source_errors)
    summary.stale = stale
    summary.stale_age_hours = stale_age_hours


def _cache_stale_age_hours(path_like: str | Path | None) -> float | None:
    if not path_like:
        return None
    path = Path(path_like)
    if not path.exists():
        return None
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - modified).total_seconds() / 3600.0
    return round(max(age_hours, 0.0), 4)


def _record_provider_error(
    source_errors: list[str] | None,
    provider_label: str,
    method_name: str,
    exc: Exception,
) -> None:
    if source_errors is None:
        return
    label = provider_label or "provider"
    source_errors.append(f"{label}.{method_name}: {exc}")


def _provider_label(provider: object) -> str:
    if isinstance(provider, dict):
        return "mapping"
    return provider.__class__.__name__


def _dedupe_errors(source_errors: list[str]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for value in source_errors:
        text = _safe_text(value)
        if text and text not in seen:
            seen.add(text)
            errors.append(text)
    return errors


def _load_history_trends(history_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not history_path:
        return {}
    path = Path(history_path)
    if not path.is_file():
        return {}
    return load_board_heat_trends(path)


def _hotspot_sort_key(item: HotspotSummary) -> tuple[float, float, float, float, float]:
    trend = item.trend_score or 0.0
    persistence = item.persistence_score or 0.0
    cooling = item.cooling_score or 0.0
    score = item.heat_score + max(trend, 0.0) * 0.35 + persistence * 0.05 - cooling * 0.6
    change = item.change_pct if item.change_pct is not None else -999.0
    rank_bonus = -float(item.rank or 999999)
    return (score, item.heat_score, change, trend, rank_bonus)


def _board_row_rank_key(item: dict[str, Any]) -> tuple[float, float, float]:
    heat = _safe_float(item.get("heat_score"))
    change = _safe_float(item.get("change_pct"))
    rank = _safe_float(item.get("rank"))
    rank_bonus = -float(rank if rank is not None else 999999.0)
    heat_value = heat if heat is not None else 0.0
    change_value = change if change is not None else -999.0
    return (heat_value, change_value, rank_bonus)


def _call_provider_frame(
    provider: object,
    method_name: str,
    *,
    source_errors: list[str] | None = None,
    provider_label: str = "",
    **kwargs: Any,
) -> pd.DataFrame | None:
    method = getattr(provider, method_name, None)
    if method is None:
        return None
    try:
        frame = method(**kwargs) if kwargs else method()
    except TypeError:
        try:
            frame = method(kwargs.get("symbol")) if kwargs else method()
        except Exception as exc:  # noqa: BLE001 - provider runtime instability is degraded.
            _record_provider_error(source_errors, provider_label, method_name, exc)
            return None
    except Exception as exc:  # noqa: BLE001 - provider runtime instability is degraded.
        _record_provider_error(source_errors, provider_label, method_name, exc)
        return None
    return frame if isinstance(frame, pd.DataFrame) else None


def _mapping_provider_frame(provider: object, key: str) -> pd.DataFrame | None:
    if not isinstance(provider, dict):
        return None
    frame = provider.get(key)
    if isinstance(frame, pd.DataFrame):
        return frame
    if isinstance(frame, list):
        return pd.DataFrame(frame)
    return None


def _mapping_constituents_frame(provider: object, topic: str, *, source: str) -> pd.DataFrame | None:
    if not isinstance(provider, dict):
        return None
    for key in (f"{source}_constituents", "constituents"):
        value = provider.get(key)
        if isinstance(value, dict):
            frame = value.get(topic)
            if isinstance(frame, pd.DataFrame):
                return frame
            if isinstance(frame, list):
                return pd.DataFrame(frame)
    return None


def _timeline_matches_topic(item: dict[str, Any], topic: str) -> bool:
    values: list[str] = []
    for key in ("topic", "hotspot", "board"):
        text = _safe_text(item.get(key))
        if text:
            values.append(text)
    raw_topics = item.get("topics")
    if isinstance(raw_topics, list):
        values.extend(_safe_text(value) for value in raw_topics if _safe_text(value))
    elif _safe_text(raw_topics):
        values.extend(part.strip() for part in _safe_text(raw_topics).replace("，", ",").split(",") if part.strip())
    if values:
        return topic in values
    return topic in _safe_text(item.get("title"))


def _normalize_related_codes(value: object) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = _safe_text(value).replace("，", ",").replace("、", ",").split(",")
    codes: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        code = _normalize_code(raw)
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _row_value(row: dict[str, Any] | pd.Series, columns: list[str]) -> Any:
    for column in columns:
        if column in row:
            return row.get(column)
    return None


def _safe_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = _safe_text(value).lower()
    return text in {"1", "true", "yes", "y", "是", "涨停", "limit_up"}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
