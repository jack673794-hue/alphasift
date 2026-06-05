# -*- coding: utf-8 -*-
"""L1 hard filter — apply strategy hard_filters to snapshot DataFrame."""

import logging
from dataclasses import replace

import pandas as pd

from alphasift.models import HardFilterConfig

logger = logging.getLogger(__name__)
_DAILY_FILTER_DEFAULTS = {
    "change_60d_min": None,
    "change_60d_max": None,
    "require_ma_bullish": False,
    "require_price_above_ma20": False,
    "signal_score_min": None,
    "macd_status_whitelist": None,
    "rsi_status_whitelist": None,
    "breakout_20d_pct_min": None,
    "breakout_20d_pct_max": None,
    "range_20d_pct_max": None,
    "volume_ratio_20d_min": None,
    "volume_ratio_20d_max": None,
    "body_pct_min": None,
    "body_pct_max": None,
    "pullback_to_ma20_pct_min": None,
    "pullback_to_ma20_pct_max": None,
    "consolidation_days_20d_min": None,
    "consolidation_days_20d_max": None,
}


class SnapshotFieldMissingError(ValueError):
    """Raised when a configured hard filter cannot be evaluated safely."""


def apply_hard_filters(df: pd.DataFrame, filters: HardFilterConfig) -> pd.DataFrame:
    """Filter snapshot DataFrame by hard conditions. Returns filtered copy."""
    result = df.copy()
    if result.empty:
        return result

    mask = pd.Series(True, index=result.index)

    if filters.exclude_st:
        name_col = _find_col(result, ["name", "股票名称", "名称"]) if mask.any() else None
        if not name_col:
            raise SnapshotFieldMissingError(
                "Missing required snapshot column for exclude_st filter: name"
            )
        mask &= ~result[name_col].str.contains(r"ST|退", na=False)

    # Numeric filters — each is optional
    mask = _filter_min(result, mask, ["amount", "成交额"], filters.amount_min)
    mask = _filter_min(result, mask, ["price", "最新价", "现价"], filters.price_min)
    mask = _filter_max(result, mask, ["price", "最新价", "现价"], filters.price_max)
    mask = _filter_min(result, mask, ["total_mv", "总市值"], filters.market_cap_min)
    mask = _filter_max(result, mask, ["total_mv", "总市值"], filters.market_cap_max)
    mask = _filter_min(result, mask, ["pe_ratio", "市盈率"], filters.pe_ttm_min)
    mask = _filter_max(result, mask, ["pe_ratio", "市盈率"], filters.pe_ttm_max)
    mask = _filter_min(result, mask, ["pb_ratio", "市净率"], filters.pb_min)
    mask = _filter_max(result, mask, ["pb_ratio", "市净率"], filters.pb_max)
    mask = _filter_min(result, mask, ["volume_ratio", "量比"], filters.volume_ratio_min)
    mask = _filter_min(result, mask, ["turnover_rate", "换手率"], filters.turnover_rate_min)
    mask = _filter_min(result, mask, ["change_pct", "涨跌幅"], filters.change_pct_min)
    mask = _filter_max(result, mask, ["change_pct", "涨跌幅"], filters.change_pct_max)

    mask = _filter_min(result, mask, ["change_60d"], filters.change_60d_min)
    mask = _filter_max(result, mask, ["change_60d"], filters.change_60d_max)
    mask = _filter_bool_true(result, mask, "ma_bullish", filters.require_ma_bullish)
    mask = _filter_bool_true(result, mask, "price_above_ma20", filters.require_price_above_ma20)
    mask = _filter_min(result, mask, ["signal_score"], filters.signal_score_min)
    mask = _filter_in(result, mask, "macd_status", filters.macd_status_whitelist)
    mask = _filter_in(result, mask, "rsi_status", filters.rsi_status_whitelist)
    mask = _filter_min(result, mask, ["breakout_20d_pct"], filters.breakout_20d_pct_min)
    mask = _filter_max(result, mask, ["breakout_20d_pct"], filters.breakout_20d_pct_max)
    mask = _filter_max(result, mask, ["range_20d_pct"], filters.range_20d_pct_max)
    mask = _filter_min(result, mask, ["volume_ratio_20d"], filters.volume_ratio_20d_min)
    mask = _filter_max(result, mask, ["volume_ratio_20d"], filters.volume_ratio_20d_max)
    mask = _filter_min(result, mask, ["body_pct"], filters.body_pct_min)
    mask = _filter_max(result, mask, ["body_pct"], filters.body_pct_max)
    mask = _filter_min(result, mask, ["pullback_to_ma20_pct"], filters.pullback_to_ma20_pct_min)
    mask = _filter_max(result, mask, ["pullback_to_ma20_pct"], filters.pullback_to_ma20_pct_max)
    mask = _filter_min(result, mask, ["consolidation_days_20d"], filters.consolidation_days_20d_min)
    mask = _filter_max(result, mask, ["consolidation_days_20d"], filters.consolidation_days_20d_max)

    return result.loc[mask].copy()


def requires_daily_features(filters: HardFilterConfig) -> bool:
    """Return whether a hard-filter config needs daily K-line features."""
    return any([
        filters.change_60d_min is not None,
        filters.change_60d_max is not None,
        filters.require_ma_bullish,
        filters.require_price_above_ma20,
        filters.signal_score_min is not None,
        bool(filters.macd_status_whitelist),
        bool(filters.rsi_status_whitelist),
        filters.breakout_20d_pct_min is not None,
        filters.breakout_20d_pct_max is not None,
        filters.range_20d_pct_max is not None,
        filters.volume_ratio_20d_min is not None,
        filters.volume_ratio_20d_max is not None,
        filters.body_pct_min is not None,
        filters.body_pct_max is not None,
        filters.pullback_to_ma20_pct_min is not None,
        filters.pullback_to_ma20_pct_max is not None,
        filters.consolidation_days_20d_min is not None,
        filters.consolidation_days_20d_max is not None,
    ])


def without_daily_filters(filters: HardFilterConfig) -> HardFilterConfig:
    """Return a copy with daily K-line filters disabled."""
    return replace(filters, **_DAILY_FILTER_DEFAULTS)


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _filter_min(
    df: pd.DataFrame,
    mask: pd.Series,
    col_names: list[str],
    value: float | None,
) -> pd.Series:
    if value is None:
        return mask
    if not mask.any():
        return mask
    col = _find_col(df, col_names)
    if not col:
        raise SnapshotFieldMissingError(
            f"Missing required snapshot column for min filter {col_names}: "
            f"configured value={value}"
        )
    series = pd.to_numeric(df[col], errors="coerce")
    return mask & series.ge(value) & series.notna()


def _filter_max(
    df: pd.DataFrame,
    mask: pd.Series,
    col_names: list[str],
    value: float | None,
) -> pd.Series:
    if value is None:
        return mask
    if not mask.any():
        return mask
    col = _find_col(df, col_names)
    if not col:
        raise SnapshotFieldMissingError(
            f"Missing required snapshot column for max filter {col_names}: "
            f"configured value={value}"
        )
    series = pd.to_numeric(df[col], errors="coerce")
    return mask & series.le(value) & series.notna()


def _filter_bool_true(
    df: pd.DataFrame,
    mask: pd.Series,
    col_name: str,
    enabled: bool,
) -> pd.Series:
    if not enabled:
        return mask
    if not mask.any():
        return mask
    if col_name not in df.columns:
        raise SnapshotFieldMissingError(
            f"Missing required daily feature column for bool filter: {col_name}"
        )
    return mask & (df[col_name] == True)  # noqa: E712


def _filter_in(
    df: pd.DataFrame,
    mask: pd.Series,
    col_name: str,
    allowed: list[str] | None,
) -> pd.Series:
    if not allowed:
        return mask
    if not mask.any():
        return mask
    if col_name not in df.columns:
        raise SnapshotFieldMissingError(
            f"Missing required daily feature column for whitelist filter: {col_name}"
        )
    allowed_set = {str(item) for item in allowed}
    return mask & df[col_name].astype(str).isin(allowed_set)
