import pandas as pd
import pytest

from alphasift.filter import SnapshotFieldMissingError, apply_hard_filters
from alphasift.models import HardFilterConfig


def test_apply_hard_filters_fails_when_required_snapshot_field_is_missing():
    df = pd.DataFrame(
        [
            {"name": "示例A", "price": 10.0, "amount": 100_000_000, "pe_ratio": 12.0},
        ]
    )

    with pytest.raises(SnapshotFieldMissingError):
        apply_hard_filters(df, HardFilterConfig(pb_max=2.0))


def test_apply_hard_filters_accepts_empty_frame_without_name_column():
    filtered = apply_hard_filters(pd.DataFrame(), HardFilterConfig())

    assert filtered.empty


def test_apply_hard_filters_drops_rows_with_unverifiable_numeric_values():
    df = pd.DataFrame(
        [
            {"name": "示例A", "price": 10.0, "amount": 100_000_000, "pb_ratio": None},
            {"name": "示例B", "price": 10.0, "amount": 100_000_000, "pb_ratio": 1.5},
        ]
    )

    filtered = apply_hard_filters(df, HardFilterConfig(pb_max=2.0))

    assert filtered["name"].tolist() == ["示例B"]


def test_apply_hard_filters_returns_empty_before_later_missing_fields():
    df = pd.DataFrame(
        [
            {"name": "示例A", "price": 10.0, "amount": 1},
        ]
    )

    filtered = apply_hard_filters(
        df,
        HardFilterConfig(amount_min=100_000_000, pb_max=2.0),
    )

    assert filtered.empty


def test_apply_hard_filters_fails_when_required_daily_features_are_missing():
    df = pd.DataFrame(
        [
            {"name": "示例A", "price": 10.0, "amount": 100_000_000},
        ]
    )

    with pytest.raises(SnapshotFieldMissingError, match="daily feature"):
        apply_hard_filters(df, HardFilterConfig(require_ma_bullish=True))


def test_apply_hard_filters_uses_daily_features_when_present():
    df = pd.DataFrame(
        [
            {"name": "示例A", "price": 10.0, "amount": 100_000_000, "ma_bullish": True, "signal_score": 70},
            {"name": "示例B", "price": 11.0, "amount": 100_000_000, "ma_bullish": False, "signal_score": 80},
        ]
    )

    result = apply_hard_filters(
        df,
        HardFilterConfig(require_ma_bullish=True, signal_score_min=65),
    )

    assert result["name"].tolist() == ["示例A"]


def test_apply_hard_filters_uses_daily_shape_features_when_present():
    df = pd.DataFrame(
        [
            {
                "name": "突破A",
                "price": 10.0,
                "amount": 100_000_000,
                "breakout_20d_pct": 0.8,
                "range_20d_pct": 18,
                "volume_ratio_20d": 1.8,
                "body_pct": 1.2,
                "pullback_to_ma20_pct": 4.0,
                "consolidation_days_20d": 10,
            },
            {
                "name": "伪突破B",
                "price": 11.0,
                "amount": 100_000_000,
                "breakout_20d_pct": -3.5,
                "range_20d_pct": 42,
                "volume_ratio_20d": 0.8,
                "body_pct": -0.5,
                "pullback_to_ma20_pct": 14.0,
                "consolidation_days_20d": 3,
            },
        ]
    )

    result = apply_hard_filters(
        df,
        HardFilterConfig(
            breakout_20d_pct_min=-1.0,
            range_20d_pct_max=30,
            volume_ratio_20d_min=1.2,
            body_pct_min=0,
            pullback_to_ma20_pct_max=8,
            consolidation_days_20d_min=8,
        ),
    )

    assert result["name"].tolist() == ["突破A"]


def test_apply_hard_filters_matches_one_pass_numeric_and_daily_filters():
    df = pd.DataFrame([
        {
            "name": "保留A",
            "price": 10.0,
            "amount": 200_000_000,
            "pb_ratio": 1.2,
            "change_pct": 2.0,
            "ma_bullish": True,
            "signal_score": 80,
            "macd_status": "bullish",
            "rsi_status": "neutral",
            "breakout_20d_pct": 0.6,
            "range_20d_pct": 20,
            "volume_ratio_20d": 1.5,
            "body_pct": 1.0,
            "pullback_to_ma20_pct": 3.0,
            "consolidation_days_20d": 10,
        },
        {
            "name": "金额不足B",
            "price": 9.0,
            "amount": 50_000_000,
            "pb_ratio": 1.1,
            "change_pct": 1.0,
            "ma_bullish": True,
            "signal_score": 90,
            "macd_status": "bullish",
            "rsi_status": "neutral",
            "breakout_20d_pct": 0.8,
            "range_20d_pct": 18,
            "volume_ratio_20d": 1.6,
            "body_pct": 0.8,
            "pullback_to_ma20_pct": 2.0,
            "consolidation_days_20d": 12,
        },
        {
            "name": "日线不符C",
            "price": 11.0,
            "amount": 220_000_000,
            "pb_ratio": 1.3,
            "change_pct": 2.5,
            "ma_bullish": False,
            "signal_score": 85,
            "macd_status": "bullish",
            "rsi_status": "neutral",
            "breakout_20d_pct": 1.0,
            "range_20d_pct": 22,
            "volume_ratio_20d": 1.8,
            "body_pct": 1.4,
            "pullback_to_ma20_pct": 2.5,
            "consolidation_days_20d": 11,
        },
        {
            "name": "形态不符D",
            "price": 12.0,
            "amount": 230_000_000,
            "pb_ratio": 1.4,
            "change_pct": 2.8,
            "ma_bullish": True,
            "signal_score": 82,
            "macd_status": "bearish",
            "rsi_status": "neutral",
            "breakout_20d_pct": -3.0,
            "range_20d_pct": 35,
            "volume_ratio_20d": 0.9,
            "body_pct": -0.2,
            "pullback_to_ma20_pct": 12.0,
            "consolidation_days_20d": 4,
        },
    ])
    filters = HardFilterConfig(
        amount_min=100_000_000,
        price_min=8,
        price_max=15,
        pb_max=2,
        change_pct_min=0,
        change_pct_max=5,
        require_ma_bullish=True,
        signal_score_min=75,
        macd_status_whitelist=["bullish"],
        rsi_status_whitelist=["neutral"],
        breakout_20d_pct_min=-1,
        range_20d_pct_max=30,
        volume_ratio_20d_min=1.2,
        body_pct_min=0,
        pullback_to_ma20_pct_max=8,
        consolidation_days_20d_min=8,
    )

    result = apply_hard_filters(df, filters)

    expected_mask = (
        (df["amount"] >= 100_000_000)
        & df["price"].between(8, 15)
        & (df["pb_ratio"] <= 2)
        & df["change_pct"].between(0, 5)
        & (df["ma_bullish"] == True)  # noqa: E712
        & (df["signal_score"] >= 75)
        & df["macd_status"].isin(["bullish"])
        & df["rsi_status"].isin(["neutral"])
        & (df["breakout_20d_pct"] >= -1)
        & (df["range_20d_pct"] <= 30)
        & (df["volume_ratio_20d"] >= 1.2)
        & (df["body_pct"] >= 0)
        & (df["pullback_to_ma20_pct"] <= 8)
        & (df["consolidation_days_20d"] >= 8)
    )
    assert result["name"].tolist() == df.loc[expected_mask, "name"].tolist()
