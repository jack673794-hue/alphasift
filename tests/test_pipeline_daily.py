from pathlib import Path

import pandas as pd

from alphasift.config import Config
from alphasift.pipeline import screen


def test_pipeline_enriches_daily_features_for_daily_strategy(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "code": "000001",
                "name": "平安银行",
                "price": 10.0,
                "change_pct": -0.5,
                "amount": 200_000_000,
                "turnover_rate": 2.0,
                "volume_ratio": 1.2,
                "pe_ratio": 8.0,
                "pb_ratio": 0.8,
                "total_mv": 100_000_000_000,
            },
            {
                "code": "600000",
                "name": "浦发银行",
                "price": 11.0,
                "change_pct": -0.8,
                "amount": 190_000_000,
                "turnover_rate": 2.0,
                "volume_ratio": 1.1,
                "pe_ratio": 9.0,
                "pb_ratio": 0.9,
                "total_mv": 90_000_000_000,
            },
        ]
    )
    df.attrs["snapshot_source"] = "test"

    def fake_enrich(frame, **kwargs):
        enriched = frame.copy()
        for idx, row in enriched.iterrows():
            is_target = row["code"] == "000001"
            enriched.at[idx, "ma_bullish"] = is_target
            enriched.at[idx, "price_above_ma20"] = True
            enriched.at[idx, "signal_score"] = 72 if is_target else 80
            enriched.at[idx, "change_60d"] = 12 if is_target else 10
            enriched.at[idx, "macd_status"] = "bullish"
            enriched.at[idx, "rsi_status"] = "neutral"
            enriched.at[idx, "volume_ratio_20d"] = 1.0 if is_target else 1.8
            enriched.at[idx, "pullback_to_ma20_pct"] = 4 if is_target else 12
        return enriched

    monkeypatch.setattr("alphasift.pipeline.fetch_snapshot_with_fallback", lambda sources, **kwargs: df)
    monkeypatch.setattr("alphasift.pipeline.enrich_daily_features", fake_enrich)

    result = screen(
        "shrink_pullback",
        use_llm=False,
        config=Config(
            llm_api_key="",
            snapshot_source_priority=["test"],
            strategies_dir=Path("strategies"),
            risk_enabled=False,
        ),
    )

    assert result.daily_enriched is True
    assert result.after_filter_count == 1
    assert result.picks[0].code == "000001"
    assert result.picks[0].ma_bullish is True
    assert any("Daily K-line enrichment attempted 2 candidates" in item for item in result.degradation)


def test_pipeline_preserves_degradation_when_hard_filter_empty(monkeypatch):
    df = pd.DataFrame([
        {
            "code": "000001",
            "name": "平安银行",
            "price": 10.0,
            "change_pct": 0.0,
            "amount": 1,
            "total_mv": 1,
            "pe_ratio": 1000.0,
            "pb_ratio": 100.0,
        }
    ])
    df.attrs["snapshot_source"] = "test"
    df.attrs["source_errors"] = ["efinance failed"]
    monkeypatch.setattr("alphasift.pipeline.fetch_snapshot_with_fallback", lambda sources, **kwargs: df)

    result = screen(
        "dual_low",
        use_llm=False,
        post_analyzers=[],
        config=Config(
            llm_api_key="",
            snapshot_source_priority=["test"],
            strategies_dir=Path("strategies"),
            risk_enabled=False,
        ),
    )

    assert result.picks == []
    assert any("Snapshot source fallback: efinance failed" in item for item in result.degradation)
    assert "No candidates after hard filter" in result.degradation


def test_pipeline_passes_industry_provider_cache_config(monkeypatch, tmp_path):
    df = pd.DataFrame([
        {
            "code": "000001",
            "name": "骞冲畨閾惰",
            "price": 10.0,
            "change_pct": 0.0,
            "amount": 100_000_000,
            "turnover_rate": 2.0,
            "volume_ratio": 1.2,
            "pe_ratio": 8.0,
            "pb_ratio": 0.8,
            "total_mv": 100_000_000_000,
        }
    ])
    df.attrs["snapshot_source"] = "test"
    calls = []

    def fake_enrich(frame, **kwargs):
        calls.append(kwargs)
        return frame, []

    monkeypatch.setattr("alphasift.pipeline.fetch_snapshot_with_fallback", lambda sources, **kwargs: df)
    monkeypatch.setattr("alphasift.pipeline.enrich_industry_concepts", fake_enrich)

    cache_dir = tmp_path / "industry-cache"
    screen(
        "dual_low",
        use_llm=False,
        post_analyzers=[],
        config=Config(
            llm_api_key="",
            snapshot_source_priority=["test"],
            strategies_dir=Path("strategies"),
            industry_provider="akshare",
            industry_provider_cache_dir=cache_dir,
            industry_provider_cache_ttl_hours=7,
            risk_enabled=False,
        ),
    )

    assert calls == [{
        "map_files": [],
        "provider": "akshare",
        "max_boards": 80,
        "provider_cache_dir": cache_dir,
        "provider_cache_ttl_hours": 7,
    }]
