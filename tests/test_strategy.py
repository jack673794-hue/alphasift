import os
from pathlib import Path

import pytest

import alphasift.strategy as strategy_module
from alphasift.strategy import list_strategies, load_all_strategies, load_strategy


def test_disabled_strategies_are_not_listed():
    strategies = load_all_strategies(Path("strategies"))

    assert "balanced_alpha" in strategies
    assert "capital_heat" in strategies
    assert "dual_low" in strategies
    assert "momentum_quality" in strategies
    assert "oversold_reversal" in strategies
    assert "quality_value" in strategies
    assert "shrink_pullback" in strategies
    assert "volume_breakout" in strategies


def test_list_strategies_returns_enabled_strategies_only():
    names = [item.name for item in list_strategies(Path("strategies"))]

    assert names == [
        "balanced_alpha",
        "capital_heat",
        "dual_low",
        "momentum_quality",
        "oversold_reversal",
        "quality_value",
        "shrink_pullback",
        "volume_breakout",
    ]


def test_load_all_strategies_allows_repo_local_custom_strategy(tmp_path):
    repo_dir = Path("strategies")
    for src in repo_dir.glob("*.yaml"):
        (tmp_path / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    (tmp_path / "custom_alpha.yaml").write_text(
        "\n".join([
            "name: custom_alpha",
            "display_name: 自定义策略",
            "description: demo",
            "screening:",
            "  enabled: true",
            "  market_scope: [cn]",
        ]),
        encoding="utf-8",
    )

    strategies = load_all_strategies(tmp_path)

    assert "custom_alpha" in strategies


def test_load_all_strategies_uses_cache_until_yaml_mtime_changes(tmp_path, monkeypatch):
    path = tmp_path / "cached_alpha.yaml"
    path.write_text(
        "\n".join([
            "name: cached_alpha",
            "display_name: 一版",
            "description: demo",
            "screening:",
            "  enabled: true",
            "  market_scope: [cn]",
        ]),
        encoding="utf-8",
    )
    calls = {"count": 0}
    original_load_strategy = strategy_module.load_strategy

    def counting_load_strategy(filepath):
        calls["count"] += 1
        return original_load_strategy(filepath)

    monkeypatch.setattr(strategy_module, "load_strategy", counting_load_strategy)

    first = strategy_module.load_all_strategies(tmp_path)
    second = strategy_module.load_all_strategies(tmp_path)

    assert calls["count"] == 1
    assert first["cached_alpha"].display_name == "一版"
    assert second["cached_alpha"].display_name == "一版"

    path.write_text(
        "\n".join([
            "name: cached_alpha",
            "display_name: 二版",
            "description: demo",
            "screening:",
            "  enabled: true",
            "  market_scope: [cn]",
        ]),
        encoding="utf-8",
    )
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    third = strategy_module.load_all_strategies(tmp_path)

    assert calls["count"] == 2
    assert third["cached_alpha"].display_name == "二版"


def test_load_strategy_rejects_unknown_hard_filter_key(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text(
        "\n".join([
            "name: broken",
            "display_name: 破损策略",
            "description: demo",
            "screening:",
            "  enabled: true",
            "  market_scope: [cn]",
            "  hard_filters:",
            "    pb_mx: 2.0",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown keys"):
        load_strategy(path)


def test_load_strategy_accepts_rule_profiles(tmp_path):
    path = tmp_path / "profiled.yaml"
    path.write_text(
        "\n".join([
            "name: profiled",
            "display_name: 规则配置策略",
            "description: demo",
            "screening:",
            "  enabled: true",
            "  market_scope: [cn]",
            "  scoring_profile:",
            "    momentum_chase_start_pct: 4.0",
            "  risk_profile:",
            "    chase_change_pct: 7.0",
            "  portfolio_profile:",
            "    max_same_bucket: 2",
            "    buckets:",
            "      周期: [钢铁, 煤炭]",
            "  scorecard_profile:",
            "    value_quality_bonus: 1.5",
            "  event_profile:",
            "    preferred_event_tags: [回购增持]",
            "    avoided_event_tags: [风险:监管]",
            "    source_weights:",
            "      announcement: 1.2",
        ]),
        encoding="utf-8",
    )

    strategy = load_strategy(path)

    assert strategy.screening.scoring_profile["momentum_chase_start_pct"] == 4.0
    assert strategy.screening.risk_profile["chase_change_pct"] == 7.0
    assert strategy.screening.portfolio_profile["max_same_bucket"] == 2
    assert strategy.screening.scorecard_profile["value_quality_bonus"] == 1.5
    assert strategy.screening.event_profile["preferred_event_tags"] == ["回购增持"]
    assert strategy.screening.event_profile["source_weights"]["announcement"] == 1.2


def test_load_strategy_rejects_unknown_profile_key(tmp_path):
    path = tmp_path / "broken_profile.yaml"
    path.write_text(
        "\n".join([
            "name: broken_profile",
            "display_name: 破损配置",
            "description: demo",
            "screening:",
            "  enabled: true",
            "  market_scope: [cn]",
            "  scoring_profile:",
            "    momentum_chase_start: 4.0",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown keys"):
        load_strategy(path)
