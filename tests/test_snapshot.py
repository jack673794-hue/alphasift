import json
import os

import pandas as pd
import pytest

from alphasift.snapshot import (
    _configure_tushare_client,
    _normalize,
    _prepare_tushare_snapshot,
    fetch_cn_snapshot,
    fetch_snapshot_with_fallback,
)


def test_normalize_efinance_maps_pb_ratio():
    df = pd.DataFrame(
        [
            {
                "股票代码": "000001",
                "股票名称": "平安银行",
                "最新价": "10.00",
                "涨跌幅": "1.23",
                "成交额": "123456789",
                "总市值": "1000000000",
                "流通市值": "800000000",
                "动态市盈率": "5.2",
                "市净率": "0.8",
                "量比": "1.1",
                "换手率": "2.5",
                "所属行业": "银行",
                "概念题材": "中特估,低估值",
            }
        ]
    )

    normalized = _normalize(df, source="efinance")

    assert normalized.loc[0, "pb_ratio"] == 0.8
    assert normalized.loc[0, "pe_ratio"] == 5.2
    assert normalized.loc[0, "industry"] == "银行"
    assert normalized.loc[0, "concepts"] == "中特估,低估值"


def test_prepare_tushare_snapshot_maps_fields_and_units():
    daily = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260430",
                "close": "10.00",
                "pct_chg": "1.23",
                "amount": "123456.789",
            }
        ]
    )
    daily_basic = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "turnover_rate": "2.5",
                "volume_ratio": "1.1",
                "pe": "5.2",
                "pb": "0.8",
                "total_mv": "100000",
                "circ_mv": "80000",
            }
        ]
    )
    stock_basic = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "symbol": "000001",
                "name": "平安银行",
                "industry": "银行",
            }
        ]
    )

    normalized = _prepare_tushare_snapshot(daily, daily_basic, stock_basic)

    assert normalized.loc[0, "code"] == "000001"
    assert normalized.loc[0, "name"] == "平安银行"
    assert normalized.loc[0, "price"] == 10.0
    assert normalized.loc[0, "change_pct"] == 1.23
    assert normalized.loc[0, "amount"] == pytest.approx(123456789)
    assert normalized.loc[0, "total_mv"] == pytest.approx(1000000000)
    assert normalized.loc[0, "circ_mv"] == pytest.approx(800000000)
    assert normalized.loc[0, "pe_ratio"] == 5.2
    assert normalized.loc[0, "pb_ratio"] == 0.8
    assert normalized.loc[0, "volume_ratio"] == 1.1
    assert normalized.loc[0, "turnover_rate"] == 2.5
    assert normalized.loc[0, "industry"] == "银行"
    assert normalized.attrs["snapshot_source"] == "tushare"


def test_fetch_tushare_requires_token(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("TUSHARE_API_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="TUSHARE_TOKEN"):
        fetch_cn_snapshot("tushare")


def test_configure_tushare_client_reads_http_url(monkeypatch):
    class FakePro:
        pass

    monkeypatch.setenv("TUSHARE_API_URL", "http://example.test")
    pro = FakePro()

    _configure_tushare_client(pro, token="token")

    assert pro._DataApi__token == "token"
    assert pro._DataApi__http_url == "http://example.test"


def test_fetch_snapshot_with_fallback_attaches_source_errors(monkeypatch):
    def fake_fetch(source):
        if source == "bad":
            raise RuntimeError("bad source")
        return pd.DataFrame([{"code": "000001", "name": "示例", "price": 10.0}])

    monkeypatch.setattr("alphasift.snapshot.fetch_cn_snapshot", fake_fetch)

    df = fetch_snapshot_with_fallback(["bad", "good"])

    assert df.attrs["source_errors"] == ["bad: bad source"]


def test_fetch_snapshot_with_fallback_skips_missing_required_columns(monkeypatch):
    def fake_fetch(source):
        if source == "missing_pb":
            return pd.DataFrame([{"code": "000001", "name": "示例", "price": 10.0}])
        return pd.DataFrame([{
            "code": "000001",
            "name": "示例",
            "price": 10.0,
            "pb_ratio": 0.8,
        }])

    monkeypatch.setattr("alphasift.snapshot.fetch_cn_snapshot", fake_fetch)

    df = fetch_snapshot_with_fallback(
        ["missing_pb", "complete"],
        required_columns=["price", "pb_ratio"],
    )

    assert df.attrs["source_errors"] == [
        "missing_pb: missing required columns pb_ratio"
    ]
    assert df.loc[0, "pb_ratio"] == 0.8


def test_fetch_snapshot_with_fallback_saves_last_good_cache_on_live_success(
    monkeypatch,
    tmp_path,
):
    cache_path = tmp_path / "snapshot.last_good.json"

    def fake_fetch(source):
        df = pd.DataFrame([{
            "code": "000001",
            "name": "示例",
            "price": 10.0,
            "pb_ratio": 0.8,
        }])
        df.attrs["snapshot_source"] = source
        return df

    monkeypatch.setattr("alphasift.snapshot.fetch_cn_snapshot", fake_fetch)

    df = fetch_snapshot_with_fallback(
        ["good"],
        required_columns=["price", "pb_ratio"],
        fallback_snapshot_path=cache_path,
    )

    assert df.attrs["snapshot_source"] == "good"
    assert df.attrs["fallback_used"] is False
    assert cache_path.is_file()
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["snapshot_source"] == "good"
    assert payload["metadata"]["row_count"] == 1

    monkeypatch.setattr(
        "alphasift.snapshot.fetch_cn_snapshot",
        lambda source: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    cached = fetch_snapshot_with_fallback(
        ["good"],
        required_columns=["price", "pb_ratio"],
        fallback_snapshot_path=cache_path,
    )

    assert cached.loc[0, "code"] == "000001"
    assert cached.loc[0, "pb_ratio"] == 0.8


def test_fetch_snapshot_with_fallback_uses_last_good_cache_after_all_sources_fail(
    monkeypatch,
    tmp_path,
):
    cache_path = tmp_path / "snapshot.last_good.json"
    live = pd.DataFrame([{
        "code": "000001",
        "name": "示例",
        "price": 10.0,
        "pb_ratio": 0.8,
    }])
    live.attrs["snapshot_source"] = "good"
    monkeypatch.setattr("alphasift.snapshot.fetch_cn_snapshot", lambda source: live)
    fetch_snapshot_with_fallback(
        ["good"],
        required_columns=["price", "pb_ratio"],
        fallback_snapshot_path=cache_path,
    )

    def fail(source):
        raise RuntimeError(f"{source} unavailable")

    monkeypatch.setattr("alphasift.snapshot.fetch_cn_snapshot", fail)

    cached = fetch_snapshot_with_fallback(
        ["efinance", "akshare_em"],
        required_columns=["price", "pb_ratio"],
        fallback_snapshot_path=cache_path,
    )

    assert cached.attrs["snapshot_source"] == "last_good_cache"
    assert cached.attrs["fallback_used"] is True
    assert cached.attrs["source_errors"] == [
        "efinance: efinance unavailable",
        "akshare_em: akshare_em unavailable",
    ]
    assert cached.loc[0, "code"] == "000001"


def test_snapshot_fallback_marks_stale_source_metadata(monkeypatch, tmp_path):
    cache_path = tmp_path / "snapshot.last_good.json"
    live = pd.DataFrame([{
        "code": "000001",
        "name": "示例",
        "price": 10.0,
    }])
    live.attrs["snapshot_source"] = "good"
    monkeypatch.setattr("alphasift.snapshot.fetch_cn_snapshot", lambda source: live)
    fetch_snapshot_with_fallback(["good"], fallback_snapshot_path=cache_path)

    old_mtime = cache_path.stat().st_mtime - (3 * 3600)
    cache_path.touch()
    os.utime(cache_path, (old_mtime, old_mtime))

    monkeypatch.setattr(
        "alphasift.snapshot.fetch_cn_snapshot",
        lambda source: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    cached = fetch_snapshot_with_fallback(
        ["efinance"],
        fallback_snapshot_path=cache_path,
    )

    assert cached.attrs["snapshot_source"] == "last_good_cache"
    assert cached.attrs["fallback_used"] is True
    assert cached.attrs["stale"] is True
    assert cached.attrs["stale_age_hours"] == pytest.approx(3.0, abs=0.1)
    assert cached.attrs["source_errors"] == ["efinance: offline"]


def test_fetch_snapshot_with_fallback_raises_all_errors(monkeypatch):
    monkeypatch.setattr(
        "alphasift.snapshot.fetch_cn_snapshot",
        lambda source: (_ for _ in ()).throw(RuntimeError(source)),
    )

    with pytest.raises(RuntimeError, match="a: a; b: b"):
        fetch_snapshot_with_fallback(["a", "b"])
