import threading
import time

import pandas as pd

import alphasift.candidate_context as candidate_context
from alphasift.candidate_context import (
    classify_announcement_categories,
    classify_negative_events,
    collect_candidate_context,
    fetch_stock_quote_summary,
)


def test_collect_candidate_context_uses_requested_providers(monkeypatch):
    class FakeAkshare:
        @staticmethod
        def stock_news_em(symbol):
            return pd.DataFrame([
                {"发布时间": "2026-04-28", "文章来源": "测试", "新闻标题": f"{symbol} 获资金关注"},
            ])

        @staticmethod
        def stock_individual_fund_flow(stock, market):
            return pd.DataFrame([
                {"日期": "2026-04-28", "主力净流入-净额": "1000万", "主力净流入-净占比": "3.5%"},
            ])

    monkeypatch.setitem(__import__("sys").modules, "akshare", FakeAkshare)
    candidates = pd.DataFrame([{"code": 1.0, "name": "平安银行"}])

    rows, errors = collect_candidate_context(
        candidates,
        providers=["news", "fund_flow"],
    )

    assert errors == []
    assert rows[0]["code"] == "000001"
    assert "获资金关注" in rows[0]["news"]
    assert "主力净流入" in rows[0]["fund_flow"]
    assert rows[0]["source_count"] == 2
    assert rows[0]["source_confidence"] == 1.0
    assert rows[0]["source_weight_score"] == 1.0
    assert "新闻:" in rows[0]["context_summary"]
    assert isinstance(rows[0]["event_tags"], list)
    assert isinstance(rows[0]["negative_event_flags"], list)


def test_fetch_stock_quote_summary_parses_tencent_quote(monkeypatch):
    captured = {}
    parts = [""] * 50
    parts[1] = "平安银行"
    parts[3] = "11.24"
    parts[32] = "2.74"
    parts[33] = "11.25"
    parts[34] = "10.88"
    parts[37] = "226304"
    parts[38] = "1.05"
    parts[39] = "5.07"
    parts[44] = "2181.19"
    parts[45] = "2181.23"

    class FakeResponse:
        text = 'v_sz000001="' + "~".join(parts) + '";'

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("alphasift.candidate_context.requests.get", fake_get)

    summary = fetch_stock_quote_summary("1")

    assert captured["url"] == "https://qt.gtimg.cn/q=sz000001"
    assert captured["headers"]["User-Agent"] == "Mozilla/5.0"
    assert "现价=11.24" in summary
    assert "换手率=1.05" in summary
    assert "市盈率=5.07" in summary


def test_collect_candidate_context_supports_quote_provider(monkeypatch):
    monkeypatch.setattr(
        candidate_context,
        "fetch_stock_quote_summary",
        lambda code: "现价=11.24，市盈率=5.07，换手率=1.05",
    )
    candidates = pd.DataFrame([{"code": "000001", "name": "平安银行"}])

    rows, errors = collect_candidate_context(candidates, providers=["quote"])

    assert errors == []
    assert rows[0]["quote"] == "现价=11.24，市盈率=5.07，换手率=1.05"
    assert rows[0]["source_count"] == 1
    assert rows[0]["source_confidence"] == 1.0
    assert "行情估值:现价=11.24" in str(rows[0]["context_summary"])


def test_collect_candidate_context_concurrent_fetch_preserves_candidate_order(monkeypatch):
    barrier = threading.Barrier(2)

    def fake_news_summary(code, *, limit=3):
        barrier.wait(timeout=1)
        if code == "000001":
            time.sleep(0.05)
        return f"{code} 获资金关注"

    monkeypatch.setattr(candidate_context, "fetch_stock_news_summary", fake_news_summary)
    candidates = pd.DataFrame(
        [
            {"code": "000001", "name": "平安银行"},
            {"code": "000002", "name": "万科A"},
        ]
    )

    rows, errors = collect_candidate_context(candidates, providers=["news"])

    assert errors == []
    assert [row["code"] for row in rows] == ["000001", "000002"]
    assert rows[0]["news"] == "000001 获资金关注"
    assert rows[1]["news"] == "000002 获资金关注"


def test_collect_candidate_context_records_row_errors_without_aborting_other_candidates(monkeypatch):
    def fake_news_summary(code, *, limit=3):
        if code == "000001":
            raise ConnectionError("disconnect")
        return f"{code} 获资金关注"

    monkeypatch.setattr(candidate_context, "fetch_stock_news_summary", fake_news_summary)
    candidates = pd.DataFrame(
        [
            {"code": "000001", "name": "平安银行"},
            {"code": "000002", "name": "万科A"},
        ]
    )

    rows, errors = collect_candidate_context(candidates, providers=["news"])

    assert [row["code"] for row in rows] == ["000002"]
    assert errors == ["000001 news: disconnect"]


def test_collect_candidate_context_uses_cache(monkeypatch, tmp_path):
    calls = []
    calls_lock = threading.Lock()

    def fake_news_summary(code, *, limit=3):
        with calls_lock:
            calls.append(code)
        return f"{code} 首次抓取"

    monkeypatch.setattr(candidate_context, "fetch_stock_news_summary", fake_news_summary)
    candidates = pd.DataFrame(
        [
            {"code": "000001", "name": "平安银行"},
            {"code": "000002", "name": "万科A"},
        ]
    )

    first, _ = collect_candidate_context(
        candidates,
        providers=["news"],
        cache_dir=tmp_path,
    )
    second, _ = collect_candidate_context(
        candidates,
        providers=["news"],
        cache_dir=tmp_path,
    )

    assert sorted(calls) == ["000001", "000002"]
    assert second == first


def test_collect_candidate_context_enriches_legacy_cache(tmp_path):
    cache = tmp_path / "000001_news.json"
    cache.write_text(
        """
        {
          "cached_at": "2999-01-01T00:00:00",
          "row": {
            "code": "000001",
            "name": "平安银行",
            "news": "公司公告回购计划，收到监管问询函",
            "context_summary": "新闻:旧摘要"
          }
        }
        """,
        encoding="utf-8",
    )
    candidates = pd.DataFrame([{"code": "000001", "name": "平安银行"}])

    rows, errors = collect_candidate_context(
        candidates,
        providers=["news"],
        cache_dir=tmp_path,
    )

    assert errors == []
    assert "回购增持" in rows[0]["event_tags"]
    assert "监管" in rows[0]["negative_event_flags"]
    assert "负面风险:监管" in rows[0]["context_summary"]


def test_collect_candidate_context_partial_sources_have_partial_confidence(monkeypatch):
    class FakeAkshare:
        @staticmethod
        def stock_news_em(symbol):
            return pd.DataFrame([
                {"发布时间": "2026-04-28", "新闻标题": f"{symbol} 获资金关注"},
            ])

        @staticmethod
        def stock_individual_fund_flow(stock, market):
            return pd.DataFrame()

    monkeypatch.setitem(__import__("sys").modules, "akshare", FakeAkshare)
    candidates = pd.DataFrame([{"code": "000001", "name": "平安银行"}])

    rows, errors = collect_candidate_context(
        candidates,
        providers=["news", "fund_flow"],
    )

    assert errors == []
    assert rows[0]["source_count"] == 1
    assert rows[0]["source_confidence"] == 0.5
    assert rows[0]["source_weight_score"] == 0.4643


def test_collect_candidate_context_accepts_custom_source_weights(monkeypatch):
    class FakeAkshare:
        @staticmethod
        def stock_news_em(symbol):
            return pd.DataFrame([
                {"发布时间": "2026-04-28", "新闻标题": f"{symbol} 获资金关注"},
            ])

        @staticmethod
        def stock_zh_a_disclosure_report_cninfo(**kwargs):
            return pd.DataFrame()

    monkeypatch.setitem(__import__("sys").modules, "akshare", FakeAkshare)
    candidates = pd.DataFrame([{"code": "000001", "name": "平安银行"}])

    rows, errors = collect_candidate_context(
        candidates,
        providers=["news", "announcement"],
        source_weights={"news": 0.5, "announcement": 2.0},
    )

    assert errors == []
    assert rows[0]["source_confidence"] == 0.5
    assert rows[0]["source_weight_score"] == 0.2


def test_classify_negative_events_from_announcement_text():
    row = {
        "code": "000001",
        "announcement": "股东拟减持股份，公司收到监管问询函",
        "news": "公司公告回购计划",
    }

    flags = classify_negative_events(row)

    assert "减持" in flags
    assert "监管" in flags


def test_classify_announcement_categories_from_announcement_text():
    row = {
        "code": "000001",
        "announcement": "公司发布年度业绩预增公告，并披露股份回购方案",
    }

    categories = classify_announcement_categories(row)

    assert "业绩" in categories
    assert "回购增持" in categories
