import json
import sys

import pandas as pd

from alphasift.cli import main
from alphasift.config import Config
from alphasift.doctor import doctor_data_sources
from alphasift.snapshot import (
    _SOURCE_HEALTH,
    _record_source_failure,
    snapshot_source_health_snapshot,
)


def test_snapshot_source_health_snapshot_reports_disabled_failures(monkeypatch):
    _SOURCE_HEALTH.clear()
    monkeypatch.setattr("alphasift.snapshot.time.monotonic", lambda: 100.0)

    _record_source_failure("sina")
    _record_source_failure("sina")
    _record_source_failure("sina")

    health = snapshot_source_health_snapshot(["sina", "efinance"])

    assert health["sina"]["failures"] == 3.0
    assert health["sina"]["total_failures"] == 3.0
    assert health["sina"]["disabled"] is True
    assert health["efinance"]["disabled"] is False
    _SOURCE_HEALTH.clear()


def test_doctor_data_sources_aggregates_snapshot_and_daily(monkeypatch, tmp_path):
    config = Config(
        snapshot_source_priority=["sina", "efinance"],
        daily_source="auto",
        fallback_snapshot_path=tmp_path / "snapshot.last_good.json",
        daily_history_cache_dir=tmp_path / "daily_history",
    )

    def fake_snapshot(sources, **kwargs):
        assert sources == ["sina", "efinance"]
        df = pd.DataFrame([{"code": "000001", "name": "平安银行", "price": 10.0}])
        df.attrs["snapshot_source"] = "sina"
        df.attrs["fallback_used"] = False
        df.attrs["stale"] = False
        df.attrs["source_errors"] = []
        return df

    def fake_daily(code, **kwargs):
        assert code == "000001"
        assert kwargs["source"] == "auto"
        df = pd.DataFrame([{"date": "2026-01-01", "close": 10.0}])
        df.attrs["daily_source"] = "tencent"
        df.attrs["source_errors"] = ["tushare after 1 attempts: no token"]
        return df

    monkeypatch.setattr("alphasift.doctor.fetch_snapshot_with_fallback", fake_snapshot)
    monkeypatch.setattr("alphasift.doctor.fetch_daily_history", fake_daily)

    result = doctor_data_sources(config)
    payload = result.to_dict()

    assert payload["status"] == "ok"
    assert payload["snapshot"]["source"] == "sina"
    assert payload["snapshot"]["rows"] == 1
    assert payload["daily"]["source"] == "tencent"
    assert payload["daily"]["fallback_used"] is True
    assert "source_health" in payload
    assert "TUSHARE_TOKEN" not in json.dumps(payload, ensure_ascii=False)


def test_cli_doctor_data_sources_no_live_json(monkeypatch, tmp_path, capsys):
    output = tmp_path / "doctor.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alphasift",
            "doctor",
            "data-sources",
            "--no-live",
            "--snapshot-source",
            "sina,efinance",
            "--daily-source",
            "auto",
            "--output",
            str(output),
            "--json",
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "skipped"
    assert payload["snapshot"]["status"] == "skipped"
    assert payload["daily"]["status"] == "skipped"
    assert payload["config"]["snapshot_source_priority"] == ["sina", "efinance"]
    assert saved["source_health"] == payload["source_health"]
