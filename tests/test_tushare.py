import os

import pytest

from alphasift.config import Config


# Pytest captures print output by default; run with `pytest tests/test_tushare.py -q -s`
# to see the smoke result table.
def test_tushare_daily_smoke_from_env():
    """Manual smoke for local Tushare connectivity.

    This test reads Tushare settings through Config.from_env(), so local .env
    values such as TUSHARE_TOKEN and TUSHARE_API_URL are honored without
    hardcoding credentials in the repository.
    """
    Config.from_env()
    token = os.getenv("TUSHARE_TOKEN", "").strip() or os.getenv("TUSHARE_API_TOKEN", "").strip()
    if not token:
        pytest.skip("TUSHARE_TOKEN/TUSHARE_API_TOKEN is not configured")

    import tushare as ts

    pro = ts.pro_api(token)
    pro._DataApi__token = token
    http_url = (
        os.getenv("TUSHARE_API_URL", "").strip()
        or os.getenv("TUSHARE_HTTP_URL", "").strip()
        or "http://api.waditu.com"
    )
    pro._DataApi__http_url = http_url

    trade_date = os.getenv("TUSHARE_TRADE_DATE", "").strip() or "20260608"
    df = pro.daily(
        trade_date=trade_date,
        fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
    )

    print(f"Tushare URL: {http_url}")
    print(f"Trade date: {trade_date}")
    print(f"Rows: {0 if df is None else len(df)}")
    if df is not None and not df.empty:
        print(df.head().to_string(index=False))

    assert df is not None
    assert not df.empty
    assert {"ts_code", "trade_date", "close"}.issubset(df.columns)
