import pandas as pd

from quantlab import download as dl

# 2020-01-02 and 2020-01-03 14:30 UTC, plus a trailing all-null row.
PAYLOAD = {"chart": {"error": None, "result": [{
    "timestamp": [1577975400, 1578061800, 1578148200],
    "indicators": {
        "quote": [{"open": [10, 11, None], "high": [12, 12, None],
                   "low": [9, 10, None], "close": [11, 10, None],
                   "volume": [100, 200, None]}],
        "adjclose": [{"adjclose": [5.5, 5.0, None]}]}}]}}


def test_parse_chart():
    df = dl.parse_chart(PAYLOAD)
    assert list(df.columns) == dl.COLUMNS
    assert list(df.index) == list(pd.to_datetime(["2020-01-02", "2020-01-03"]))
    assert df["Adj Close"].tolist() == [5.5, 5.0]   # null row dropped


def test_download_caches(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dl, "fetch", lambda t: calls.append(t) or dl.parse_chart(PAYLOAD))
    dl.download("spy")
    dl.download("SPY")                     # second call reads the cache
    assert calls == ["spy"]
    prices = dl.load("SPY")
    assert prices.name == "SPY" and prices.tolist() == [5.5, 5.0]
    assert dl.load("SPY", adjusted=False).tolist() == [11, 10]


def test_download_many_keeps_going(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)

    def fake(t):
        if t == "BAD":
            raise ValueError("no such ticker")
        return dl.parse_chart(PAYLOAD)
    monkeypatch.setattr(dl, "fetch", fake)
    out = dl.download_many(["AAA", "BAD", "BBB"], pause=0)
    assert str(out["BAD"]).startswith("failed")
    assert (tmp_path / "BBB.csv").exists()


def test_parse_chart_handles_pre_1970_dates():
    payload = {"chart": {"error": None, "result": [{
        "timestamp": [-1325635200 + 14 * 3600],          # 1927-12-30, before the 1970 epoch
        "indicators": {"quote": [{"open": [17.66], "high": [17.66], "low": [17.66],
                                  "close": [17.66], "volume": [0]}],
                       "adjclose": [{"adjclose": [17.66]}]}}]}}
    df = dl.parse_chart(payload)
    assert str(df.index[0].date()) == "1927-12-30"


def test_popular_list_is_valid():
    from quantlab.universe import POPULAR, popular_tickers
    tickers = popular_tickers()
    assert len(tickers) == len(set(tickers)) >= 40
    for t in tickers:
        assert dl.clean_ticker(t) == t
