import json

import pandas as pd

from quantlab import chart

DF = pd.DataFrame(
    {"Open": [10.0, 11.0], "High": [12.0, 12.0], "Low": [9.0, 10.0],
     "Close": [11.0, 10.0], "Adj Close": [5.5, 5.0], "Volume": [100, 200]},
    index=pd.to_datetime(["2024-01-02", "2024-01-03"]))


def test_adjusted_ohlc_scales_whole_bar():
    adj = chart.adjusted_ohlc(DF)
    assert adj["Close"].tolist() == [5.5, 5.0]          # equals Adj Close
    assert adj["Open"].tolist() == [5.0, 5.5]           # same factor (0.5)
    assert adj["High"].iloc[0] == 6.0 and adj["Volume"].iloc[1] == 200


def test_candles_format():
    rows = chart.candles(DF)
    assert rows[0] == {"time": "2024-01-02", "open": 5.0, "high": 6.0,
                       "low": 4.5, "close": 5.5, "volume": 100}
    assert chart.candles(DF, adjusted=False)[0]["close"] == 11.0


def test_render_fills_template():
    html = chart.render("spy", DF)
    assert "__TICKER__" not in html and "/*__DATA__*/" not in html
    assert "<title>SPY Candles</title>" in html
    assert "LightweightCharts" in html                  # library inlined
    data = json.loads(html.split("const DATA = ")[1].split(";\n")[0])
    assert data["ticker"] == "SPY" and len(data["bars"]) == 2


def test_plots_in_payload():
    from quantlab.strategy import Plot
    values = pd.Series([float("nan"), 7.25], index=DF.index)
    html = chart.render("spy", DF, plots=[Plot("SMA 2", values), Plot("Z", values, "lower")],
                        strategy="SmaCrossover(fast=2)")
    data = json.loads(html.split("const DATA = ")[1].split(";\n")[0])
    assert data["strategy"] == "SmaCrossover(fast=2)"
    assert data["plots"][0] == {"name": "SMA 2", "pane": "price", "color": None,
                                "points": [{"time": "2024-01-03", "value": 7.25}]}  # NaN skipped
    assert data["plots"][1]["pane"] == "lower"


def test_cli_args():
    from quantlab.strategies import SmaCrossover
    ticker, s = chart._parse_args(["QQQ", "smacrossover", "fast=20"])
    assert ticker == "QQQ" and isinstance(s, SmaCrossover) and s.fast == 20
