import numpy as np
import pandas as pd
import pytest

from quantlab import chart
from quantlab import download as dl
from quantlab.metrics import bars_per_day, periods_per_year

NY = "America/New_York"


def payload(times, closes):
    return {"chart": {"error": None, "result": [{"timestamp": times, "indicators": {"quote": [{
        "open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
        "close": closes, "volume": [None] * len(closes)}]}}]}}


def test_parse_intraday_drops_live_snapshot_and_uses_ny_time():
    t0 = int(pd.Timestamp("2026-11-20 19:50", tz=NY).timestamp())
    df = dl.parse_intraday(payload([t0, t0 + 300, t0 + 599], [10.0, 11.0, 12.0]))   # 19:59:59 = snapshot
    assert list(df.index.strftime("%H:%M")) == ["19:50", "19:55"]
    assert str(df.index.tz) == NY and (df["Volume"] == 0).all()


def test_merge_keeps_old_bars_and_prefers_new():
    idx = pd.date_range("2026-01-05 09:30", periods=4, freq="5min", tz=NY)
    old = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    new = pd.DataFrame({"Close": [30.0, 40.0, 50.0]}, index=idx[2:].append(idx[-1:] + pd.Timedelta("5min")))
    m = dl.merge_bars(old, new)
    assert m["Close"].tolist() == [1.0, 2.0, 30.0, 40.0, 50.0]


def test_4h_bins_follow_the_new_york_clock_across_daylight_saving():
    # 1h bars around the November DST change (EDT -> EST on 2026-11-01)
    idx = pd.date_range("2026-10-30 04:00", "2026-11-03 19:00", freq="1h", tz=NY)
    idx = idx[(idx.hour >= 4) & (idx.hour < 20)]
    df = pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 1.0}, index=idx)
    out = dl.resample_bars(df, "4h")
    assert sorted(set(out.index.strftime("%H:%M"))) == ["04:00", "08:00", "12:00", "16:00"]
    assert (out["Volume"] == 4).all()


def test_clean_spikes_trims_bad_print_only():
    idx = pd.date_range("2026-01-05 04:00", periods=80, freq="5min", tz=NY)
    close = 100 + np.sin(np.arange(80) / 5)
    df = pd.DataFrame({"Open": close, "High": close + 0.1, "Low": close - 0.1, "Close": close,
                       "Volume": 0.0}, index=idx)
    df.iloc[60, df.columns.get_loc("Low")] = 90.0          # bad print: 10% wick
    df.iloc[70, df.columns.get_loc("Close")] = 103.0       # real move: body itself moved
    df.iloc[70, df.columns.get_loc("High")] = 103.1
    c = dl.clean_spikes(df)
    assert c.attrs["spikes_trimmed"] == 1
    assert c["Low"].iloc[60] == pytest.approx(min(df["Open"].iloc[60], df["Close"].iloc[60]))
    assert c["High"].iloc[70] == 103.1                     # genuine move untouched


def test_tkey_and_bars_per_year():
    assert chart.tkey(pd.Timestamp("2026-11-20")) == "2026-11-20"
    t = pd.Timestamp("2026-11-20 09:30", tz=NY)
    assert chart.tkey(t) == int(pd.Timestamp("2026-11-20 09:30").timestamp())   # wall clock as UTC
    idx = pd.date_range("2026-01-05 09:30", periods=78 * 3, freq="5min", tz=NY)
    idx = pd.DatetimeIndex([d for d in idx])
    assert bars_per_day(pd.bdate_range("2026-01-01", periods=10)) == 1
    r = pd.Series(0.0, index=pd.bdate_range("2026-01-01", periods=10))
    assert periods_per_year(r) == 252


def test_unknown_timeframe():
    with pytest.raises(ValueError, match="Unknown timeframe"):
        dl.check_timeframe("2m")
