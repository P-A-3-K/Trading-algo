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


def intraday_bars(days=3, seed=0):
    idx = []
    for d in pd.bdate_range("2026-03-02", periods=days):
        idx += list(pd.date_range(f"{d.date()} 04:00", f"{d.date()} 19:55", freq="5min", tz=NY))
    idx = pd.DatetimeIndex(idx)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx))))
    return pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999,
                         "close": close, "volume": 1.0}, index=idx)


def test_regular_hours_and_hold_masks():
    from quantlab.strategy import hold_allowed, regular_hours
    b = intraday_bars(1)
    rth, hold = regular_hours(b.index), hold_allowed(b.index)
    times = b.index.strftime("%H:%M")
    assert times[rth][0] == "09:30" and times[rth][-1] == "15:55"
    assert times[hold][0] == "09:25" and times[hold][-1] == "15:50"   # carry INTO 09:30..15:55 only


def test_day_trading_position_strategy_is_flat_outside_regular_hours():
    from quantlab.backtest import simulate
    from quantlab.strategies import BuyAndHold
    b = intraday_bars(3)
    bt = simulate(b, BuyAndHold(flat_at_close=True).run(b), cost_bps=0)
    held = bt.positions                         # position held during each bar
    t = held.index.strftime("%H:%M")
    assert (held[(t < "09:30") | (t >= "16:00")] == 0).all()
    assert (held[(t >= "09:30") & (t < "16:00")] == 1).all()
    assert repr(BuyAndHold(flat_at_close=True)) == "BuyAndHold(flat_at_close=True)"
    assert repr(BuyAndHold()) == "BuyAndHold()"


def test_day_trading_trade_strategy_closes_at_end_of_day():
    from quantlab.strategy import Strategy, check_no_lookahead

    class Hold(Strategy):
        name = "enter once and hold"
        def compute(self, bars):
            i = 20   # 05:40, pre-market
            self.trade(bars.index[i], bars.open.iloc[i], 1)

    b = intraday_bars(2)
    assert Hold(flat_at_close=True).run(b).trades == []          # pre-market entry dropped

    class HoldRth(Hold):
        def compute(self, bars):
            i = int(np.argmax(bars.index.strftime("%H:%M") == "10:00"))
            self.trade(bars.index[i], bars.open.iloc[i], 1)

    (t,) = HoldRth(flat_at_close=True).run(b).trades
    assert t.exit_reason == "end of day" and t.exit_date.strftime("%H:%M") == "15:55"
    assert HoldRth().run(b).trades[0].exit_date is None           # no day trading: still open
    assert check_no_lookahead(HoldRth(flat_at_close=True), b) == []


def test_bool_setting_parsing():
    from quantlab.strategies import SmaCrossover
    assert SmaCrossover(flat_at_close="1").flat_at_close is True
    assert SmaCrossover(flat_at_close="no").flat_at_close is False
    with pytest.raises(ValueError, match="yes or no"):
        SmaCrossover(flat_at_close="maybe")
    info = SmaCrossover.info()["params"]["flat_at_close"]
    assert info["type"] == "bool" and info["intraday_only"] is True
