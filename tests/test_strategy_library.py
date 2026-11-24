import numpy as np
import pandas as pd
import pytest

from quantlab import strategies
from quantlab.data import synthetic_prices
from quantlab.strategy import check_no_lookahead


def session(day, opens, ranges):
    """One regular session of 5m bars from 9:30; each bar: (open, high, low, close)."""
    idx = pd.date_range(f"{day} 09:30", periods=len(opens), freq="5min", tz="America/New_York")
    return pd.DataFrame([{"open": o, "high": h, "low": l, "close": c, "volume": 1.0}
                         for o, h, l, c in zip(opens, *ranges)], index=idx)


def flat_day(day, n=78, px=100.0):
    return session(day, [px] * n, ([px + 0.1] * n, [px - 0.1] * n, [px] * n))


def test_orb_long_breakout_target_and_one_trade_per_day():
    d = flat_day("2024-03-04")
    d.iloc[7:] = [[100.4, 100.45, 100.35, 100.4, 1.0]] * (len(d) - 7)   # after the break: above the stop
    d.iloc[6] = [100.0, 100.3, 99.95, 100.3, 1.0]          # 10:00 bar breaks the 100.1 range high
    d.iloc[8] = [100.4, 100.9, 100.3, 100.8, 1.0]          # reaches the 2R target: 100.1 + 2 x 0.2 = 100.5
    d.iloc[20] = [100.0, 100.8, 99.0, 100.0, 1.0]           # later breaks: ignored (one trade a day)
    t = strategies.get("OpeningRangeBreakout")(rr=2.0).run(d).trades
    assert len(t) == 1
    x = t[0]
    assert x.side == 1 and x.entry_price == pytest.approx(100.1) and x.stop == pytest.approx(99.9)
    assert x.exit_price == pytest.approx(100.5) and x.exit_reason == "target"
    assert x.entry_date.strftime("%H:%M") == "10:00"


def test_orb_stop_first_flat_at_close_and_daily_bars():
    d = flat_day("2024-03-05")
    d.iloc[6] = [100.0, 100.3, 99.85, 100.0, 1.0]          # breaks up AND reaches the stop in the same bar
    t = strategies.get("OpeningRangeBreakout")().run(d).trades
    assert len(t) == 1 and t[0].exit_reason == "stop" and t[0].exit_price == pytest.approx(99.9)
    e = flat_day("2024-03-06")
    e.iloc[11:] = [[99.8, 99.85, 99.75, 99.8, 1.0]] * (len(e) - 11)    # after the break: below the stop
    e.iloc[10] = [100.0, 100.0, 99.8, 99.85, 1.0]            # short break, then drifts: closed at the close
    t = strategies.get("OpeningRangeBreakout")(rr=0).run(e).trades
    assert t[0].side == -1 and t[0].exit_reason == "end of day" and t[0].exit_date.strftime("%H:%M") == "15:55"
    assert strategies.get("OpeningRangeBreakout")(allow_short=False).run(e).trades == []
    c = synthetic_prices(n_days=300, seed=1)
    daily = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1.0})
    assert strategies.get("OpeningRangeBreakout")().run(daily).trades == []


def test_ts_momentum_only_changes_at_month_starts():
    c = synthetic_prices(n_days=1500, seed=2)
    bars = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0})
    pos = strategies.get("TimeSeriesMomentum")().run(bars).positions
    changes = pos.index[pos.diff().fillna(0).ne(0)]
    months = pd.Series(bars.index.to_period("M"))
    first = set(bars.index[(months != months.shift(1)).to_numpy()])
    assert len(changes) and set(changes) <= first
    with pytest.raises(ValueError):
        strategies.get("TimeSeriesMomentum")(lookback_months=1, skip_months=1)


def test_index_dip_rules():
    c = synthetic_prices(n_days=1500, seed=3)
    bars = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c * (1 + 0.009 * np.sin(np.arange(1500))),
                         "volume": 1.0})
    pos = strategies.get("IndexDip")(trend_len=0, max_hold=3).run(bars).positions
    runs = (pos.diff().fillna(pos) != 0).cumsum()[pos > 0]
    assert pos.max() == 1.0 and runs.value_counts().max() <= 3                # max_hold respected


@pytest.mark.parametrize("name, kw", [("IndexDip", {}), ("IndexDip", {"signal": "rsi2", "trend_len": 50}),
                                      ("TimeSeriesMomentum", {"lookback_months": 3}),
                                      ("OpeningRangeBreakout", {"range_minutes": 15})])
def test_no_lookahead(name, kw):
    if name == "OpeningRangeBreakout":
        rng = np.random.default_rng(5)
        days = [flat_day(d) for d in pd.bdate_range("2024-03-04", periods=6).strftime("%Y-%m-%d")]
        bars = pd.concat(days)
        bars["close"] = bars["close"] + rng.normal(0, 0.15, len(bars)).cumsum() * 0.1
        bars["high"] = bars[["open", "close"]].max(axis=1) + 0.05
        bars["low"] = bars[["open", "close"]].min(axis=1) - 0.05
    else:
        c = synthetic_prices(n_days=800, seed=4)
        bars = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1.0})
    assert check_no_lookahead(strategies.get(name)(**kw), bars, checks=25) == []
