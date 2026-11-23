import numpy as np
import pandas as pd
import pytest

from quantlab import optimize, scan
from quantlab.data import synthetic_prices
from quantlab.metrics import cagr, max_drawdown, round_trips, sharpe
from quantlab.strategies import SmaCrossover


def ohlc(n=900, seed=3):
    c = synthetic_prices(n_days=n, seed=seed)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1000})


def test_fast_segment_stats_match_the_metrics_module():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=600)
    r = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    held = pd.Series(np.where(rng.random(len(idx)) < 0.3, 0, 1.0), index=idx)
    r[held == 0] = 0.0
    s = optimize.segment_stats(r, held, "x")
    trips = round_trips(r, held)
    assert s["x_cagr"] == pytest.approx(cagr(r))
    assert s["x_sharpe"] == pytest.approx(sharpe(r))
    assert s["x_max_dd"] == pytest.approx(max_drawdown(r))
    assert s["x_win_rate"] == pytest.approx((trips["return"] > 0).mean())


def fake_load(ticker, tf):
    if ticker == "BAD":
        raise ValueError("Yahoo Finance has no data for 'BAD'.")
    if tf == "5m":
        return ohlc(100)                              # too short: reported, not fatal
    return ohlc(900, seed={"AAA": 1, "BBB": 2}[ticker])


def test_scan_markets_summaries_and_settings_across_markets():
    seen = []
    res = scan.scan(SmaCrossover, {"fast": (10, 30, 10), "slow": (50, 100, 50)},
                    [("AAA", "1d"), ("BBB", "1d"), ("BAD", "1d"), ("AAA", "5m"), ("AAA", "1d")],
                    load=fake_load, progress=lambda d, n, l: seen.append((d, n)))
    m = {(r["ticker"], r["tf"]): r for r in res["markets"]}
    assert len(res["markets"]) == 4                                 # the repeat is dropped
    assert "no data" in m[("BAD", "1d")]["error"]
    assert "only 100 bars" in m[("AAA", "5m")]["error"]
    good = m[("AAA", "1d")]
    assert good["error"] is None and set(good["settings"]) == {"fast", "slow"}
    # the market's settings are the in-sample winner, not the out-of-sample one
    t = optimize.sweep(ohlc(900, 1), SmaCrossover, {"fast": (10, 30, 10), "slow": (50, 100, 50)}).table
    best = t.sort_values("is_sharpe", ascending=False).iloc[0]
    assert good["settings"] == {"fast": int(best["fast"]), "slow": int(best["slow"])}
    assert good["best_oos_sharpe"] == pytest.approx(best["oos_sharpe"])
    assert res["settings"] and all(r["markets"] == 2 for r in res["settings"])
    assert seen[-1] == (seen[-1][1], seen[-1][1])                   # finished at 100%


def test_scan_limits_and_cancel():
    with pytest.raises(ValueError, match="too many"):
        scan.scan(SmaCrossover, {"fast": (10, 30, 10)}, [(f"T{i}", "1d") for i in range(scan.MAX_MARKETS + 1)])
    with pytest.raises(ValueError, match="backtests"):
        scan.scan(SmaCrossover, {"fast": (2, 400, 1), "slow": (5, 500, 50)},
                  [(chr(65 + i), "1d") for i in range(26)])                # 26 x 3,990 backtests

    def stop(d, n, label):
        if d > 0:
            raise scan.Cancelled()
    with pytest.raises(scan.Cancelled):
        scan.scan(SmaCrossover, {"fast": (10, 30, 10)}, [("AAA", "1d")], load=fake_load, progress=stop)
