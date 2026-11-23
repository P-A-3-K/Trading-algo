import json

import numpy as np
import pandas as pd
import pytest

from quantlab import optimize, prop


def intraday(day_returns, bars=4):
    """Bar returns: each day's move split over `bars` bars."""
    idx, vals = [], []
    for d, moves in zip(pd.bdate_range("2024-01-02", periods=len(day_returns)), day_returns):
        for i, m in enumerate(moves if isinstance(moves, (list, tuple)) else [moves] + [0.0] * (bars - 1)):
            idx.append(pd.Timestamp(d) + pd.Timedelta(hours=9 + i)); vals.append(m)
    return pd.Series(vals, index=pd.DatetimeIndex(idx))


def test_daily_parts_sees_the_intraday_low():
    r = intraday([[0.0, -0.04, 0.05, 0.0], [0.01, 0.0, 0.0, 0.0]])
    ret, low = prop.daily_parts(r, 1.0)
    assert ret[0] == pytest.approx(0.96 * 1.05 - 1) and low[0] == pytest.approx(-0.04)
    assert low[1] == 0.0
    ret2, low2 = prop.daily_parts(r, 2.0)                        # size doubles every bar's move
    assert low2[0] == pytest.approx(-0.08)


def run(days, rules, target):
    ret = np.array([[d[0] for d in days]]); low = np.array([[d[1] for d in days]])
    o, t = prop.run_phase(ret, low, rules, target)
    return int(o[0]), int(t[0])


def test_phase_rules():
    r = prop.Rules()
    assert run([(0.04, 0), (0.04, 0), (0.03, 0)], r, 0.10) == (1, 2)          # 1.04*1.04*1.03 >= 1.10
    assert run([(-0.04, -0.04), (-0.04, -0.04), (-0.03, -0.03)], r, 0.10) == (-1, 2)
    assert run([(0.0, -0.06)], r, 0.10) == (-1, 0)                            # one bad day: daily loss
    assert run([(0.01, 0)] * 3, r, 0.10) == (0, 3)                            # neither within the time
    tr = prop.Rules(max_loss=0.06, trailing=True, daily_loss=0.10)
    # up 5%, then down 6% from the high: fails with a trailing limit, not with a static one
    days = [(0.05, 0), (-0.058, -0.058), (0.0, 0.0)]
    assert run(days, tr, 0.10)[0] == -1 and run(days, prop.Rules(max_loss=0.06, daily_loss=0.10), 0.10)[0] == 0
    assert run([(0.0, 0.0)] * 3 + [(0.11, 0)], prop.Rules(min_days=5), 0.10) == (0, 4)


def test_check_extremes_and_safe_size():
    calm = intraday([0.002] * 300)                                    # steady gains: always passes
    out = prop.check(calm, prop.Rules(target2=0), sizes=[1.0], sims=200)
    row = next(r for r in out["rows"] if r["size"] == 1.0)
    assert row["pass"] == 1.0 and row["fail"] == 0.0
    rng = np.random.default_rng(1)
    wild = intraday(list(rng.normal(0, 0.03, 400)))
    s = prop.safe_size(wild, prop.Rules())
    dd, day = prop.history_risk(wild, s)
    assert -dd <= 0.05 + 1e-9 and -day <= 0.025 + 1e-9                  # half of the limits
    dd2, day2 = prop.history_risk(wild, s + 0.02)
    assert -dd2 > 0.05 or -day2 > 0.025                                 # and not more
    with pytest.raises(ValueError, match="too few"):
        prop.check(intraday([0.01] * 10))


def test_prop_return_metric_in_sweeps():
    r = np.array([0.01, -0.02, 0.015, 0.005] * 100)
    for dd in (-0.20, -0.02):                    # 20% drawdown: scaled to 1/4; 2%: scaled up 2.5x
        size = 0.05 / -dd
        assert optimize.prop_return(r, dd, 252) == pytest.approx(np.expm1(252 * np.mean(np.log1p(size * r))))
    assert np.isnan(optimize.prop_return(r, 0.0, 252))
    from quantlab.data import synthetic_prices
    from quantlab.strategies import SmaCrossover
    c = synthetic_prices(n_days=900, seed=2)
    bars = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0})
    t = optimize.sweep(bars, SmaCrossover, {"fast": (10, 30, 10), "slow": (60, 60, 1)}).table
    assert {"is_prop", "oos_prop"} <= set(t.columns)
