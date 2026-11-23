import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import run_backtest, simulate, simulate_trades
from quantlab.costs import PRESETS, CostModel, flat
from quantlab.data import synthetic_prices
from quantlab.strategy import Trade

NY = "America/New_York"


def daily_bars(n=300, seed=1, volume=1e6):
    c = synthetic_prices(n_days=n, seed=seed)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": volume})


def test_flat_preset_matches_old_cost_bps():
    b = daily_bars()
    pos = pd.Series(np.tile([1.0, 1.0, 0.0, 0.0], len(b) // 4), index=b.index)
    old = run_backtest(b["close"], pos, cost_bps=5)
    new = run_backtest(b["close"], pos, costs=flat(5).prepare(b))
    assert np.allclose(old.returns, new.returns)
    assert np.isclose(old.costs.sum(), new.costs.sum())


def test_slippage_scales_with_bar_range_and_spread_with_hours():
    b = daily_bars()
    calm, wild = b.copy(), b.copy()
    wild["high"], wild["low"] = b["close"] * 1.05, b["close"] * 0.95
    m = CostModel(impact=0)
    assert m.prepare(wild).linear.mean() > m.prepare(calm).linear.mean() * 4
    # intraday: wider spread outside 9:30-16:00
    idx = pd.date_range("2026-03-02 09:00", "2026-03-02 17:00", freq="5min", tz=NY)
    ib = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}, index=idx)
    lin = CostModel(spread_bps=1, extended_spread_bps=10, slippage_range=0, impact=0).prepare(ib).linear
    t = idx.strftime("%H:%M")
    assert np.allclose(lin[(t >= "09:30") & (t < "16:00")], 1e-4)
    assert np.allclose(lin[(t < "09:30") | (t >= "16:00")], 10e-4)


def test_impact_grows_with_order_size_and_account():
    thin = daily_bars(volume=1_000)          # ~$100k a day traded
    m_small, m_big = CostModel(account=10_000), CostModel(account=1_000_000)
    k_small, k_big = m_small.prepare(thin).impact[-1], m_big.prepare(thin).impact[-1]
    assert k_big == pytest.approx(k_small * 10, rel=1e-6)          # sqrt(100x account)
    p = m_small.prepare(thin)
    assert p(len(thin) - 1, 1.0) - p(len(thin) - 1, 0.5) > 0.5 * (p(len(thin) - 1, 0.5))  # convex in size


def test_costs_charged_on_execution_bar():
    idx = pd.date_range("2026-03-02 15:50", periods=4, freq="5min", tz=NY)   # 15:50, 15:55, 16:00, 16:05
    b = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}, index=idx)
    m = CostModel(spread_bps=1, extended_spread_bps=10, slippage_range=0, impact=0)
    pos = pd.Series([0.0, 1.0, 1.0, 1.0], index=idx)        # decided at 15:55's close
    bt = run_backtest(b["close"], pos, costs=m.prepare(b))
    assert bt.costs.iloc[2] == pytest.approx(1e-4)            # regular-hours price at 15:55, not 16:00's
    tr = simulate_trades(b, [Trade(idx[2], 100.0, 1, 1.0, idx[3], 100.0)], costs=m.prepare(b))
    assert tr.costs.sum() == pytest.approx(2 * 10e-4)         # both fills after hours


def test_cost_summary_and_presets():
    from quantlab.chart import cost_summary
    b = daily_bars()
    pos = pd.Series(1.0, index=b.index)
    bt = run_backtest(b["close"], pos, costs=flat(10).prepare(b))
    s = cost_summary(bt)
    assert s["costs_paid"] == pytest.approx(10_000 * 0.001)  # one $10k buy at 10 bp
    from quantlab.metrics import cagr
    assert s["cagr_gross"] > cagr(bt.returns)              # before costs beats after costs
    assert set(PRESETS) == {"realistic", "flat5", "none", "expensive"}


def test_costs_from_address():
    from quantlab.app import costs_of
    assert costs_of({})[0] == "realistic"
    name, m = costs_of({"cost": "custom", "c_spread": "2", "c_account": "50000"})
    assert name == "custom" and m.spread_bps == 2 and m.account == 50_000
    with pytest.raises(ValueError):
        costs_of({"cost": "cheap"})
    with pytest.raises(ValueError):
        costs_of({"cost": "custom", "c_spread": "-1"})
