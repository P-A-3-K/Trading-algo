import numpy as np
import pandas as pd

from quantlab.backtest import run_backtest
from quantlab.data import synthetic_prices


def test_no_lookahead():
    """A strategy that 'knows' tomorrow must not profit from today's info."""
    prices = synthetic_prices(n_days=500, seed=1)
    # Position on day t = sign of day t's own return (only knowable at close).
    rets = prices.pct_change().fillna(0)
    pos = (rets > 0).astype(float)
    result = run_backtest(prices, pos, cost_bps=0)
    # If the engine leaked, this would compound to an absurd number.
    assert result.equity.iloc[-1] < 10


def test_costs_reduce_returns():
    prices = synthetic_prices(n_days=500, seed=2)
    pos = pd.Series(np.tile([1.0, 0.0], 250), index=prices.index)
    free = run_backtest(prices, pos, cost_bps=0).equity.iloc[-1]
    costly = run_backtest(prices, pos, cost_bps=10).equity.iloc[-1]
    assert costly < free


def test_buy_and_hold_matches_price():
    prices = synthetic_prices(n_days=300, seed=3)
    pos = pd.Series(1.0, index=prices.index)
    eq = run_backtest(prices, pos, cost_bps=0).equity
    assert np.isclose(eq.iloc[-1], prices.iloc[-1] / prices.iloc[0], rtol=1e-9)


def test_trades_list_sides_and_prices():
    from quantlab.backtest import trades
    idx = pd.date_range("2024-01-01", periods=7)
    pos = pd.Series([0, 1, 1, 0.5, -1, -1, 0], index=idx)
    px = pd.Series(np.arange(10.0, 17.0), index=idx)
    t = trades(pos, px)
    assert t["side"].tolist() == ["buy", "sell", "sell", "short", "cover"]
    assert t["price"].tolist() == [11.0, 13.0, 14.0, 14.0, 16.0]   # close of signal day
    assert t.iloc[3][["from", "to"]].tolist() == [0.0, -1.0]       # flip = sell + short


def test_trades_match_backtest_trade_count():
    from quantlab.backtest import trades
    from quantlab.strategies import sma_crossover
    prices = synthetic_prices(n_days=800, seed=4)
    pos = sma_crossover(prices, 10, 30)
    assert len(trades(pos, prices)) == run_backtest(prices, pos).trades
