import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import held_positions, simulate, simulate_trades, trade_fills
from quantlab.strategy import Strategy, Trade

IDX = pd.bdate_range("2024-01-01", periods=5)
BARS = pd.DataFrame({"open": [100, 101, 103, 106, 104], "high": [101, 104, 107, 108, 105],
                     "low": [99, 100, 102, 103, 100], "close": [100, 103, 106, 104, 101],
                     "volume": 1}, index=IDX, dtype=float)


def test_long_trade_returns_compound_to_exit_over_entry():
    t = Trade(IDX[1], 101.0, side=1, size=1.0, exit_date=IDX[3], exit_price=107.0)
    bt = simulate_trades(BARS, [t], cost_bps=0)
    assert np.isclose(bt.equity.iloc[-1], 107 / 101)            # exactly the trade's return
    assert bt.returns.iloc[1] == pytest.approx(103 / 101 - 1)    # entry day: close vs entry price
    assert bt.returns.iloc[3] == pytest.approx(107 / 106 - 1)    # exit day: exit vs previous close
    assert bt.positions.tolist() == [0, 1, 1, 0, 0] and bt.trades == 2


def test_short_half_size_same_day_and_costs():
    t = Trade(IDX[2], 103.0, side=-1, size=0.5, exit_date=IDX[2], exit_price=107.0)
    bt = simulate_trades(BARS, [t], cost_bps=10)
    assert bt.returns.iloc[2] == pytest.approx(-0.5 * (107 / 103 - 1) - 2 * 0.5 * 0.001)
    assert bt.positions.abs().sum() == 0                          # never held at a close


def test_open_trade_is_valued_at_last_close():
    t = Trade(IDX[3], 106.0, side=1, size=0.25)
    bt = simulate_trades(BARS, [t], cost_bps=0)
    expected = (1 + 0.25 * (104 / 106 - 1)) * (1 + 0.25 * (101 / 104 - 1))   # day 3, then day 4
    assert np.isclose(bt.equity.iloc[-1], expected)
    assert held_positions(IDX, [t]).tolist() == [0, 0, 0, 0.25, 0.25]


def test_fills_have_exact_prices():
    f = trade_fills([Trade(IDX[1], 101.0, -1, 0.5, IDX[3], 99.5)])
    assert f["side"].tolist() == ["short", "cover"] and f["price"].tolist() == [101.0, 99.5]
    assert f["to"].tolist() == [-0.5, 0.0]


class OneTrade(Strategy):
    name = "one trade"
    def compute(self, bars):
        self.trade(bars.index[1], bars.open.iloc[1], side=1, exit_date=bars.index[3], exit_price=107.0)


class Overlap(Strategy):
    name = "overlap"
    def compute(self, bars):
        self.trade(bars.index[0], 100, 1, exit_date=bars.index[3], exit_price=104)
        self.trade(bars.index[2], 103, 1)


def test_strategy_records_trades_and_simulate_uses_them():
    out = OneTrade().run(BARS)
    assert out.trades and out.positions.tolist() == [0, 1, 1, 0, 0]
    assert np.isclose(simulate(BARS, out, 0).equity.iloc[-1], 107 / 101)
    with pytest.raises(ValueError, match="overlap"):
        Overlap().run(BARS)
