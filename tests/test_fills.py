"""Fill report on a made-up MT5 trade history."""
from types import SimpleNamespace as NS

import pytest

from quantlab import fills as F

T0 = 1_780_000_000


def order(ticket, typ, price, sl=0.0, tp=0.0):
    return NS(ticket=ticket, type=typ, price_open=price, sl=sl, tp=tp)


def deal(order_, pos, typ, entry, price, reason=3, magic=7_700_123, symbol="US100.cash", t=0, profit=0.0):
    return NS(order=order_, position_id=pos, type=typ, entry=entry, price=price, reason=reason, magic=magic,
              symbol=symbol, time=T0 + t, volume=1.0, profit=profit, commission=-0.5, fee=0.0, swap=0.0)


def history():
    orders = [
        order(1, 4, 20000.0, sl=19900.0),            # buy stop at 20000, stop loss 19900
        order(2, 1, 0.0),                            # server's stop-loss close order (no price)
        order(3, 5, 20000.0, sl=20100.0),            # sell stop at 20000
        order(4, 0, 19950.0),                        # robot's end-of-day buy-to-close at 19950
        order(5, 4, 100.0, sl=99.0),                 # someone else's trade
    ]
    deals = [
        deal(1, 10, F.DEAL_BUY, F.ENTRY_IN, 20002.0),                       # 1 bp worse
        deal(2, 10, F.DEAL_SELL, F.ENTRY_OUT, 19896.0, reason=4, t=600),    # stop: 2 bp worse
        deal(3, 11, F.DEAL_SELL, F.ENTRY_IN, 20000.0, t=86400),             # exact
        deal(4, 11, F.DEAL_BUY, F.ENTRY_OUT, 19949.0, t=90000),             # closed 0.5 bp better
        deal(5, 12, F.DEAL_BUY, F.ENTRY_IN, 100.0, magic=42, symbol="X"),
        deal(5, 12, F.DEAL_SELL, F.ENTRY_OUT, 101.0, magic=42, symbol="X"),
    ]
    return orders, deals


def test_slippage_per_fill():
    trips = F.round_trips(*history())
    assert list(trips["symbol"]) == ["US100.cash", "US100.cash"]            # magic 42 ignored
    a, b = trips.iloc[0], trips.iloc[1]
    assert a["entry by"] == "stop" and a["entry slip bp"] == pytest.approx(1.0)
    assert a["exit by"] == "stop loss" and a["exit asked"] == 19900.0
    assert a["exit slip bp"] == pytest.approx(4 / 19900 * 1e4)
    assert b["side"] == "short" and b["entry slip bp"] == pytest.approx(0.0)
    assert b["exit by"] == "robot" and b["exit slip bp"] == pytest.approx(-1 / 19950 * 1e4)
    assert b["result bp"] == pytest.approx(-(19949 / 20000 - 1) * 1e4)


def test_cost_summary_and_magic_filter():
    trips = F.round_trips(*history())
    trips["notional"] = 20000.0                      # commission 1.0 per trade = 0.5 bp
    s = F.summary(trips, {"US100.cash": 0.3}, {"US100.cash": 1.05})
    row = s.loc["US100.cash"]
    assert row["trades"] == 2 and row["commission bp"] == pytest.approx(0.5)
    expected = 0.6 + trips["entry slip bp"].mean() + trips["exit slip bp"].mean() + 0.5
    assert row["cost per trip bp"] == pytest.approx(expected) and row["backtest assumed bp"] == 1.05
    assert "Cost per round trip" in F.show(trips, s)
    only = F.round_trips(*history(), magics=[42])
    assert list(only["symbol"]) == ["X"]
