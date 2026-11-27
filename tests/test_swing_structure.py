"""Swing structure: levels, confirmation times and 'taken' levels."""
import numpy as np
import pandas as pd

from strategies.swing_structure import SwingStructure, next_beyond, pivot_levels, swing_table


def test_levels_are_swings_of_swings():
    h = np.array([1, 3, 2, 4, 3, 6, 2, 5, 3, 4, 1, 2, 1.0])
    idx, level, known = pivot_levels(h, True, 3)
    assert list(idx) == [1, 3, 5, 7, 9, 11]              # every level-1 swing high
    assert dict(zip(idx, level))[5] == 2                 # 6 is above its neighbours 4 and 5
    assert sum(level == 2) == 1
    assert dict(zip(idx, known))[1] == 2                 # level 1: known one candle later
    assert dict(zip(idx, known))[5] == 8                 # level 2: known when the next swing (7) is


def test_taken_means_a_later_candle_trades_beyond():
    v = np.array([5, 3, 4, 6, 2.0])
    assert list(next_beyond(v, True)) == [3, 2, 3, 5, 5]  # 5 is first exceeded at index 3; 6 never
    assert list(next_beyond(v, False)) == [1, 4, 4, 4, 5]


def test_strategy_draws_marks_and_lines_without_trading():
    rng = np.random.default_rng(3)
    c = 100 + np.cumsum(rng.normal(0, 1, 400))
    idx = pd.bdate_range("2024-01-01", periods=400)
    bars = pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1.0}, index=idx)
    out = SwingStructure().run(bars)
    assert (out.positions == 0).all() and not out.trades
    levels = {m.level for m in out.marks}
    assert levels == {1, 2, 3}
    assert all(l.level >= 2 and l.end >= l.start for l in out.level_lines)
    t = swing_table(bars, 3)
    # higher levels are rarer, and every swing is confirmed after its own candle
    assert (t.level == 1).sum() > (t.level == 2).sum() > (t.level == 3).sum() > 0
    assert (t.known_i > t.i).all()
    # a taken high really was exceeded at that candle, and not before
    r = t[(t.kind == "high") & t.taken_i.notna()].iloc[0]
    hi = bars["high"].to_numpy()
    assert hi[int(r.taken_i)] > r.price and (hi[r.i + 1:int(r.taken_i)] <= r.price).all()


def test_chart_payload_has_marks_and_levels():
    from quantlab.chart import lines_payload, marks_payload
    out = SwingStructure(lines_from=1).run(pd.DataFrame(
        {"open": [1, 2, 3, 2, 1, 2.0], "high": [1, 2, 3, 2, 1, 2.0], "low": [1, 2, 3, 2, 1, 2.0],
         "close": [1, 2, 3, 2, 1, 2.0], "volume": 1.0}, index=pd.bdate_range("2024-01-01", periods=6)))
    m, l = marks_payload(out.marks), lines_payload(out.level_lines)
    assert {"time": "2024-01-03", "price": 3.0, "kind": "high", "level": 1, "text": ""} in m
    assert any(x["kind"] == "low" and x["t0"] == "2024-01-05" for x in l)
