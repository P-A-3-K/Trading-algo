import numpy as np
import pandas as pd

from quantlab import chart
from quantlab.metrics import (full_stats, longest_drawdown, round_trips, sortino)

IDX = pd.date_range("2024-01-01", periods=8)


def test_longest_drawdown_counts_days_below_peak():
    r = pd.Series([0.1, -0.1, 0.05, 0.10, -0.01, -0.01, 0.05, 0.0], index=IDX)
    # equity: 1.1, .99, 1.04, 1.144 (new peak), 1.133, 1.121, 1.177 (new peak), 1.177
    assert longest_drawdown(r) == 2


def test_sortino_ignores_upside():
    calm = pd.Series([0.01, -0.005] * 4, index=IDX)
    spiky_up = pd.Series([0.05, -0.01] * 4, index=IDX)
    assert sortino(spiky_up) > sortino(calm) > 0


def test_round_trips():
    held = pd.Series([0, 1, 1, 0, 0, 1, 1, 1], index=IDX, dtype=float)
    r = pd.Series([0, 0.10, -0.05, -0.001, 0, 0.02, 0.02, 0.02], index=IDX)
    trips = round_trips(r, held)
    assert trips["days"].tolist() == [2, 3]
    assert np.isclose(trips["return"].iloc[0], 1.10 * 0.95 * 0.999 - 1)   # incl. exit cost day
    assert trips["entry"].iloc[1] == IDX[5]


def test_full_stats_trade_fields():
    held = pd.Series([0, 1, 1, 0, 0, 1, 1, 0], index=IDX, dtype=float)
    r = pd.Series([0, 0.10, -0.05, 0, 0, -0.02, -0.02, 0], index=IDX)
    st = full_stats(r, held, fills=4)
    assert st["round_trips"] == 2 and st["win_rate"] == 0.5 and st["fills"] == 4
    assert st["time_in_market"] == 0.5


def test_stats_payload_rows():
    s = {"total_return": 1.0, "cagr": 0.1, "win_rate": 0.6, "sharpe": float("nan")}
    b = {"total_return": 2.0, "cagr": 0.12, "win_rate": 1.0}
    rows = {r["key"]: r for r in chart.stats_payload(s, b)}
    assert rows["final_value"]["strategy"] == 20000 and rows["final_value"]["benchmark"] == 30000
    assert rows["win_rate"]["benchmark"] is None          # meaningless for buy & hold
    assert rows["sharpe"]["strategy"] is None             # NaN becomes "–"
    assert rows["volatility"]["better"] == "lower"
