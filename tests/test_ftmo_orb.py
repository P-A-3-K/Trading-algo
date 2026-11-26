"""FTMO ORB check on synthetic candles."""
import numpy as np
import pandas as pd

from quantlab import ftmo_orb as F
from tests.test_edges import fake_intraday


def test_edge_and_real_costs_per_trade():
    ftmo = fake_intraday(5, "2026-01-02", "2026-06-30")
    ftmo["spread"] = ftmo["close"] * 0.0001                     # 1 bp spread -> 0.5 bp half-spread
    etf = fake_intraday(6, "2023-01-02", "2026-06-30")
    log = []
    t = F.check(["US100.cash", "NOPE"], mt5_loader=lambda s: ftmo if s == "US100.cash" else 1 / 0,
                etf_loader=lambda s: etf, progress=log.append)
    assert list(t.index) == ["US100.cash"] and any("skip NOPE" in x for x in log)
    row = t.loc["US100.cash"]
    assert row["ETF"] == "QQQ" and row["ETF trades"] > 100 and row["FTMO trades"] > 50
    assert abs(row["half spread bp"] - 0.5) < 0.05
    # a round trip pays the half-spread twice; slippage adds to it
    assert 0.9 < row["cost bp 0%"] < 1.1 and row["cost bp 0%"] < row["cost bp 2%"] < row["cost bp 5%"]
    assert row["sharpe 0%"] >= row["sharpe 5%"]
    assert "edge / cost" in F.show(t)
