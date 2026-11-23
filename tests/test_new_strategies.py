import numpy as np
import pandas as pd
import pytest

from quantlab import strategies
from quantlab.data import synthetic_prices


def bars(n=1500, seed=8):
    c = synthetic_prices(n_days=n, seed=seed)
    return pd.DataFrame({"open": c, "high": c * 1.005, "low": c * 0.995, "close": c, "volume": 1e6})


def test_every_shipped_strategy_is_documented():
    for sid, cls in strategies.available().items():
        info = cls.info()
        assert len(info["guide"]) > 200, f"{sid} needs a guide"
        missing = [k for k, p in info["params"].items() if not p["help"]]
        assert not missing, f"{sid}: settings without help text: {missing}"


def test_rsi_exits_follow_the_rules():
    from strategies.rsi import RsiReversion, rsi
    b = bars()
    for short in (False, True):
        s = RsiReversion(period=5, buy_below=25, sell_above=60, allow_short=short)
        p, r = s.run(b).positions, rsi(b["close"], 5)
        prev = p.shift(1).fillna(0)
        assert (r[(prev != 1) & (p == 1)] < 25).all()              # long entries oversold
        assert (r[(prev == 1) & (p != 1)] > 60).all()              # long exits above sell level
        if short:
            assert ((prev != -1) & (p == -1)).any()
            assert (r[(prev != -1) & (p == -1)] > 75).all()
            assert (r[(prev == -1) & (p != -1)] < 40).all()
        else:
            assert (p >= 0).all()


def test_macd_position_matches_lines():
    from strategies.macd import MacdCrossover
    b = bars()
    out = MacdCrossover(fast=12, slow=26, signal=9, allow_short=True).run(b)
    lines = {pl.name: pl.values for pl in out.plots}
    live = lines["MACD"].notna()
    expected = np.where(lines["MACD"] > lines["Signal"], 1.0, -1.0)
    assert np.array_equal(out.positions[live].to_numpy(), expected[live.to_numpy()])
    assert (out.positions[~live] == 0).all()
    with pytest.raises(ValueError, match="shorter"):
        MacdCrossover(fast=30, slow=26)
