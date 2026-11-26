"""Edge finder on synthetic markets (no MT5 needed)."""
import numpy as np
import pandas as pd

from quantlab import edges as E


def fake_market(seed, trend):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2017-06-01", "2026-11-20")
    steps = rng.normal(0, 0.01, len(idx))
    # trending regimes: drift flips sign every ~6 months
    drift = trend * 0.002 * np.sign(np.sin(np.arange(len(idx)) / 40))
    close = 100 * np.exp(np.cumsum(steps + drift))
    return pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995, "close": close,
                         "volume": 1000.0}, index=idx)


def test_trending_markets_show_an_edge_and_noise_does_not():
    data = {"EURUSD": fake_market(1, 1), "XAUUSD": fake_market(2, 1), "BTCUSD": fake_market(3, 0)}
    log = []
    markets, port, groups = E.run(list(data) + ["NOPE"], loader=lambda s: data[s], progress=log.append)
    assert any("skip NOPE" in line for line in log)
    assert set(markets["market"]) == set(data) and len(markets) == 3 * len(E.STRATEGIES)
    trend = markets[markets.market != "BTCUSD"]
    assert (trend["sharpe A"] > 0.5).mean() > 0.7                 # trend rules catch the regimes
    assert list(port.index) == list(E.STRATEGIES) and (port["markets"] == 3).all()
    assert port["sharpe A"].notna().all() and port["sharpe B"].notna().all()
    assert {"FX", "Metals", "Crypto"} <= set(groups["group"])
    text = E.show(markets, port, groups)
    assert "PORTFOLIO" in text and "EURUSD" in text


def test_weekend_returns_fold_into_monday():
    r = pd.Series([0.01, 0.01, 0.01], index=pd.to_datetime(["2026-11-20", "2026-11-21", "2026-11-23"]))
    w = E.weekdays_only(r)
    assert list(w.index.strftime("%a")) == ["Fri", "Mon"] and w.iloc[1] == np.float64(1.01 ** 2 - 1)


def test_long_short_donchian():
    from quantlab import strategies
    up = np.r_[np.linspace(100, 150, 120), np.linspace(150, 90, 120)]
    idx = pd.bdate_range("2024-01-01", periods=len(up))
    bars = pd.DataFrame({"open": up, "high": up, "low": up, "close": up, "volume": 1.0}, index=idx)
    pos = strategies.get("Breakout")(entry=20, exit=10, allow_short=True).run(bars)
    pos = pos if isinstance(pos, pd.Series) else pos.positions
    assert pos.iloc[100] == 1.0 and pos.iloc[-1] == -1.0
