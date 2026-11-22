import json

import numpy as np
import pandas as pd
import pytest

from quantlab import optimize
from quantlab.app import default_ranges, nice_step
from quantlab.data import synthetic_prices
from quantlab.strategies import MeanReversion, SmaCrossover


def bars(n=900, seed=3):
    return pd.DataFrame({"close": synthetic_prices(n_days=n, seed=seed)})


def test_values_inclusive_and_float_safe():
    assert optimize.values(5, 20, 5) == [5, 10, 15, 20]
    assert optimize.values(0.25, 1.0, 0.25, is_int=False) == [0.25, 0.5, 0.75, 1.0]
    assert optimize.values(0.1, 0.3, 0.1, is_int=False) == [0.1, 0.2, 0.3]   # no 0.30000000000000004 drop
    with pytest.raises(ValueError):
        optimize.values(10, 5, 1)


def test_grid_keeps_unvaried_settings_at_default():
    g = optimize.grid(SmaCrossover, {"fast": (10, 30, 10)})
    assert g == [{"fast": 10, "slow": 200}, {"fast": 20, "slow": 200}, {"fast": 30, "slow": 200}]
    with pytest.raises(ValueError, match="no setting"):
        optimize.grid(SmaCrossover, {"speed": (1, 2, 1)})
    with pytest.raises(ValueError, match="at least"):
        optimize.grid(SmaCrossover, {"fast": (1, 30, 10)})          # outside allowed range


def test_grid_limit():
    with pytest.raises(ValueError, match="too many"):
        optimize.grid(SmaCrossover, {"fast": (2, 400, 1), "slow": (5, 500, 1)})


def test_sweep_splits_and_skips_invalid():
    b = bars()
    res = optimize.sweep(b, SmaCrossover, {"fast": (10, 50, 20), "slow": (30, 70, 20)}, split=0.6)
    # fast 10/30/50 x slow 30/50/70 = 9, invalid where fast >= slow: (30,30),(50,30),(50,50)
    assert len(res.table) == 6 and len(res.skipped) == 3
    assert res.split_date == b.index[int(len(b) * 0.6)]
    assert set(res.table.columns) >= {"is_sharpe", "oos_sharpe", "is_trades", "oos_final", "is_rank"}
    # IS + OOS growth compounds to the full-period result
    from quantlab.backtest import run_backtest
    row = res.table.iloc[0]
    full = run_backtest(b["close"], SmaCrossover(fast=int(row.fast), slow=int(row.slow)).run(b).positions)
    assert np.isclose(row.is_final / 1e4 * row.oos_final / 1e4, full.equity.iloc[-1])


def test_overfitting_report_fields():
    res = optimize.sweep(bars(), MeanReversion, {"lookback": (10, 40, 10), "entry": (0.5, 2.0, 0.5)})
    rep = optimize.overfitting_report(res)
    assert rep["combos"] == 16 and -1 <= rep["rank_corr"] <= 1
    assert 1 <= rep["best_is_oos_rank"] <= 16


def test_default_ranges_are_round_and_allowed():
    assert nice_step(398, True) == 25
    r = default_ranges(SmaCrossover)
    assert r["fast"] == [25, 400, 25] and r["slow"] == [50, 500, 50]
    e = default_ranges(MeanReversion)["entry"]
    assert e[0] >= 0.1 and e[1] <= 4.0


def test_optimize_page(tmp_path, monkeypatch):
    from tests.test_app import get  # reuse helper
    import threading
    from quantlab import app, download as dl
    close = synthetic_prices(n_days=600, seed=9)
    pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Adj Close": close,
                  "Volume": 1}, index=close.index.rename("Date")).to_csv(tmp_path / "TEST.csv")
    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    srv = app.make_server(port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        status, body = get(base + "/optimize?ticker=TEST&strategy=SmaCrossover")
        data = json.loads(body.split("const DATA = ")[1].split(";\n")[0])
        assert status == 200 and data["result"] is None and data["app"]["ranges"]["fast"] == [25, 400, 25]
        q = "fast_from=10&fast_to=30&fast_step=10&slow_from=40&slow_to=60&slow_step=20&split=0.7&rank=cagr&run=1"
        status, body = get(base + "/optimize?ticker=TEST&strategy=SmaCrossover&" + q)
        data = json.loads(body.split("const DATA = ")[1].split(";\n")[0])
        r = data["result"]
        assert status == 200 and r["count"] == 6 and r["params"] == ["fast", "slow"]
        assert r["axis_values"] == {"fast": [10, 20, 30], "slow": [40, 60]}
        assert data["app"]["rank"] == "cagr"
        assert get(base + "/optimize?ticker=TEST&strategy=BuyAndHold")[0] == 400
        assert get(base + "/optimize?ticker=TEST&strategy=SmaCrossover&fast_from=30&fast_to=10&fast_step=5"
                   "&slow_from=40&slow_to=60&slow_step=20&run=1")[0] == 400
    finally:
        srv.shutdown(); srv.server_close()
