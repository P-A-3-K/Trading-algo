import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import backtest
from quantlab.data import synthetic_prices
from quantlab.strategies import BUILTIN, MeanReversion, SmaCrossover
from quantlab.strategy import Param, Strategy, check_no_lookahead


def bars(n=600, seed=0):
    return pd.DataFrame({"close": synthetic_prices(n_days=n, seed=seed)})


def test_defaults_and_overrides():
    s = SmaCrossover(fast=20)
    assert (s.fast, s.slow) == (20, 200)
    assert repr(s) == "SmaCrossover(fast=20, slow=200)"


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        SmaCrossover(fsat=10)
    with pytest.raises(ValueError, match="at least"):
        SmaCrossover(fast=1)
    with pytest.raises(ValueError, match="whole number"):
        SmaCrossover(fast="ten")
    with pytest.raises(ValueError, match="shorter"):
        SmaCrossover(fast=100, slow=50)


def test_param_types():
    assert SmaCrossover(fast="30").fast == 30            # strings from a web form
    assert MeanReversion(entry="1.5").entry == 1.5
    assert MeanReversion.info()["params"]["entry"]["type"] == "float"


def test_run_returns_positions_and_plots():
    res = SmaCrossover(fast=10, slow=30).run(bars())
    assert res.positions.isin([0.0, 1.0]).all()           # warm-up NaN -> 0
    assert [p.name for p in res.plots] == ["SMA 10", "SMA 30"]
    lower = MeanReversion().run(bars()).plots
    assert [p.pane for p in lower] == ["price", "lower"]


def test_run_resets_plots_between_runs():
    s = SmaCrossover()
    s.run(bars()); res = s.run(bars())
    assert len(res.plots) == 2


class TooBig(Strategy):
    name = "too big"
    def compute(self, bars):
        return pd.Series(2.0, index=bars.index)


class WrongLength(Strategy):
    name = "wrong length"
    def compute(self, bars):
        return pd.Series(1.0, index=bars.index[:-1])


def test_output_is_checked():
    with pytest.raises(ValueError, match="between -1 and 1"):
        TooBig().run(bars())
    with pytest.raises(ValueError, match="one value per bar"):
        WrongLength().run(bars())


class Peeker(Strategy):
    name = "peeker"
    def compute(self, bars):
        tomorrow = bars.close.pct_change().shift(-1)      # uses the future!
        return (tomorrow > 0).astype(float)


def test_lookahead_detector():
    data = bars()
    for cls in BUILTIN:
        assert check_no_lookahead(cls(), data) == [], cls.name
    assert len(check_no_lookahead(Peeker(), data)) > 0


def test_backtest_runs_strategy():
    res = backtest(SmaCrossover(fast=10, slow=30), bars())
    assert res.strategy == "SmaCrossover(fast=10, slow=30)"
    assert len(res.plots) == 2 and np.isfinite(res.equity.iloc[-1])
