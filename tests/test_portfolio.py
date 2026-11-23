import json

import numpy as np
import pandas as pd
import pytest

from quantlab import portfolio as pf
from quantlab import prop


def fake_leg_results(table):
    """leg_results replacement: each (ticker) -> daily returns from `table`."""
    def f(leg, history=None, costs=None):
        if leg.ticker not in table:
            raise ValueError(f"no data for {leg.ticker!r}")
        r = table[leg.ticker]
        return prop.DailyResults(r, r.clip(upper=0)), r * 0.5
    return f


@pytest.fixture()
def legs_data(monkeypatch):
    days = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0.001, 0.01, 300), index=days)
    b = pd.Series(rng.normal(0.001, 0.01, 300), index=days)
    late = pd.Series(rng.normal(0.0, 0.01, 200), index=days[100:])
    monkeypatch.setattr(pf, "leg_results", fake_leg_results({"A": a, "B": b, "LATE": late, "NEG": -a}))
    return a, b, late


def test_weights_combine_daily(legs_data):
    a, b, _ = legs_data
    res = pf.run([pf.Leg("A", "1d", "SmaCrossover", {}, 0.6), pf.Leg("B", "1d", "SmaCrossover", {}, 0.4)])
    assert np.allclose(res["returns"], 0.6 * a + 0.4 * b)
    assert np.allclose(res["low"], 0.6 * a.clip(upper=0) + 0.4 * b.clip(upper=0))   # worst points added up
    assert res["days"] == 300 and res["total_weight"] == pytest.approx(1.0)
    assert sum(res["equal_risk"].values()) == pytest.approx(1.0)


def test_only_the_common_period_and_failed_legs(legs_data):
    res = pf.run([pf.Leg("A", "1d", "SmaCrossover", {}, 0.5), pf.Leg("LATE", "1h", "SmaCrossover", {}, 0.5),
                  pf.Leg("NOPE", "1d", "SmaCrossover", {}, 0.5)])
    assert res["days"] == 200 and res["errors"][0]["leg"] == 2 and "no data" in res["errors"][0]["error"]
    with pytest.raises(ValueError, match="No leg"):
        pf.run([pf.Leg("NOPE", "1d", "SmaCrossover", {}, 1.0)])


def test_diversification_and_hedging(legs_data):
    a, _, _ = legs_data
    hedge = pf.run([pf.Leg("A", "1d", "SmaCrossover", {}, 0.5), pf.Leg("NEG", "1d", "SmaCrossover", {}, 0.5)])
    assert hedge["corr"]["values"][0][1] == pytest.approx(-1.0)
    assert np.allclose(hedge["returns"], 0.0)                                         # perfect hedge: flat
    two = pf.run([pf.Leg("A", "1d", "SmaCrossover", {}, 0.5), pf.Leg("B", "1d", "SmaCrossover", {}, 0.5)])
    one = pf.run([pf.Leg("A", "1d", "SmaCrossover", {}, 1.0)])
    assert two["portfolio"]["vol"] < one["portfolio"]["vol"]                          # unrelated legs: smoother
    lev = pf.run([pf.Leg("A", "1d", "SmaCrossover", {}, 2.0)])
    assert lev["portfolio"]["vol"] == pytest.approx(2 * one["portfolio"]["vol"])
    with pytest.raises(ValueError, match="weight"):
        pf.run([pf.Leg("A", "1d", "SmaCrossover", {}, -0.5)])


def test_prop_check_on_a_portfolio(legs_data):
    res = pf.run([pf.Leg("A", "1d", "SmaCrossover", {}, 0.5), pf.Leg("B", "1d", "SmaCrossover", {}, 0.5)])
    out = prop.check(pf.prop_source(res), sims=200)
    assert out["days"] == 300 and len(out["rows"]) >= 8
