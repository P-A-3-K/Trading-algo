"""Portfolio: several strategies (legs) traded together in one account.

    from quantlab.portfolio import Leg, run
    res = run([Leg("GLD", "1d", "SmaCrossover", {"fast": 25, "slow": 150}, 0.5),
               Leg("SPY", "1d", "RsiReversion", {"period": 2, "buy_below": 10, "sell_above": 70}, 0.5)])

How legs are combined:
  * each leg is backtested on its own (ticker, timeframe, strategy, settings), with costs;
  * its results are turned into one return per trading day (and that day's lowest point);
  * the account's daily return = sum of weight x leg return (rebalanced to the weights
    daily, the usual simplification). Weights can add up to more than 100% (leverage,
    as on a prop-firm CFD account) or less (some cash);
  * only the period ALL legs cover is used, so a leg with a long history can't
    dominate the comparison;
  * the day's lowest point adds up each leg's lowest point: cautious, as if every
    leg hit its worst moment at the same time (what prop-firm loss limits care about).

Why bother: legs that don't move together cancel part of each other's bad days, so the
combined account can have a better return per unit of risk (Sharpe) than any single leg.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import download as dl
from . import prop, strategies

TRADING_DAYS = 252
MAX_LEGS = 12


@dataclass
class Leg:
    ticker: str
    timeframe: str
    strategy: str                   # strategy id, e.g. "SmaCrossover"
    settings: dict = field(default_factory=dict)
    weight: float = 1.0             # fraction of the account, e.g. 0.5 = 50%

    def label(self) -> str:
        s = strategies.get(self.strategy)
        shown = {k: v for k, v in self.settings.items() if not (k == "flat_at_close" and not v)}
        args = ", ".join(f"{k}={v}" for k, v in shown.items())
        return f"{self.ticker} {self.timeframe} · {s.name}" + (f" ({args})" if args else "")


def leg_results(leg: Leg, history: str = None, costs=None):
    """(strategy DailyResults, buy & hold daily returns) for one leg."""
    from .backtest import run_backtest, simulate
    from .costs import CostModel
    cls = strategies.get(leg.strategy)
    strat = cls(**leg.settings)
    bars = dl.load_bars(leg.ticker, leg.timeframe, history if leg.timeframe != "1d" else None)
    prepared = (costs or CostModel()).prepare(bars)
    r = simulate(bars, strat.run(bars), costs=prepared).returns
    bh = run_backtest(bars["close"], pd.Series(1.0, index=bars.index), costs=prepared).returns
    return prop.daily_results(r), prop.daily_results(bh).ret


def stats(r: pd.Series, low: pd.Series = None) -> dict:
    """Headline numbers for a daily return series."""
    from .optimize import prop_return
    r = r.astype(float)
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    sd = r.std()
    years = len(r) / TRADING_DAYS
    return {"cagr": float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else None,
            "vol": float(sd * np.sqrt(TRADING_DAYS)),
            "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
            "max_dd": dd,
            "prop": prop_return(r.to_numpy(), dd, TRADING_DAYS),
            "worst_day": float((low if low is not None else r).min()),
            "final": float(eq.iloc[-1])}


def equal_risk_weights(returns: dict, total: float = 1.0) -> dict:
    """Weights so each leg contributes the same volatility (1 / its volatility), summing to `total`."""
    inv = {k: 1 / v.std() if v.std() > 0 else 0.0 for k, v in returns.items()}
    s = sum(inv.values())
    return {k: total * v / s if s else total / len(inv) for k, v in inv.items()}


def run(legs: list, history: str = None, costs=None, start=None, end=None) -> dict:
    """Backtest every leg, combine them, and compare."""
    if not legs:
        raise ValueError("Add at least one leg (ticker + timeframe + strategy).")
    if len(legs) > MAX_LEGS:
        raise ValueError(f"At most {MAX_LEGS} legs.")
    for leg in legs:
        if leg.weight < 0 or leg.weight > 10:
            raise ValueError(f"{leg.ticker}: weight must be between 0% and 1000%")
    daily, bench, errors = {}, {}, []
    for i, leg in enumerate(legs):
        try:
            d, b = leg_results(leg, history, costs)
        except Exception as e:                     # bad ticker, no data: report, keep going
            errors.append({"leg": i, "label": leg.label() if leg.strategy else leg.ticker,
                           "error": str(e) or type(e).__name__})
            continue
        daily[i], bench[i] = d, b
    if not daily:
        raise ValueError("No leg could be backtested: " + "; ".join(e["error"] for e in errors))
    # The period every leg covers
    first = max(d.ret.index[0] for d in daily.values())
    last = min(d.ret.index[-1] for d in daily.values())
    if start:
        first = max(first, pd.Timestamp(start))
    if end:
        last = min(last, pd.Timestamp(end))
    if last <= first:
        raise ValueError("The legs' histories don't overlap. Use tickers/timeframes with a common period.")
    days = pd.DatetimeIndex(sorted(set().union(*(d.ret.loc[first:last].index for d in daily.values()))))
    if len(days) < 40:
        raise ValueError(f"Only {len(days)} trading days in common: too few to judge a portfolio.")
    R = pd.DataFrame({i: d.ret.reindex(days).fillna(0.0) for i, d in daily.items()})
    L = pd.DataFrame({i: d.low.reindex(days).fillna(0.0) for i, d in daily.items()})
    B = pd.DataFrame({i: b.reindex(days).fillna(0.0) for i, b in bench.items()})
    w = pd.Series({i: legs[i].weight for i in daily})
    port_r, port_low = (R * w).sum(axis=1), (L * w).sum(axis=1)
    bench_r = (B * w).sum(axis=1)
    leg_rows = []
    for i in daily:
        leg_rows.append({"leg": i, "label": legs[i].label(), "weight": legs[i].weight,
                         **stats(R[i], L[i]),
                         "weighted": stats(R[i] * legs[i].weight, L[i] * legs[i].weight)})
    corr = R.corr().fillna(0.0)
    avg_sharpe = float(np.mean([r["sharpe"] for r in leg_rows]))
    return {
        "legs": leg_rows, "errors": errors,
        "portfolio": stats(port_r, port_low), "benchmark": stats(bench_r),
        "corr": {"legs": list(map(int, corr.index)), "values": corr.round(3).values.tolist()},
        "avg_corr": float(corr.values[np.triu_indices(len(corr), 1)].mean()) if len(corr) > 1 else None,
        "avg_leg_sharpe": avg_sharpe,
        "equal_risk": {int(k): float(v) for k, v in equal_risk_weights({i: R[i] for i in daily},
                                                               total=float(w.sum()) or 1.0).items()},
        "total_weight": float(w.sum()),
        "start": str(days[0].date()), "end": str(days[-1].date()), "days": len(days),
        "returns": port_r, "low": port_low, "bench_returns": bench_r, "leg_returns": R,
    }


def prop_source(result: dict) -> prop.DailyResults:
    """The combined account, for the Prop firm check."""
    return prop.DailyResults(result["returns"], result["low"])
