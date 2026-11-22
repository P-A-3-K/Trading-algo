"""Parameter sweep (grid search): backtest every combination of settings.

    from quantlab.optimize import sweep
    res = sweep(bars, SmaCrossover, {"fast": (5, 100, 5), "slow": (20, 300, 10)})
    res.table.sort_values("is_sharpe", ascending=False).head()

The overfitting guard is built in. The data is split by date:
  * in-sample (IS):      the first `split` share of history; rank settings here
  * out-of-sample (OOS): the rest; the optimizer never ranks by it

The best IS result always looks good, partly by luck: with hundreds of
combinations, some fit the past's noise. How the top IS settings do OOS is
the honest estimate of what you'd have got.
"""
import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .metrics import cagr, max_drawdown, round_trips, sharpe

MAX_COMBINATIONS = 5000
START = 10_000


def values(start, stop, step, is_int=True) -> list:
    """start, start+step, ... up to and including stop (inclusive grid)."""
    if step <= 0:
        raise ValueError("step must be positive")
    if stop < start:
        raise ValueError(f"'to' ({stop}) must not be below 'from' ({start})")
    n = int(np.floor((stop - start) / step + 1e-9)) + 1
    vals = [start + i * step for i in range(n)]
    return [int(round(v)) for v in vals] if is_int else [round(float(v), 10) for v in vals]


def grid(cls, ranges: dict) -> list:
    """Every combination of settings, as dicts. ranges: {name: (from, to, step)};
    settings left out stay at their default."""
    for name in ranges:
        if name not in cls.params:
            raise ValueError(f"{cls.name} has no setting {name!r}")
    axes = {}
    for name, p in cls.params.items():
        if name in ranges:
            lo, hi, st = ranges[name]
            axes[name] = values(p.coerce(name, lo), p.coerce(name, hi), st, p.is_int)
        else:
            axes[name] = [p.default]
    combos = [dict(zip(axes, c)) for c in itertools.product(*axes.values())]
    if len(combos) > MAX_COMBINATIONS:
        raise ValueError(f"{len(combos):,} combinations is too many (max {MAX_COMBINATIONS:,}). "
                         "Use bigger steps or narrower ranges.")
    return combos


def segment_stats(returns: pd.Series, held: pd.Series, prefix: str) -> dict:
    """Headline numbers for one stretch of the backtest."""
    if len(returns) < 2:
        return {}
    trips = round_trips(returns, held)
    changes = held.diff().fillna(held.iloc[0]).ne(0).sum()
    return {
        f"{prefix}_final": START * float((1 + returns).prod()),
        f"{prefix}_cagr": cagr(returns),
        f"{prefix}_sharpe": sharpe(returns),
        f"{prefix}_max_dd": max_drawdown(returns),
        f"{prefix}_trades": int(changes),
        f"{prefix}_win_rate": float((trips["return"] > 0).mean()) if len(trips) else np.nan,
        f"{prefix}_exposure": float((held != 0).mean()),
    }


@dataclass
class SweepResult:
    table: pd.DataFrame            # one row per valid combination
    params: list                   # setting names that were varied
    split_date: pd.Timestamp       # first out-of-sample day
    start: pd.Timestamp
    end: pd.Timestamp
    skipped: list = field(default_factory=list)   # (settings, reason) that were invalid
    seconds: float = 0.0
    benchmark: dict = field(default_factory=dict)  # buy & hold, IS and OOS


def sweep(bars: pd.DataFrame, cls, ranges: dict, split: float = 0.7,
          cost_bps: float = 5.0) -> SweepResult:
    """Backtest every combination in `ranges` and score it in-sample and out-of-sample."""
    if not 0.3 <= split <= 0.95:
        raise ValueError("in-sample share must be between 30% and 95%")
    t0 = time.time()
    cut = int(len(bars) * split)
    split_date = bars.index[cut]
    rows, skipped = [], []
    for settings in grid(cls, ranges):
        try:
            strategy = cls(**settings)
        except ValueError as e:                 # e.g. fast >= slow
            skipped.append((settings, str(e)))
            continue
        bt = run_backtest(bars["close"], strategy.run(bars).positions, cost_bps)
        row = dict(settings)
        row.update(segment_stats(bt.returns.iloc[:cut], bt.positions.iloc[:cut], "is"))
        # OOS starts flat-or-holding exactly as the strategy was: no fresh entry cost
        row.update(segment_stats(bt.returns.iloc[cut:], bt.positions.iloc[cut:], "oos"))
        rows.append(row)
    table = pd.DataFrame(rows)
    bh = run_backtest(bars["close"], pd.Series(1.0, index=bars.index), cost_bps)
    bench = {**segment_stats(bh.returns.iloc[:cut], bh.positions.iloc[:cut], "is"),
             **segment_stats(bh.returns.iloc[cut:], bh.positions.iloc[cut:], "oos")}
    if len(table):
        for side in ("is", "oos"):              # rank 1 = best Sharpe in that period
            table[f"{side}_rank"] = table[f"{side}_sharpe"].rank(ascending=False, method="min").astype(int)
    return SweepResult(table, list(ranges), split_date, bars.index[0], bars.index[-1],
                       skipped, time.time() - t0, bench)


def overfitting_report(res: SweepResult, by: str = "sharpe", top: int = 10) -> dict:
    """How well did in-sample ranking predict out-of-sample results?"""
    t = res.table.dropna(subset=[f"is_{by}", f"oos_{by}"])
    if len(t) < 3:
        return {}
    best = t.sort_values(f"is_{by}", ascending=False)
    topk = best.head(top)
    oos_rank_of_best = int((t[f"oos_{by}"] > best.iloc[0][f"oos_{by}"]).sum()) + 1
    return {
        "metric": by,
        "combos": len(t),
        "best_is": float(best.iloc[0][f"is_{by}"]),
        "best_is_oos": float(best.iloc[0][f"oos_{by}"]),
        "best_is_oos_rank": oos_rank_of_best,
        "top_is_mean": float(topk[f"is_{by}"].mean()),
        "top_oos_mean": float(topk[f"oos_{by}"].mean()),
        "all_oos_median": float(t[f"oos_{by}"].median()),
        # Spearman rank correlation: +1 = IS ranking held up perfectly, 0 = no link
        "rank_corr": float(t[f"is_{by}"].rank().corr(t[f"oos_{by}"].rank())),
    }
