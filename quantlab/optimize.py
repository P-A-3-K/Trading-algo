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

from .backtest import run_backtest, simulate
from .metrics import cagr, max_drawdown, round_trips, sharpe

MAX_COMBINATIONS = 5000
START = 10_000
PROP_DD = 0.05          # "prop-safe" size: scaled so the worst drawdown is 5% (half a 10% limit)
PROP_MAX_SIZE = 20.0


def prop_return(r: np.ndarray, max_dd: float, ppy: float) -> float:
    """Return per year if the position were scaled so its worst drawdown is PROP_DD.

    A smooth strategy can be traded bigger inside a prop firm's loss limit, so this
    ranks by what you could earn there, not by the raw return."""
    if not max_dd or max_dd >= 0:
        return np.nan
    size = min(PROP_DD / -max_dd, PROP_MAX_SIZE)
    return float(np.expm1(ppy * np.mean(np.log1p(np.maximum(size * r, -0.999999)))))



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
    params = cls.all_params()
    for name in ranges:
        if name not in params:
            raise ValueError(f"{cls.name} has no setting {name!r}")
    axes = {}
    for name, p in params.items():
        if name in ranges and p.is_bool:                       # yes/no: one or both
            lo, hi = p.coerce(name, str(int(ranges[name][0]))), p.coerce(name, str(int(ranges[name][1])))
            axes[name] = sorted({lo, hi})
        elif name in ranges:
            lo, hi, st = ranges[name]
            axes[name] = values(p.coerce(name, lo), p.coerce(name, hi), st, p.is_int)
        else:
            axes[name] = [p.default]
    combos = [dict(zip(axes, c)) for c in itertools.product(*axes.values())]
    if len(combos) > MAX_COMBINATIONS:
        raise ValueError(f"{len(combos):,} combinations is too many (max {MAX_COMBINATIONS:,}). "
                         "Use bigger steps or narrower ranges.")
    return combos


def segment_stats(returns: pd.Series, held: pd.Series, prefix: str, ppy: float = None) -> dict:
    """Headline numbers for one stretch of the backtest.

    Same numbers as metrics.cagr / sharpe / max_drawdown / round_trips, worked out
    with plain numpy arrays: a sweep calls this thousands of times.
    """
    if len(returns) < 2:
        return {}
    from .metrics import periods_per_year
    ppy = ppy or periods_per_year(returns)
    r = np.nan_to_num(returns.to_numpy(dtype=float))
    h = np.nan_to_num(held.to_numpy(dtype=float))
    n = len(r)
    growth = np.cumprod(1 + r)
    total = growth[-1]
    sd = r.std(ddof=1)
    # Round trips: each stretch of bars with a position, plus the exit bar after it (its cost)
    in_trade = h != 0
    edges = np.diff(np.concatenate(([0], in_trade.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    if len(starts):
        logs = np.concatenate(([0.0], np.cumsum(np.log1p(np.maximum(r, -0.999999)))))
        last = np.minimum(ends + 1, n - 1)                  # include the exit bar if there is one
        trip = np.expm1(logs[last + 1] - logs[starts])
        win_rate = float((trip > 0).mean())
    else:
        win_rate = np.nan
    changes = np.count_nonzero(np.diff(h)) + (h[0] != 0)
    max_dd = float((growth / np.maximum.accumulate(growth) - 1).min())
    return {
        f"{prefix}_final": START * float(total),
        f"{prefix}_cagr": float(total ** (ppy / n) - 1),
        f"{prefix}_sharpe": float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0,
        f"{prefix}_max_dd": max_dd,
        f"{prefix}_prop": prop_return(r, max_dd, ppy),
        f"{prefix}_trades": int(changes),
        f"{prefix}_win_rate": win_rate,
        f"{prefix}_exposure": float(in_trade.mean()),
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
          cost_bps: float = 5.0, costs=None, progress=None) -> SweepResult:
    """Backtest every combination in `ranges` and score it in-sample and out-of-sample.

    progress: optional function(done, total), called before each combination."""
    if not 0.3 <= split <= 0.95:
        raise ValueError("in-sample share must be between 30% and 95%")
    t0 = time.time()
    cut = int(len(bars) * split)
    split_date = bars.index[cut]
    rows, skipped = [], []
    prepared = costs.prepare(bars) if costs is not None else None   # once, not per combination
    from .metrics import periods_per_year
    ppy_is, ppy_oos = periods_per_year(bars.index[:cut].to_series()), periods_per_year(bars.index[cut:].to_series())
    combos = grid(cls, ranges)
    for i, settings in enumerate(combos):
        if progress is not None:
            progress(i, len(combos))            # may raise to cancel
        try:
            strategy = cls(**settings)
        except ValueError as e:                 # e.g. fast >= slow
            skipped.append((settings, str(e)))
            continue
        bt = simulate(bars, strategy.run(bars), cost_bps, prepared)
        row = dict(settings)
        row.update(segment_stats(bt.returns.iloc[:cut], bt.positions.iloc[:cut], "is", ppy_is))
        # OOS starts flat-or-holding exactly as the strategy was: no fresh entry cost
        row.update(segment_stats(bt.returns.iloc[cut:], bt.positions.iloc[cut:], "oos", ppy_oos))
        rows.append(row)
    table = pd.DataFrame(rows)
    bh = run_backtest(bars["close"], pd.Series(1.0, index=bars.index), cost_bps, prepared)
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


# ---- Walk-forward analysis ---------------------------------------------------------------

WF_MAX_COMBINATIONS = 2000
YEAR = 252


def _metric(R: np.ndarray, H: np.ndarray, metric: str, ppy: float = YEAR) -> np.ndarray:
    """Score every row (one combination) of a slice of daily returns; higher = better."""
    n = R.shape[1]
    if metric == "sharpe":
        sd = R.std(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where(sd > 0, R.mean(axis=1) / sd * np.sqrt(ppy), -np.inf)
        return score
    growth = np.prod(1 + R, axis=1)
    if metric == "final":
        return growth
    if metric == "cagr":
        return growth ** (ppy / n) - 1
    if metric == "max_dd":
        eq = np.cumprod(1 + R, axis=1)
        return (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
    if metric == "trades":                       # fewer is better
        changes = (np.diff(H, axis=1) != 0).sum(axis=1)
        return -changes.astype(float)
    if metric == "prop":                         # return at the size that makes the worst drawdown 5%
        eq = np.cumprod(1 + R, axis=1)
        dd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
        out = np.array([prop_return(R[i], dd[i], ppy) for i in range(len(R))])
        return np.where(np.isfinite(out), out, -np.inf)
    raise ValueError(f"unknown metric {metric!r}")


@dataclass
class WalkForwardResult:
    folds: list                    # one dict per fold: periods, chosen settings, scores
    returns: pd.Series             # stitched out-of-sample daily returns (test windows only)
    held: pd.Series                # position held each day of the stitched run
    benchmark: pd.Series           # buy & hold daily returns, same days
    hindsight: pd.Series           # best single setting over the same days (not knowable!)
    hindsight_settings: dict
    params: list
    combos: int
    skipped: list
    seconds: float


def walk_forward(bars: pd.DataFrame, cls, ranges: dict, train_years: float = 8,
                 test_years: float = 2, anchored: bool = False, rank: str = "sharpe",
                 cost_bps: float = 5.0, costs=None) -> WalkForwardResult:
    """Re-optimize on each training window, trade the winner on the next test window.

    rolling (default): every training window is `train_years` long.
    anchored: training always starts at the first day and grows.
    """
    t0 = time.time()
    from .metrics import bars_per_day
    bpd = bars_per_day(bars.index)
    ppy = YEAR * bpd                         # bars per year (intraday: many per day)
    train, test = int(round(train_years * ppy)), int(round(test_years * ppy))
    if train < max(20, 5 * bpd) or test < max(5, bpd):
        raise ValueError("training needs at least 5 trading days (20 bars) and testing "
                         "at least 1 trading day (5 bars)")
    n = len(bars)
    if train + test > n:
        have, need = n / bpd, (train + test) / bpd
        raise ValueError(f"not enough history: {have:.0f} trading days, but training + testing "
                         f"needs {need:.0f}")
    combos, skipped = [], []
    for settings in grid(cls, ranges):
        try:
            combos.append((settings, cls(**settings)))
        except ValueError as e:
            skipped.append((settings, str(e)))
    if not combos:
        raise ValueError("No valid combinations: " + (skipped[0][1] if skipped else "empty grid"))
    if len(combos) > WF_MAX_COMBINATIONS:
        raise ValueError(f"{len(combos):,} combinations is too many for walk-forward "
                         f"(max {WF_MAX_COMBINATIONS:,}). Use bigger steps.")
    R = np.empty((len(combos), n))
    H = np.empty((len(combos), n))
    prepared = costs.prepare(bars) if costs is not None else None
    for j, (_, strategy) in enumerate(combos):
        bt = simulate(bars, strategy.run(bars), cost_bps, prepared)
        R[j], H[j] = bt.returns.to_numpy(), bt.positions.to_numpy()

    idx = bars.index
    wf_r, wf_h = np.zeros(n), np.zeros(n)
    folds, prev = [], None
    start = train
    while start < n:
        end = min(start + test, n)
        lo = 0 if anchored else start - train
        score = _metric(R[:, lo:start], H[:, lo:start], rank, ppy)
        best = int(np.nanargmax(np.where(np.isfinite(score), score, -np.inf)))
        wf_r[start:end] = R[best, start:end]
        wf_h[start:end] = H[best, start:end]
        if prev is not None and prev != best:      # switching settings costs a trade
            wf_r[start] -= abs(H[best, start - 1] - H[prev, start - 1]) * cost_bps / 10_000
        seg = R[best, start:end]
        folds.append({
            "train_start": idx[lo], "train_end": idx[start - 1],
            "test_start": idx[start], "test_end": idx[end - 1],
            "settings": dict(combos[best][0]),
            "train_score": float(score[best]),
            "train_cagr": float(np.prod(1 + R[best, lo:start]) ** (ppy / (start - lo)) - 1),
            "test_return": float(np.prod(1 + seg) - 1),
            "test_cagr": float(np.prod(1 + seg) ** (ppy / len(seg)) - 1),
            "test_sharpe": float(seg.mean() / seg.std() * np.sqrt(ppy)) if seg.std() > 0 else 0.0,
        })
        prev, start = best, end

    first = train
    span = idx[first:]
    hind = _metric(R[:, first:], H[:, first:], rank, ppy)
    hbest = int(np.nanargmax(np.where(np.isfinite(hind), hind, -np.inf)))
    bh = run_backtest(bars["close"], pd.Series(1.0, index=idx), cost_bps, prepared).returns.iloc[first:]
    return WalkForwardResult(
        folds, pd.Series(wf_r[first:], index=span), pd.Series(wf_h[first:], index=span),
        bh, pd.Series(R[hbest, first:], index=span), dict(combos[hbest][0]),
        list(ranges), len(combos), skipped, time.time() - t0)
