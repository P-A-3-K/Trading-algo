"""Prop firm check: would this strategy pass a funded-account challenge, and at what size?

A prop firm challenge, typically:
  * Phase 1: make +10% ...           * Phase 2: then make +5% (on a fresh account)
  * ... without ever losing 10% of the starting balance (max loss), and
  * without losing 5% of the starting balance within one day (daily loss).
Pass both and you trade a "funded" account under the same loss rules.

How the check works:
  1. Take the strategy's own returns, bar by bar, and scale them by a position
     SIZE (2x = twice the position, e.g. with leverage; 0.5x = half).
  2. Build thousands of possible futures by reshuffling blocks of real trading
     days (the same block bootstrap as "Could it be luck?").
  3. Play the challenge rules on every future: count passes, failures and the
     time it took; then see how long a funded account survives.

The intraday low of every day counts for the loss limits (firms watch your equity,
open trades included). The profit target counts at the day's close.

Big caveat: the futures are built from the past. If the past had a strong trend
(gold 2008-2026), the futures assume more of it.
"""
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class Rules:
    target: float = 0.10            # phase 1 profit target (fraction of the start balance)
    target2: float = 0.05           # phase 2 target; 0 = one-step challenge
    max_loss: float = 0.10          # account fails below start x (1 - max_loss) ...
    trailing: bool = False          # ... or below the highest close minus max_loss, until it reaches the start
    daily_loss: float = 0.05        # fails when equity drops this much (of the start balance) within a day
    max_days: int = 0               # time limit per phase in trading days; 0 = none
    min_days: int = 0               # minimum trading days before a pass counts

    def describe(self) -> str:
        t = f"+{self.target:.0%}" + (f" then +{self.target2:.0%}" if self.target2 else "")
        dd = f"{self.max_loss:.0%} {'trailing' if self.trailing else 'max'} loss"
        lim = f", {self.max_days} days per phase" if self.max_days else ""
        return f"{t}, {dd}, {self.daily_loss:.0%} daily loss{lim}"


PRESETS = {
    "two_step": ("Two-step (10% / 5%, 10% max loss, 5% daily)", Rules()),
    "one_step": ("One-step (10%, 6% trailing loss, 3% daily)",
                 Rules(target=0.10, target2=0.0, max_loss=0.06, trailing=True, daily_loss=0.03)),
    "strict": ("Strict two-step (8% / 5%, 8% max loss, 4% daily)",
               Rules(target=0.08, target2=0.05, max_loss=0.08, daily_loss=0.04)),
}
SIZES = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def daily_parts(returns: pd.Series, size: float):
    """Per trading day at a given size: (close-to-close return, lowest point within the day).

    lowest point = worst equity during the day relative to the day's start
    (e.g. -0.03 = 3% under the start of the day at some bar)."""
    r = np.nan_to_num(returns.to_numpy(dtype=float)) * size
    logs = np.log1p(np.maximum(r, -0.999999))
    days = pd.Index(returns.index.date if hasattr(returns.index, "date") else returns.index)
    codes, uniq = pd.factorize(days)
    start = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    cum = np.cumsum(logs)
    before = np.r_[0.0, cum][start]                     # log equity at each day's start
    day_of_bar = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(r)]))
    rel = cum - before[day_of_bar]                      # log equity within the day vs its start
    low = np.minimum.reduceat(rel, start)
    close = np.r_[cum[start[1:] - 1], cum[-1]] - before
    return np.expm1(close), np.expm1(np.minimum(low, 0.0))


def bootstrap_days(n_days_hist: int, sims: int, length: int, block: int, rng) -> np.ndarray:
    """Indices of historical days for `sims` futures of `length` days, in blocks."""
    block = max(1, min(block, n_days_hist))
    n_blocks = -(-length // block)
    starts = rng.integers(0, n_days_hist - block + 1, size=(sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(sims, -1)
    return idx[:, :length]


def run_phase(ret: np.ndarray, low: np.ndarray, rules: Rules, target: float):
    """Play one phase on (sims x days) arrays. Returns (outcome, day): outcome
    1 = target reached, -1 = broke a loss rule, 0 = neither within the days given."""
    sims, days = ret.shape
    close = np.cumprod(1 + ret, axis=1)                 # equity at each day's close
    start = np.concatenate([np.ones((sims, 1)), close[:, :-1]], axis=1)
    lowest = start * (1 + low)                          # worst equity during each day
    if rules.trailing:
        peak = np.maximum.accumulate(np.maximum(start, 1.0), axis=1)
        floor = np.minimum(peak - rules.max_loss, 1.0)  # trails the high, stops at the start balance
    else:
        floor = np.full_like(close, 1.0 - rules.max_loss)
    broke = (lowest <= floor) | (start - lowest >= rules.daily_loss)
    hit = close >= 1 + target
    if rules.min_days:
        hit[:, :rules.min_days - 1] = False
    first = lambda m: np.where(m.any(axis=1), m.argmax(axis=1), days)
    t_hit, t_broke = first(hit), first(broke)
    outcome = np.where(t_broke < t_hit, -1, np.where(t_hit < days, 1, 0))
    when = np.where(outcome == 1, t_hit, np.where(outcome == -1, t_broke, days))
    return outcome, when


def check(returns: pd.Series, rules: Rules = None, sizes=None, sims: int = 2000,
          horizon_days: int = 2 * TRADING_DAYS, funded_days: int = TRADING_DAYS,
          block: int = 10, seed: int = 0) -> dict:
    """Pass / fail odds of the challenge at several position sizes.

    returns: the strategy's returns per bar (any timeframe), with a DatetimeIndex.
    horizon_days: how long we let each phase run when there's no time limit.
    """
    rules = rules or Rules()
    safe_exact = safe_size(returns, rules)
    sizes = sorted(set(list(sizes or SIZES) + ([safe_exact] if safe_exact and not sizes else [])))
    rng = np.random.default_rng(seed)
    n_hist = len(pd.Index(returns.index.date).unique())
    if n_hist < 40:
        raise ValueError(f"Only {n_hist} trading days of results: too few to simulate a challenge.")
    length = rules.max_days or horizon_days
    picks = [bootstrap_days(n_hist, sims, length, block, rng) for _ in range(2)]
    fund_pick = bootstrap_days(n_hist, sims, funded_days, block, rng)
    rows = []
    for size in sizes:
        ret, low = daily_parts(returns, size)
        o1, t1 = run_phase(ret[picks[0]], low[picks[0]], rules, rules.target)
        if rules.target2:
            o2, t2 = run_phase(ret[picks[1]], low[picks[1]], rules, rules.target2)
            passed = (o1 == 1) & (o2 == 1)
            failed = (o1 == -1) | ((o1 == 1) & (o2 == -1))
            t = t1 + t2 + 2
        else:
            passed, failed, t = o1 == 1, o1 == -1, t1 + 1
        # Funded: same loss rules, no target; does it survive a year, and what does it make?
        fo, ft = run_phase(ret[fund_pick], low[fund_pick], rules, target=np.inf)
        survive = fo != -1
        f_eq = np.cumprod(1 + ret[fund_pick], axis=1)[:, -1]
        # The past, at this size: worst drawdown and worst day (history, not simulation)
        worst_dd, _ = history_risk(returns, size)
        rows.append({
            "size": size,
            "pass": float(passed.mean()), "fail": float(failed.mean()),
            "open": float(1 - passed.mean() - failed.mean()),
            "months": float(np.median(t[passed]) / 21) if passed.any() else None,
            "funded_survive": float(survive.mean()),
            "funded_return": float(np.median(f_eq[survive] - 1)) if survive.any() else None,
            "hist_dd": worst_dd, "hist_worst_day": float(low.min()),
            "cagr": float(np.prod(1 + ret) ** (TRADING_DAYS / len(ret)) - 1),
        })
    safe = safe_exact
    return {"rules": asdict(rules), "describe": rules.describe(), "rows": rows, "safe": safe,
            "days": n_hist, "sims": sims, "horizon_days": length, "funded_days": funded_days,
            "block": block}


def history_risk(returns: pd.Series, size: float):
    """(worst drawdown, worst day) the past would have had at this size, from intraday lows."""
    ret, low = daily_parts(returns, size)
    day_start = np.r_[1.0, np.cumprod(1 + ret)[:-1]]
    return float((day_start * (1 + low) / np.maximum.accumulate(day_start) - 1).min()), float(low.min())


def safe_size(returns: pd.Series, rules: Rules, margin: float = 0.5, top: float = 10.0):
    """Largest size (2 decimals) whose PAST worst drawdown and worst day use at most
    `margin` (half) of the limits: the future is usually rougher than the past."""
    def ok(size):
        dd, day = history_risk(returns, size)
        return -dd <= rules.max_loss * margin and -day <= rules.daily_loss * margin
    if not ok(0.01):
        return None
    lo, hi = 0.01, top
    if ok(hi):
        return hi
    for _ in range(30):                        # risk grows with size: bisect
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return math.floor(lo * 100) / 100            # round down: never above the safe level
