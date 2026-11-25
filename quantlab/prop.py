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
    # Public rules of two firms we looked at (Nov 2026). Firms change rules: check theirs.
    "ftmo_2step": ("FTMO-style 2-Step (10% / 5%, 10% max, 5% daily, min 4 days)",
                   Rules(target=0.10, target2=0.05, max_loss=0.10, daily_loss=0.05, min_days=4)),
    "the5ers_hs": ("The5ers High Stakes-style (8% / 5%, 10% max, 5% daily, min 3 days)",
                   Rules(target=0.08, target2=0.05, max_loss=0.10, daily_loss=0.05, min_days=3)),
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


@dataclass
class DailyResults:
    """Results already per trading day: close-to-close return and the day's lowest
    point (both as fractions), e.g. a combined portfolio. A size scales them as if
    the position were rebalanced to that size every day."""
    ret: pd.Series
    low: pd.Series


def parts_of(source):
    """function(size) -> (daily return, daily low) arrays, for bar returns or DailyResults."""
    if isinstance(source, DailyResults):
        r, lo = source.ret.to_numpy(dtype=float), source.low.to_numpy(dtype=float)
        return lambda size: (np.maximum(size * r, -0.999999), np.minimum(np.maximum(size * lo, -0.999999), 0.0))
    return lambda size: daily_parts(source, size)


def daily_results(returns: pd.Series) -> DailyResults:
    """Bar returns -> DailyResults indexed by trading date."""
    ret, low = daily_parts(returns, 1.0)
    days = pd.Index(returns.index.date).unique() if isinstance(returns.index, pd.DatetimeIndex) else returns.index
    idx = pd.DatetimeIndex(days)
    return DailyResults(pd.Series(ret, index=idx), pd.Series(low, index=idx))


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


def check(returns, rules: Rules = None, sizes=None, sims: int = 2000,
          horizon_days: int = 2 * TRADING_DAYS, funded_days: int = TRADING_DAYS,
          block: int = 10, seed: int = 0) -> dict:
    """Pass / fail odds of the challenge at several position sizes.

    returns: the strategy's returns per bar (any timeframe), with a DatetimeIndex,
             or DailyResults (e.g. a combined portfolio).
    horizon_days: how long we let each phase run when there's no time limit.
    """
    rules = rules or Rules()
    parts = parts_of(returns)
    safe_exact = safe_size(returns, rules)
    sizes = sorted(set(list(sizes or SIZES) + ([safe_exact] if safe_exact and not sizes else [])))
    rng = np.random.default_rng(seed)
    n_hist = len(parts(1.0)[0])
    if n_hist < 40:
        raise ValueError(f"Only {n_hist} trading days of results: too few to simulate a challenge.")
    length = rules.max_days or horizon_days
    picks = [bootstrap_days(n_hist, sims, length, block, rng) for _ in range(2)]
    fund_pick = bootstrap_days(n_hist, sims, funded_days, block, rng)
    rows = []
    for size in sizes:
        ret, low = parts(size)
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


def history_risk(returns, size: float):
    """(worst drawdown, worst day) the past would have had at this size, from intraday lows."""
    ret, low = parts_of(returns)(size)
    day_start = np.r_[1.0, np.cumprod(1 + ret)[:-1]]
    return float((day_start * (1 + low) / np.maximum.accumulate(day_start) - 1).min()), float(low.min())


def safe_size(returns, rules: Rules, margin: float = 0.5, top: float = 10.0):
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


# ---- Challenge plan: the whole journey, retries and payouts included ---------------------

@dataclass
class Plan:
    """How to size the account at each stage (sizes are multiples of the strategy's position)."""
    label: str
    challenge_size: float = 1.0
    funded_size: float = 1.0
    cushion_size: float = None      # step down to this size in the challenge ...
    cushion_dd: float = 0.04        # ... while equity is this far below the phase's start


DEFAULT_PLANS = [
    Plan("Safe size all the way", None, None),
    Plan("1x all the way", 1.0, 1.0),
    Plan("1.5x challenge, then 1x", 1.5, 1.0),
    Plan("2x challenge, then 1x", 2.0, 1.0),
    Plan("2x with cushion (1x after -4%), then 1x", 2.0, 1.0, cushion_size=1.0, cushion_dd=0.04),
    Plan("3x challenge, then 1x", 3.0, 1.0),
    Plan("2x challenge, then safe size", 2.0, None),
]


def simulate_plan(ret: np.ndarray, low: np.ndarray, rules: Rules, plan: Plan, days: int,
                  fee: float, account: float, split: float, refund: bool, payout_days: int = 21):
    """Play the journey on (sims x days) arrays of daily return / daily low at size 1.

    Returns per-simulation arrays: first funded day (or -1), attempts bought, fees paid,
    payouts received, and whether the funded account was lost.
    """
    sims = ret.shape[0]
    phase = np.ones(sims, dtype=int)                # 1, 2 = challenge phases, 3 = funded
    eq = np.ones(sims)                              # equity relative to the stage's start
    peak = np.ones(sims)
    n_days = np.zeros(sims, dtype=int)              # trading days in the current stage
    attempts = np.ones(sims)
    fees = np.full(sims, float(fee))
    paid = np.zeros(sims)
    refunded = np.zeros(sims, dtype=bool)
    funded_at = np.full(sims, -1)
    lost = np.zeros(sims, dtype=bool)
    target = np.where(phase == 1, rules.target, rules.target2)
    for t in range(days):
        in_ch = phase < 3
        size = np.where(in_ch, plan.challenge_size, plan.funded_size)
        if plan.cushion_size is not None:
            size = np.where(in_ch & (eq <= 1 - plan.cushion_dd), plan.cushion_size, size)
        r = np.maximum(size * ret[:, t], -0.999999)
        lo = np.minimum(np.maximum(size * low[:, t], -0.999999), 0.0)
        start = eq
        lowest = start * (1 + lo)
        eq = start * (1 + r)
        n_days += 1
        if rules.trailing:
            floor = np.minimum(peak - rules.max_loss, 1.0)
        else:
            floor = 1.0 - rules.max_loss
        broke = (lowest <= floor) | (start - lowest >= rules.daily_loss)
        peak = np.maximum(peak, eq)
        target = np.where(phase == 1, rules.target, rules.target2)
        timeout = in_ch & (rules.max_days > 0) & (n_days >= rules.max_days)
        passed = in_ch & ~broke & (eq >= 1 + target) & (n_days >= max(rules.min_days, 1))
        # Funded: monthly payout of the profit (account resets to its start)
        fund = phase == 3
        pay_now = fund & ~broke & (n_days % payout_days == 0) & (eq > 1)
        if pay_now.any():
            paid += np.where(pay_now, (eq - 1) * account * split, 0.0)
            if refund:
                give = pay_now & ~refunded
                paid += np.where(give, fee, 0.0)
                refunded |= give
            eq = np.where(pay_now, 1.0, eq)
            peak = np.where(pay_now, 1.0, peak)
        # Failures: a challenge is bought again; a lost funded account too
        fail = (in_ch & (broke | timeout)) | (fund & broke)
        lost |= fund & broke
        attempts += fail
        fees += np.where(fail, fee, 0.0)
        # Phase changes
        to_phase2 = passed & (phase == 1) & (rules.target2 > 0)
        to_funded = passed & ((phase == 2) | ((phase == 1) & (rules.target2 <= 0)))
        funded_at = np.where(to_funded & (funded_at < 0), t, funded_at)
        phase = np.where(fail, 1, np.where(to_phase2, 2, np.where(to_funded, 3, phase)))
        reset = fail | to_phase2 | to_funded
        eq = np.where(reset, 1.0, eq)
        peak = np.where(reset, 1.0, peak)
        n_days = np.where(reset, 0, n_days)
    return funded_at, attempts, fees, paid, lost


def plans(returns, rules: Rules = None, plan_list=None, months: int = 36, sims: int = 2000,
          fee: float = 540.0, account: float = 100_000.0, split: float = 0.8, refund: bool = True,
          block: int = 10, seed: int = 1) -> dict:
    """Compare sizing plans over the whole journey (challenge -> funded), retries included.

    Every simulated future is `months` long; when a challenge (or a funded account) fails,
    a new challenge is bought straight away. Payouts: monthly, profit x split; the fee is
    refunded with the first payout when `refund` is True.
    """
    rules = rules or Rules()
    plan_list = plan_list or DEFAULT_PLANS
    parts = parts_of(returns)
    r1, l1 = parts(1.0)
    n_hist = len(r1)
    if n_hist < 40:
        raise ValueError(f"Only {n_hist} trading days of results: too few to simulate a challenge.")
    days = int(months * 21)
    rng = np.random.default_rng(seed)
    pick = bootstrap_days(n_hist, sims, days, block, rng)
    R, L = r1[pick], l1[pick]
    safe = safe_size(returns, rules)
    rows = []
    for p in plan_list:
        p = Plan(p.label, p.challenge_size if p.challenge_size is not None else (safe or 0.1),
                 p.funded_size if p.funded_size is not None else (safe or 0.1), p.cushion_size, p.cushion_dd)
        funded_at, attempts, fees, paid, lost = simulate_plan(R, L, rules, p, days, fee, account, split, refund)
        got = funded_at >= 0
        within = lambda m: float((got & (funded_at < m * 21)).mean())
        rows.append({
            "label": p.label, "challenge_size": round(p.challenge_size, 2), "funded_size": round(p.funded_size, 2),
            "cushion": p.cushion_size is not None,
            "funded_6": within(6), "funded_12": within(12), "funded_24": within(24), "funded_ever": float(got.mean()),
            "median_months": float(np.median(funded_at[got]) / 21) if got.any() else None,
            "attempts": float(attempts.mean()), "fees": float(fees.mean()),
            "payouts": float(paid.mean()), "net": float((paid - fees).mean()),
            "net_p10": float(np.percentile(paid - fees, 10)), "net_p90": float(np.percentile(paid - fees, 90)),
            "lost_funded": float(lost.mean()),
        })
    return {"rules": asdict(rules), "describe": rules.describe(), "rows": rows, "safe": safe,
            "months": months, "sims": sims, "fee": fee, "account": account, "split": split,
            "refund": refund, "days": n_hist, "block": block}
