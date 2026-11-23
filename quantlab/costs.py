"""Trading costs: what each trade really costs, beyond the price chart.

    cost of a trade (as a fraction of the account) =
        size x (commission + spread + slippage)          <- linear in size
      + impact x daily_vol x sqrt(order $ / daily $ volume) x size   <- grows faster

  * commission: broker fee, in basis points (1 bp = 0.01%)
  * spread: the half bid/ask spread you cross. Wider outside regular hours
    (pre-market / after-hours trading is thin)
  * slippage: the price moving against you while you fill; scales with how
    volatile the stock is right now: a share of its typical bar range
  * market impact: bigger orders push the price. The standard "square-root
    law": cost ~ volatility x sqrt(your order / typical daily dollar volume)

Costs are charged on the bar where the trade happens.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    commission_bps: float = 0.0       # broker fee per trade
    spread_bps: float = 1.0           # half-spread, regular hours (9:30-16:00)
    extended_spread_bps: float = 10.0 # half-spread, pre-market / after-hours
    slippage_range: float = 0.05      # x typical bar range (0.05 = 5% of it)
    impact: float = 0.1               # square-root impact coefficient (0 = off)
    account: float = 10_000.0         # account size in $ (for impact)

    def describe(self) -> str:
        parts = [f"spread {self.spread_bps:g} bp" + (f" ({self.extended_spread_bps:g} bp outside 9:30-16:00)"
                                                    if self.extended_spread_bps != self.spread_bps else "")]
        if self.commission_bps:
            parts.insert(0, f"commission {self.commission_bps:g} bp")
        if self.slippage_range:
            parts.append(f"slippage {self.slippage_range * 100:g}% of bar range")
        if self.impact:
            parts.append(f"impact on ${self.account:,.0f}")
        return ", ".join(parts)

    def as_dict(self) -> dict:
        return asdict(self)

    def prepare(self, bars: pd.DataFrame) -> "PreparedCosts":
        """Per-bar cost rates for these bars."""
        from .strategy import regular_hours
        idx = bars.index
        close = bars["close"].astype(float)
        intraday = isinstance(idx, pd.DatetimeIndex) and idx.tz is not None

        # spread: wider outside regular hours (daily bars count as regular)
        rth = regular_hours(idx) if intraday else np.ones(len(idx), dtype=bool)
        spread = np.where(rth, self.spread_bps, self.extended_spread_bps)
        linear = (self.commission_bps + spread) / 10_000

        # slippage: share of the typical bar range (median of the last 20 bars)
        if {"high", "low"} <= set(bars.columns):
            rng = (bars["high"] - bars["low"]) / close
        else:                                     # close-only data: use moves instead
            rng = close.pct_change().abs() * 2
        typical = rng.rolling(20, min_periods=1).median().fillna(0).to_numpy()
        linear = linear + self.slippage_range * typical

        # impact: volatility x sqrt(order / average daily dollar volume)
        k = np.zeros(len(idx))
        if self.impact and "volume" in bars.columns and self.account > 0:
            day = pd.Index(idx.date) if intraday else pd.Index(idx)
            dollar = (close * bars["volume"]).groupby(day).sum()
            daily_close = close.groupby(day).last()
            adv = dollar.rolling(20, min_periods=1).mean()
            vol = daily_close.pct_change().rolling(20, min_periods=2).std().bfill().fillna(0.02)
            per_day = (self.impact * vol * np.sqrt(self.account / adv.replace(0, np.nan))).fillna(0)
            k = per_day.reindex(day).to_numpy()
        return PreparedCosts(np.asarray(linear, dtype=float), np.asarray(k, dtype=float))


@dataclass
class PreparedCosts:
    linear: np.ndarray     # cost per unit traded, per bar (fraction)
    impact: np.ndarray     # impact factor per bar: cost = impact x size^1.5

    def __call__(self, i, size):
        """Cost (fraction of the account) of trading `size` (0-1+) on bar i."""
        size = np.abs(size)
        return size * self.linear[i] + self.impact[i] * size ** 1.5


def flat(cost_bps: float) -> CostModel:
    """The old simple model: a fixed cost per trade, nothing else."""
    return CostModel(commission_bps=0, spread_bps=cost_bps, extended_spread_bps=cost_bps,
                     slippage_range=0, impact=0)


PRESETS = {
    "realistic": CostModel(),
    "flat5": flat(5.0),
    "none": flat(0.0),
    "expensive": CostModel(commission_bps=2, spread_bps=10, extended_spread_bps=40,
                           slippage_range=0.1, impact=0.2),
}
