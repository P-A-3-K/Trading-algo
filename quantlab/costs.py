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
    slippage_range: float = 0.05      # x typical 5-minute range (longer bars scaled by sqrt(time))
    impact: float = 0.1               # square-root impact coefficient (0 = off)
    account: float = 10_000.0         # account size in $ (for impact)
    bar_spread: bool = True           # use each bar's own spread when the data has one (MT5)
    swap_long_bps: float = None       # cost per night held, bp of the position (None: from the data)
    swap_short_bps: float = None

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
        if self.bar_spread and "spread" in bars.columns:  # the broker's real spread, bar by bar
            real = (bars["spread"].astype(float) / 2 / close * 10_000).to_numpy()   # half-spread, bp
            spread = np.where(np.isfinite(real) & (real > 0), real, spread)
        linear = (self.commission_bps + spread) / 10_000

        # slippage: share of the typical bar range (median of the last 20 bars)
        if {"high", "low"} <= set(bars.columns):
            rng = (bars["high"] - bars["low"]) / close
        else:                                     # close-only data: use moves instead
            rng = close.pct_change().abs() * 2
        typical = rng.rolling(20, min_periods=1).median().fillna(0).to_numpy()
        # An order slips by a share of a 5-minute candle's range whatever the chart's timeframe:
        # longer candles are scaled down by sqrt(time) (a day = 390 regular-hours minutes)
        if intraday and len(idx) > 1:
            minutes = pd.Series(idx[1:] - idx[:-1]).median().total_seconds() / 60
        else:
            minutes = 390.0
        scale = min(1.0, (5.0 / max(minutes, 1e-9)) ** 0.5)
        linear = linear + self.slippage_range * typical * scale

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
        # swaps: cost per night a position is held (CFDs); a weekend counts 3 nights
        sl = self.swap_long_bps if self.swap_long_bps is not None else bars.attrs.get("swap_long_bps")
        ss = self.swap_short_bps if self.swap_short_bps is not None else bars.attrs.get("swap_short_bps")
        nights = nights_between(idx) if (sl or ss) else np.zeros(len(idx))
        return PreparedCosts(np.asarray(linear, dtype=float), np.asarray(k, dtype=float),
                             nights, (sl or 0.0) / 10_000, (ss or 0.0) / 10_000)


def nights_between(index) -> np.ndarray:
    """Rollovers between each bar and the one before it (CFD swaps are charged at
    17:00 New York; a weekend counts 3 nights). Daily bars: calendar days between them."""
    n = len(index)
    if n < 2 or not isinstance(index, pd.DatetimeIndex):
        return np.zeros(n)
    if index.tz is not None:                         # intraday: the trading day starts at 17:00 NY
        day = (index.tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)).normalize()
    else:
        day = index.normalize()
    d = np.diff(day.to_numpy()).astype("timedelta64[D]").astype(float)
    return np.r_[0.0, np.maximum(d, 0.0)]


@dataclass
class PreparedCosts:
    linear: np.ndarray     # cost per unit traded, per bar (fraction)
    impact: np.ndarray     # impact factor per bar: cost = impact x size^1.5
    nights: np.ndarray = None   # rollovers between bar i-1 and bar i
    swap_long: float = 0.0      # cost per night per unit long (fraction)
    swap_short: float = 0.0     # ... per unit short

    def __call__(self, i, size):
        """Cost (fraction of the account) of trading `size` (0-1+) on bar i."""
        size = np.abs(size)
        return size * self.linear[i] + self.impact[i] * size ** 1.5

    def holding(self, held_before: np.ndarray) -> np.ndarray:
        """Swap cost on each bar for the position carried into it from the bar before."""
        if self.nights is None or not (self.swap_long or self.swap_short):
            return np.zeros(len(held_before))
        h = np.nan_to_num(np.asarray(held_before, dtype=float))
        rate = np.where(h > 0, self.swap_long, np.where(h < 0, self.swap_short, 0.0))
        return np.abs(h) * rate * self.nights


def flat(cost_bps: float) -> CostModel:
    """The old simple model: a fixed cost per trade, nothing else."""
    return CostModel(commission_bps=0, spread_bps=cost_bps, extended_spread_bps=cost_bps,
                     slippage_range=0, impact=0, bar_spread=False, swap_long_bps=0.0, swap_short_bps=0.0)


PRESETS = {
    "realistic": CostModel(),
    "flat5": flat(5.0),
    "none": flat(0.0),
    "expensive": CostModel(commission_bps=2, spread_bps=10, extended_spread_bps=40,
                           slippage_range=0.1, impact=0.2),
}
