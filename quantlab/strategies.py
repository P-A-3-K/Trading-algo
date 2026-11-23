"""Built-in strategies, written as Strategy plug-ins (see quantlab/strategy.py).

The plain functions at the bottom keep lessons 1 and 2 working.
"""
import pandas as pd

from .strategy import Param, Strategy


class BuyAndHold(Strategy):
    name = "Buy & hold"
    description = "Always fully invested. The benchmark every strategy must beat."
    params = {}
    guide = """
## In short
Buy on the first bar and never sell. No skill, no decisions.

## Why it's here
Every other strategy is compared with it. Beating buy & hold **after costs**, with a smaller
drawdown or a better Sharpe, is the bar a strategy must clear to be worth the effort.

## Watch out
- Over long periods stock indices have risen, so buy & hold is hard to beat on return.
- Its weak spot is pain: worst drawdowns of -50% or more (2000-2002, 2008) are normal.
"""

    def compute(self, bars):
        return pd.Series(1.0, index=bars.index)


class SmaCrossover(Strategy):
    name = "SMA crossover"
    description = ("Trend following: long when the fast moving average is above "
                   "the slow one, in cash otherwise.")
    params = {"fast": Param(50, 2, 400, "Fast average (days)",
                            help="Bars in the quick moving average. Shorter = reacts sooner, more trades and false signals."),
              "slow": Param(200, 5, 500, "Slow average (days)",
                            help="Bars in the slow moving average, the 'trend' line. Must be longer than the fast one.")}
    guide = """
## In short
Follow the trend: be invested while the short-term average price is above the long-term one.

## The rules
- **Fast average** = mean of the last *fast* closes; **slow average** = mean of the last *slow* closes.
- Be **long** while fast > slow ("golden cross"), in **cash** while fast < slow ("death cross").
- Decided at each bar's close, traded from the next bar.

## Settings
- 50 / 200 on daily bars is the famous pair. On intraday bars the numbers count bars, not days.
- Closer averages (e.g. 20 / 50) trade more often and react faster.

## When it tends to work
- Long, persistent trends and slow bear markets: it sat out most of 2008.

## When it struggles
- Sideways markets ("whipsaw"): repeated buy-high, sell-low crossings.
- Fast crashes and rebounds (2020): it sells near the bottom and buys back higher.

## Watch out
- Its main benefit is usually a **smaller drawdown**, not a higher return. Judge it on both.
"""

    def validate(self):
        if self.fast >= self.slow:
            raise ValueError("fast must be shorter than slow")

    def compute(self, bars):
        fast = bars.close.rolling(self.fast).mean()
        slow = bars.close.rolling(self.slow).mean()
        self.plot(f"SMA {self.fast}", fast)
        self.plot(f"SMA {self.slow}", slow)
        return (fast > slow).astype(float)


class MeanReversion(Strategy):
    name = "Mean reversion (z-score)"
    description = ("Buy when the price is unusually far below its recent average "
                   "(z-score under -entry), in cash otherwise.")
    params = {"lookback": Param(20, 5, 250, "Average over (days)",
                                help="Bars used for the average and the standard deviation of the price."),
              "entry": Param(1.0, 0.1, 4.0, "Entry z-score", step=0.1,
                             help="How many standard deviations below the average the price must be to buy. Higher = rarer, deeper dips.")}
    guide = """
## In short
Buy when the price is unusually far **below its recent average**, betting it drifts back.

## The rules
- **z-score** = (close - average) / standard deviation, both over the last *lookback* bars.
- Be **long** while the z-score is below -*entry* (e.g. -1.0), in **cash** otherwise.
- So it exits as soon as the price is no longer stretched.

## Settings
- **Average over**: the window that defines "normal". 20 bars is about a month of daily data.
- **Entry z-score**: 1.0 trades often; 2.0 only on sharp drops.

## When it tends to work
- Assets that oscillate around a level, and short-term overreactions.

## When it struggles
- Strong trends: a falling stock stays "cheap" and keeps falling. No stop-loss.
- It's invested only a small part of the time, so even a real edge earns little per year.

## Watch out
- Lower panel: the z-score and the entry level (dashed).
"""

    def compute(self, bars):
        mean = bars.close.rolling(self.lookback).mean()
        std = bars.close.rolling(self.lookback).std()
        z = (bars.close - mean) / std
        self.plot(f"SMA {self.lookback}", mean)
        self.plot("Z-score", z, pane="lower")
        self.plot("Entry level", pd.Series(-self.entry, index=bars.index), pane="lower")
        return (z < -self.entry).astype(float)


BUILTIN = [BuyAndHold, SmaCrossover, MeanReversion]


def available() -> dict:
    """Every strategy that can run: the built-ins plus strategies/*.py files
    (see quantlab/discovery.py). Blocked ones are left out."""
    from . import discovery
    found = discovery.scan(builtin={c.__name__ for c in BUILTIN})
    out = {cls.__name__: cls for cls in BUILTIN}
    out.update({sid: f.cls for sid, f in found.strategies.items() if not f.problem})
    return out


def catalog():
    """For the app's menus: (built-in classes, discovery Scan with files and problems)."""
    from . import discovery
    return BUILTIN, discovery.scan(builtin={c.__name__ for c in BUILTIN})


def get(name: str):
    """Find a strategy class by name (case-insensitive)."""
    from . import discovery
    builtin, found = catalog()
    for sid, f in found.strategies.items():
        if sid.lower() == str(name).lower() and f.problem:
            raise ValueError(f"{sid} ({f.file}) is blocked: {f.problem}.")
    by_lower = {k.lower(): v for k, v in available().items()}
    cls = by_lower.get(str(name).lower())
    if cls is None:
        raise ValueError(f"Unknown strategy {name!r}. Choose from: " + ", ".join(available()))
    return cls


def _bars(prices: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"close": prices})


def buy_and_hold(prices: pd.Series) -> pd.Series:
    return BuyAndHold().run(_bars(prices)).positions


def sma_crossover(prices: pd.Series, fast=50, slow=200) -> pd.Series:
    return SmaCrossover(fast=fast, slow=slow).run(_bars(prices)).positions


def mean_reversion(prices: pd.Series, lookback=20, entry_z=1.0) -> pd.Series:
    return MeanReversion(lookback=lookback, entry=entry_z).run(_bars(prices)).positions
