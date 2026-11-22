"""Built-in strategies, written as Strategy plug-ins (see quantlab/strategy.py).

The plain functions at the bottom keep lessons 1 and 2 working.
"""
import pandas as pd

from .strategy import Param, Strategy


class BuyAndHold(Strategy):
    name = "Buy & hold"
    description = "Always fully invested. The benchmark every strategy must beat."
    params = {}

    def compute(self, bars):
        return pd.Series(1.0, index=bars.index)


class SmaCrossover(Strategy):
    name = "SMA crossover"
    description = ("Trend following: long when the fast moving average is above "
                   "the slow one, in cash otherwise.")
    params = {"fast": Param(50, 2, 400, "Fast average (days)"),
              "slow": Param(200, 5, 500, "Slow average (days)")}

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
    params = {"lookback": Param(20, 5, 250, "Average over (days)"),
              "entry": Param(1.0, 0.1, 4.0, "Entry z-score", step=0.1)}

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
    """All strategies the app can run, by class name."""
    return {cls.__name__: cls for cls in BUILTIN}


def get(name: str):
    """Find a strategy class by name (case-insensitive)."""
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
