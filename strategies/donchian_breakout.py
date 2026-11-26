"""Donchian breakout: a classic trend-following rule (the 'Turtle traders' entry).

Buy when the close makes a new N-day high; sell when it makes a new M-day low.
Any Strategy class in this folder shows up in the QuantLab app automatically.
"""
import pandas as pd

from quantlab.strategy import Param, Strategy


class Breakout(Strategy):
    name = "Donchian breakout"
    description = ("Trend following: buy when the price closes at a new N-day high, "
                   "sell when it closes at a new M-day low.")
    params = {"entry": Param(55, 5, 250, "Buy on a new high over (days)",
                             help="Buy when the close breaks above the highest high of this many previous bars."),
              "exit": Param(20, 5, 250, "Sell on a new low over (days)",
                            help="Sell when the close breaks below the lowest low of this many previous bars. Shorter = quicker exits."),
              "allow_short": Param(False, label="Also trade breakdowns",
                                   help="Yes: short on a new low over the entry period, cover on a new high over the exit period.")}
    guide = """
## In short
Buy strength: when the price closes at a new N-bar high, a trend may be starting. Get out
when it closes at a new M-bar low.

## The rules
- **Upper channel** = highest high of the previous *entry* bars (not counting today).
- **Lower channel** = lowest low of the previous *exit* bars.
- Close above the upper channel: **buy**. Close below the lower channel: **sell**. Otherwise hold.

## Settings
- 55 / 20 are the "Turtle traders" numbers from the 1980s, made for commodity futures.
- A shorter exit channel locks in gains sooner but gets shaken out more.

## When it tends to work
- Markets with big, sustained trends (commodities, currencies, some momentum stocks).

## When it struggles
- Stock indices that drift up with frequent pullbacks: many false breakouts.

## Watch out
- Many small losses and a few big wins is normal for breakouts: judge the average win vs average loss.
"""

    def compute(self, bars):
        # shift(1): compare today's close with the PREVIOUS N days, not including today
        upper = bars.high.rolling(self.entry).max().shift(1)
        lower = bars.low.rolling(self.exit).min().shift(1)
        self.plot(f"{self.entry}-day high", upper)
        self.plot(f"{self.exit}-day low", lower)
        if self.allow_short:
            return self._long_short(bars)
        signal = pd.Series(float("nan"), index=bars.index)
        signal[bars.close > upper] = 1.0      # breakout: get in
        signal[bars.close < lower] = 0.0      # breakdown: get out
        return signal.ffill().fillna(0.0)     # hold the last decision

    def _long_short(self, bars):
        """Long on an entry-period high, out on an exit-period low; mirror image for shorts."""
        hi_in = bars.high.rolling(self.entry).max().shift(1).to_numpy()
        lo_in = bars.low.rolling(self.entry).min().shift(1).to_numpy()
        hi_out = bars.high.rolling(self.exit).max().shift(1).to_numpy()
        lo_out = bars.low.rolling(self.exit).min().shift(1).to_numpy()
        close = bars.close.to_numpy()
        pos, side = [], 0.0
        for c, a, b, x, y in zip(close, hi_in, lo_in, hi_out, lo_out):
            if side > 0 and c < y:
                side = 0.0
            elif side < 0 and c > x:
                side = 0.0
            if side == 0:
                if c > a:
                    side = 1.0
                elif c < b:
                    side = -1.0
            pos.append(side)
        return pd.Series(pos, index=bars.index)
