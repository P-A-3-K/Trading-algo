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
    params = {"entry": Param(55, 5, 250, "Buy on a new high over (days)"),
              "exit": Param(20, 5, 250, "Sell on a new low over (days)")}

    def compute(self, bars):
        # shift(1): compare today's close with the PREVIOUS N days, not including today
        upper = bars.high.rolling(self.entry).max().shift(1)
        lower = bars.low.rolling(self.exit).min().shift(1)
        self.plot(f"{self.entry}-day high", upper)
        self.plot(f"{self.exit}-day low", lower)
        signal = pd.Series(float("nan"), index=bars.index)
        signal[bars.close > upper] = 1.0      # breakout: get in
        signal[bars.close < lower] = 0.0      # breakdown: get out
        return signal.ffill().fillna(0.0)     # hold the last decision
