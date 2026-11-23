"""MACD crossover: a classic trend-following signal."""
import pandas as pd

from quantlab.strategy import Param, Strategy


class MacdCrossover(Strategy):
    name = "MACD crossover"
    description = ("Trend following: long while the MACD line is above its signal line "
                   "(optionally short while it's below).")
    params = {
        "fast": Param(12, 2, 100, "Fast EMA (bars)",
                      help="Length of the quick exponential moving average. Smaller = reacts sooner, more false signals."),
        "slow": Param(26, 5, 300, "Slow EMA (bars)",
                      help="Length of the slow exponential moving average. Must be longer than the fast one."),
        "signal": Param(9, 2, 100, "Signal line (bars)",
                        help="Smoothing of the MACD line. The trade happens when MACD crosses this line."),
        "allow_short": Param(False, label="Also go short",
                             help="Yes: hold a short position while MACD is below its signal line. "
                                  "No: sit in cash instead."),
    }
    guide = """
## In short
MACD measures **momentum**: the gap between a fast and a slow moving average of the price.
When that gap starts widening upwards, a new up-trend may be starting.

## The rules
- **MACD line** = fast EMA - slow EMA (EMA = exponential moving average, which weighs recent bars more).
- **Signal line** = an EMA of the MACD line itself.
- Be **long** while MACD is above the signal line.
- While it's below: **cash**, or **short** if "Also go short" is Yes.
- Decisions are made at each bar's close and traded from the next bar.

## Settings
- **Fast / Slow EMA**: the classic 12 / 26 comes from daily charts in the 1970s. On 5-minute bars the same numbers cover about 1 and 2 hours.
- **Signal line**: 9 is standard. Shorter = earlier but jumpier signals.

## When it tends to work
- Markets that trend for weeks or months, where a late entry still catches most of the move.
- Assets with long, smooth trends (indices, commodities in a cycle).

## When it struggles
- **Sideways, choppy markets**: MACD flips back and forth around the signal line, and every flip costs money (watch "Costs paid").
- Sharp V-shaped reversals: the signal arrives after the move.

## Watch out
- MACD is built from moving averages, so it's always **late**: compare the trade arrows with the turning points on the chart.
- Lower panel: MACD, signal line and the zero line. The histogram many charts show is just MACD minus signal.
"""

    def validate(self):
        if self.fast >= self.slow:
            raise ValueError("fast EMA must be shorter than slow EMA")

    def compute(self, bars):
        ema = lambda x, n: x.ewm(span=n, adjust=False).mean()
        macd = ema(bars.close, self.fast) - ema(bars.close, self.slow)
        signal = ema(macd, self.signal)
        warm = pd.Series(range(len(bars)), index=bars.index) < self.slow   # until the slow EMA has data
        self.plot("MACD", macd.mask(warm), pane="lower")
        self.plot("Signal", signal.mask(warm), pane="lower")
        self.plot("Zero level", pd.Series(0.0, index=bars.index), pane="lower")
        above = (macd > signal).astype(float)
        pos = above if not self.allow_short else above * 2 - 1
        return pos.mask(warm, 0.0)
