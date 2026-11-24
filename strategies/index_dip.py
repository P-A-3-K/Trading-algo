"""Index dip-buying: buy a short-term dip in an up-trend, sell the bounce.

Two well-known signals for stock indices (SPY, QQQ, IWM, DIA), on daily candles:
  * IBS (internal bar strength) = (close - low) / (high - low): where the close sits in
    today's range. Below 0.2 = closed near the low of the day.
  * RSI(2): a 2-day RSI. Below 10 = two hard down days.
Trend condition: only while the close is above its 200-day average (no dip-buying in
bear markets: the classic way these strategies avoid crashes).
Exit: the close rises above a short average (5 days by default), or after max_hold days.
Decided at the close, so the trade is "at the close" (the backtester holds it from the
next bar on).
"""
import pandas as pd

from quantlab.indicators import rsi, sma
from quantlab.strategy import Param, Strategy


class IndexDip(Strategy):
    name = "Index dip (IBS / RSI2, trend filter)"
    description = ("Buy short-term dips in stock indices while above the 200-day average; sell when "
                   "the close gets back above a short average.")
    params = {
        "signal": Param("ibs", choices=("ibs", "rsi2"), label="Dip signal",
                        help="ibs: today closed near its low. rsi2: a 2-day RSI is very low (two hard down days)."),
        "ibs_below": Param(0.2, 0.05, 0.5, "IBS below", step=0.05,
                           help="IBS = (close - low) / (high - low). 0.2 = closed in the bottom 20% of the day's range."),
        "rsi_below": Param(10, 2, 40, "RSI(2) below",
                           help="With the rsi2 signal: buy when the 2-day RSI is below this."),
        "trend_len": Param(200, 0, 400, "Trend average (days, 0 = off)",
                           help="Only buy while the close is above this average. 0 turns the trend filter off."),
        "exit_len": Param(5, 2, 50, "Exit when close above average of",
                          help="Sell when the close rises above this short moving average (the bounce)."),
        "max_hold": Param(10, 0, 60, "Max days in a trade (0 = no limit)",
                          help="Sell anyway after this many bars, so a dip that keeps falling doesn't stay forever."),
    }
    guide = """
## In short
Stock indices have a long-documented habit: after a sharp short-term drop, they tend to **bounce
within days**, as long as the bigger trend is up. This strategy buys those dips at the close and
sells into the bounce.

## The rules
- **Trend**: only while today's close is above its *Trend average* (200 days). Set 0 to turn it off.
- **Dip** (pick one):
  - **IBS** = (close - low) / (high - low): where today closed within its range. Buy when below *IBS below* (0.2).
  - **RSI(2)**: a 2-day RSI. Buy when below *RSI(2) below* (10).
- **Exit**: when the close rises above its *Exit* average (5 days), or after *Max days* (10).
- Long only. Trades happen at the close.

## When it tends to work
- Broad stock indices in bull markets: SPY, QQQ, IWM, DIA. It's in the market only a fraction of the time.
- In a portfolio next to trend-following legs: different timing, low correlation.

## When it struggles
- Sharp bear markets: the trend filter keeps it out of most, but the first drop below the average hurts.
- Single stocks and commodities: the edge is much weaker outside broad indices.
- The edge has shrunk since it became popular: judge it on the test period, not the 1990s.

## Watch out
- Many short trades: costs matter. Compare the before/after-cost returns.
- Lower panel: the dip signal and its level.
"""

    def compute(self, bars):
        close = bars.close
        rng = (bars.high - bars.low).where(lambda x: x > 0)
        if self.signal == "ibs":
            sig, level = (close - bars.low) / rng, self.ibs_below
            self.plot("IBS", sig, pane="lower")
        else:
            sig, level = rsi(close, 2), self.rsi_below
            self.plot("RSI 2", sig, pane="lower")
        self.plot("Buy level", pd.Series(float(level), index=bars.index), pane="lower")
        up = close > sma(close, self.trend_len) if self.trend_len else pd.Series(True, index=bars.index)
        exit_avg = sma(close, self.exit_len)
        if self.trend_len:
            self.plot(f"SMA {self.trend_len}", sma(close, self.trend_len))
        self.plot(f"Exit SMA {self.exit_len}", exit_avg)

        entry = ((sig < level) & up).tolist()
        bounce = (close > exit_avg).tolist()
        pos, state, held = [], 0.0, 0
        for e, b in zip(entry, bounce):
            if state:
                held += 1
                if b or (self.max_hold and held >= self.max_hold):
                    state, held = 0.0, 0
            elif e:
                state, held = 1.0, 0
            pos.append(state)
        return pd.Series(pos, index=bars.index)
