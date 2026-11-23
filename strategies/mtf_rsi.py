"""Multi-timeframe RSI: buy dips on the chart's timeframe, but only when a higher
timeframe's RSI says the bigger trend is up (and the mirror image for shorts).

Example on 5-minute candles with the 4-hour filter:
  * 4h RSI above 50  -> the bigger picture is up: longs allowed
  * 5m RSI below 30  -> a short-term dip: buy it
  * sell when 5m RSI recovers above 55, or when the 4h RSI drops back below 50

The 4h RSI comes from CLOSED 4-hour candles only (see quantlab/mtf.py): at 09:00 the
strategy knows the 04:00-08:00 candle, not the one that finishes at 12:00.
"""
import pandas as pd

from quantlab.indicators import rsi
from quantlab.strategy import Param, Strategy

TIMEFRAMES = ("15m", "30m", "1h", "4h", "1d")


class MtfRsi(Strategy):
    name = "MTF RSI (trend + dip)"
    description = ("Buy short-term RSI dips only while a higher timeframe's RSI is above the trend "
                   "level; exit on the bounce or when the higher trend turns.")
    params = {
        "htf": Param("4h", choices=TIMEFRAMES, label="Higher timeframe",
                     help="The timeframe that sets the direction. Pick one higher than the chart's "
                          "(e.g. 4h on a 5m or 15m chart). Built from the chart's own candles."),
        "htf_period": Param(14, 2, 100, "Higher RSI length (bars)",
                            help="RSI length on the higher timeframe, in its own candles (14 x 4h = about 2 weeks)."),
        "trend_level": Param(50, 30, 70, "Trend: higher RSI above",
                             help="Longs only while the higher-timeframe RSI is above this (shorts: below 100 minus it). "
                                  "50 = 'more up than down lately'."),
        "period": Param(14, 2, 100, "RSI length (bars)",
                        help="RSI length on the chart's own timeframe: this one times the entries."),
        "buy_below": Param(30, 5, 50, "Buy when RSI below",
                           help="Dip level on the chart's timeframe."),
        "sell_above": Param(55, 30, 95, "Sell when RSI above",
                            help="Bounce level: take profit when the short-term RSI recovers above this."),
        "allow_short": Param(False, label="Also go short",
                             help="Yes: also short overbought spikes (RSI above 100 - 'buy below') while the higher "
                                  "RSI is below 100 - 'trend level'."),
    }
    guide = """
## In short
**Multi-timeframe analysis**: let a higher timeframe decide the direction, and the chart's own
timeframe decide the moment. Here both use RSI: the higher one as a trend filter, the lower one to
buy short-term dips *in* that trend instead of against it.

## The rules
- **Trend** (higher timeframe, e.g. 4h): up while its RSI is above *Trend level* (50).
- **Entry** (chart's timeframe): buy at the close of a bar whose RSI is below *Buy when RSI below* (30), if the trend is up.
- **Exit**: when the chart's RSI recovers above *Sell when RSI above* (55), **or** the higher RSI falls back below the trend level.
- **Shorts** (optional): the mirror image: higher RSI below 100 - trend level, chart RSI above 100 - buy level.
- No peeking: the higher-timeframe RSI only uses **finished** candles (at 09:00 you know the 04:00-08:00 4h candle, not the one ending at 12:00).

## Settings
- **Higher timeframe**: must be higher than the chart's (4h on a 5m/15m chart, 1d on a 1h chart). The optimizer can test several.
- **Trend level**: 50 is neutral; 55-60 asks for a clearer trend (fewer trades).

## When it tends to work
- Markets that trend for days/weeks with short pullbacks: the filter avoids buying dips in a falling market.

## When it struggles
- Choppy markets where the higher RSI flips around 50: the filter adds little and exits early.
- Costs on 5m/15m charts: compare "Return per year before costs" with the after-cost number.

## Watch out
- Compare with plain **RSI reversion** on the same market (same RSI settings): if MTF doesn't beat it out-of-sample, the filter isn't helping.
- Lower panel: both RSIs and the levels. The higher RSI moves in steps: one step per finished higher candle.
"""

    def validate(self):
        if self.buy_below >= self.sell_above:
            raise ValueError("'buy below' must be lower than 'sell above'")

    def compute(self, bars):
        fast = rsi(bars.close, self.period)
        slow = self.on_higher(bars, self.htf, lambda b: rsi(b.close, self.htf_period))
        lvl = float(self.trend_level)
        self.plot(f"RSI {self.period}", fast, pane="lower")
        self.plot(f"{self.htf} RSI {self.htf_period}", slow, pane="lower", style="step")
        self.plot("Buy level", pd.Series(float(self.buy_below), index=bars.index), pane="lower")
        self.plot("Trend level", pd.Series(lvl, index=bars.index), pane="lower")

        short_in, short_out = 100 - self.buy_below, 100 - self.sell_above
        pos, state = [], 0.0
        for f, s in zip(fast.tolist(), slow.tolist()):
            if f != f or s != s:                          # warm-up: an RSI isn't known yet
                pos.append(0.0 if state == 0 else state)
                continue
            if state == 1 and (f > self.sell_above or s < lvl):
                state = 0.0
            elif state == -1 and (f < short_out or s > 100 - lvl):
                state = 0.0
            if state == 0:
                if f < self.buy_below and s > lvl:
                    state = 1.0
                elif self.allow_short and f > short_in and s < 100 - lvl:
                    state = -1.0
            pos.append(state)
        return pd.Series(pos, index=bars.index)
