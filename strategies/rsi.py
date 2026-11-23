"""RSI reversion: buy oversold dips, sell on the bounce."""
import pandas as pd

from quantlab.strategy import Param, Strategy


def rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI: average gain vs average loss over `period` bars, scaled 0-100."""
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


class RsiReversion(Strategy):
    name = "RSI reversion"
    description = ("Mean reversion: buy when RSI falls below the oversold level, sell once it "
                   "recovers above the exit level (optionally the mirror image for shorts).")
    params = {
        "period": Param(14, 2, 100, "RSI length (bars)",
                        help="How many bars RSI looks back. 14 is the classic; shorter = more extreme readings, more trades."),
        "buy_below": Param(30, 5, 50, "Buy when RSI below",
                           help="Oversold level. Lower = waits for deeper dips: fewer, but more stretched, entries."),
        "sell_above": Param(55, 30, 95, "Sell when RSI above",
                            help="Exit level for longs. Around 50 = take the bounce early; 70 = hold for a full rebound."),
        "allow_short": Param(False, label="Also go short",
                             help="Yes: also short when RSI is above 100 - 'buy below' (overbought) and cover "
                                  "below 100 - 'sell above'."),
    }
    guide = """
## In short
RSI (Relative Strength Index, 0-100) compares recent gains with recent losses. Very low readings
mean the price fell hard and fast: this strategy bets that such drops **snap back**.

## The rules
- RSI = 100 - 100 / (1 + average gain / average loss), over the last *RSI length* bars.
- **Buy** at the close of a bar where RSI is below *Buy when RSI below* (e.g. 30).
- **Sell** at the close of a bar where RSI is above *Sell when RSI above* (e.g. 55).
- In between, keep whatever position you have.
- With *Also go short*: short when RSI is above 100 - buy level (e.g. 70), cover when it drops below 100 - sell level (e.g. 45).

## Settings
- **RSI length**: 14 is standard. Short lengths (2-5) are a popular variant for quick dips.
- **Buy below / Sell above**: the gap between them decides how long trades last.

## When it tends to work
- Assets that **range** or trend up slowly with pullbacks: broad stock indices have historically rewarded buying dips.
- Short holding periods: the edge is in the quick rebound.

## When it struggles
- **Strong down-trends and crashes**: RSI can stay "oversold" while the price keeps falling (2008, 2020). There is **no stop-loss** here.
- Shorting in a rising market: overbought can stay overbought for months.

## Watch out
- High win rates are typical, but the occasional big loss can cancel many small wins: look at "Average loss" and "Worst drawdown".
- Lower panel: RSI with the buy and sell levels.
"""

    def validate(self):
        if self.buy_below >= self.sell_above:
            raise ValueError("'buy below' must be lower than 'sell above'")

    def compute(self, bars):
        r = rsi(bars.close, self.period)
        self.plot(f"RSI {self.period}", r, pane="lower")
        self.plot("Buy level", pd.Series(float(self.buy_below), index=bars.index), pane="lower")
        self.plot("Sell level", pd.Series(float(self.sell_above), index=bars.index), pane="lower")
        # Step through the bars keeping the current position, so long and short
        # each use their own exit rule.
        short_in, short_out = 100 - self.buy_below, 100 - self.sell_above
        pos, state = [], 0.0
        for v in r.to_numpy():
            if v != v:                                   # warm-up: no RSI yet
                pos.append(0.0)
                continue
            if state == 1 and v > self.sell_above:
                state = 0.0
            elif state == -1 and v < short_out:
                state = 0.0
            if state == 0:
                if v < self.buy_below:
                    state = 1.0
                elif self.allow_short and v > short_in:
                    state = -1.0
            pos.append(state)
        return pd.Series(pos, index=bars.index)
