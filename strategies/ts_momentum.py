"""Time-series momentum: hold an asset while its own past return is positive.

Each month (at the first trading day's close), look at the asset's return over the last
`lookback_months` (optionally skipping the most recent `skip_months`). Positive -> hold it
for the month; negative -> cash (or short). The classic "trend following" behind many
managed-futures funds; its strength comes from running it on MANY unrelated assets at
once (stocks, bonds, gold, commodities): use the Portfolio page for that.
"""
import pandas as pd

from quantlab.strategy import Param, Strategy

DAYS_PER_MONTH = 21


class TimeSeriesMomentum(Strategy):
    name = "Time-series momentum (monthly)"
    description = ("Each month, hold the asset if its return over the past months is positive, "
                   "otherwise cash (or short).")
    params = {
        "lookback_months": Param(12, 1, 24, "Look back (months)",
                                 help="Compare today's price with this many months ago. 12 is the classic."),
        "skip_months": Param(0, 0, 3, "Skip most recent (months)",
                             help="Ignore the last month(s) of the look-back (12-1 momentum skips 1), which reduces "
                                  "whipsaw from short-term reversals."),
        "rebalance": Param("monthly", choices=("monthly", "weekly"), label="Re-decide",
                           help="How often the signal is checked: first trading day of each month or week."),
        "allow_short": Param(False, label="Short when negative",
                             help="Yes: short the asset when its past return is negative, instead of holding cash."),
    }
    guide = """
## In short
**Trend following, the simplest way**: if an asset went up over the past year, hold it; if it went
down, step aside. One decision a month. Alone it's a modest risk reducer; spread over many
**unrelated** assets it's one of the most studied strategies there is.

## The rules
- On the first trading day of each month (or week), at the close: return = close / close *Look back* months ago - 1, optionally ending *Skip* months ago.
- Positive: hold the asset until the next check. Negative: cash (or short, if allowed).
- A month counts as 21 trading days.

## How to use it well
- Run it on several ETFs from different asset classes on the **Portfolio** page, e.g. SPY (US stocks), EFA (other developed), TLT (long bonds), IEF (medium bonds), GLD (gold), DBC (commodities), with **Equal risk** weights.
- Daily candles. Intraday data adds nothing here.

## When it tends to work
- Long trends: 2008 (bonds and gold up, stocks down), 2014 (oil down), 2022 (stocks and bonds down, commodities up).

## When it struggles
- Sharp reversals (V-shaped recoveries): it's out at the bottom and back in late.
- Sideways years: many small losses from switching in and out.
"""

    def validate(self):
        if self.skip_months >= self.lookback_months:
            raise ValueError("'skip' must be shorter than 'look back'")

    def compute(self, bars):
        close = bars.close
        look = self.lookback_months * DAYS_PER_MONTH
        skip = self.skip_months * DAYS_PER_MONTH
        past = close.shift(skip) / close.shift(look) - 1
        self.plot(f"{self.lookback_months}-month return", past * 100, pane="lower")
        self.plot("Zero", pd.Series(0.0, index=bars.index), pane="lower")
        # Re-decide on the first bar of each month / week (known at that bar: no peeking
        # at whether tomorrow starts a new month)
        days = pd.DatetimeIndex(bars.index.date) if bars.index.tz is not None else bars.index
        period = days.to_period("M" if self.rebalance == "monthly" else "W")
        new_period = pd.Series(period != pd.Series(period).shift(1).to_numpy(), index=bars.index)
        first_bar_of_day = pd.Series(days != pd.Series(days).shift(1).to_numpy(), index=bars.index)
        decide = new_period & first_bar_of_day
        want = pd.Series(float("nan"), index=bars.index)
        want[decide & (past > 0)] = 1.0
        want[decide & (past <= 0)] = -1.0 if self.allow_short else 0.0
        return want.ffill().fillna(0.0)
