"""Gap fade: bet that the opening gap closes back toward yesterday's close.

Every trading day (regular hours, New York):
  * Gap = today's 9:30 open vs yesterday's 16:00 close.
  * If the gap is between `min_gap_pct` and `max_gap_pct`, at the close of the first
    5-minute candle trade AGAINST it (short a gap up, buy a gap down)...
  * ...unless that candle already closed the gap.
  * Exit when a candle closes back at/through yesterday's close (gap filled), or after
    `exit_minutes` from the open, and always before the close.
One trade a day at most, flat overnight (no CFD swaps). Needs 5m (or 15m) candles.
"""
import numpy as np
import pandas as pd

from quantlab.strategy import Param, Strategy

OPEN, CLOSE = 9 * 60 + 30, 16 * 60


def session_layout(idx):
    """(minute each bar ENDS on the New York clock, bar length, trading-day number, regular-hours mask)."""
    ny = idx.tz_convert("America/New_York")
    step = int(pd.Series(idx[1:] - idx[:-1]).median().total_seconds() // 60) or 5
    end = (ny.hour * 60 + ny.minute + step).to_numpy()
    regular = (end - step >= OPEN) & (end <= CLOSE)
    day = pd.factorize(pd.Index(ny.date))[0]
    return end, step, day, regular


class GapFade(Strategy):
    name = "Gap fade (opening gap fill)"
    description = ("Trade against the opening gap: short a gap up, buy a gap down, exit when the gap fills "
                   "or by mid-day. Flat at the close.")
    params = {
        "min_gap_pct": Param(0.2, 0.05, 2.0, "Minimum gap (%)", step=0.05,
                             help="Skip tiny gaps: the move to fill them is too small for the costs."),
        "max_gap_pct": Param(1.5, 0.2, 5.0, "Maximum gap (%)", step=0.1,
                             help="Skip huge gaps: those usually come with real news and keep going."),
        "exit_minutes": Param(150, 30, 385, "Give up after (minutes from the open)", step=15,
                              help="150 = exit at 12:00 if the gap hasn't filled by then."),
        "allow_short": Param(True, label="Also fade gaps up (short)",
                             help="Yes: short gaps up too. No: only buy gaps down."),
    }
    guide = """
## In short
Overnight news makes the market open above or below yesterday's close (a **gap**). Small and
medium gaps often **close back** during the morning as the opening rush fades. This strategy
trades against the gap and gets out once it fills, or by mid-day. Always flat by the close.

## The rules
- **Gap** = today's 9:30 open vs yesterday's 16:00 close (regular hours only).
- If the gap is between the *minimum* and *maximum*, at the close of the first candle: short a gap up, buy a gap down.
- **Exit**: a candle closes at or through yesterday's close (filled), or *Give up after* minutes, and always before 16:00.

## Watch out
- Needs 5m or 15m candles. Big news gaps tend to run, not fill: that's what the maximum is for.
"""

    def validate(self):
        if self.min_gap_pct >= self.max_gap_pct:
            raise ValueError("the minimum gap must be smaller than the maximum")

    def compute(self, bars):
        idx, n = bars.index, len(bars)
        pos, gap_line = np.zeros(n), np.full(n, np.nan)
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None and n >= 2:
            self._days(bars, pos, gap_line)
        self.plot("Gap (%)", pd.Series(gap_line, index=idx), pane="lower", style="step")
        return pd.Series(pos, index=idx)

    def _days(self, bars, pos, gap_line):
        end, step, day, regular = session_layout(bars.index)
        close, opn = bars["close"].to_numpy(float), bars["open"].to_numpy(float)
        lo, hi = self.min_gap_pct / 100, self.max_gap_pct / 100
        give_up = OPEN + self.exit_minutes
        reg = np.flatnonzero(regular)
        prev_close = np.nan
        for rows in (np.split(reg, np.flatnonzero(np.diff(day[reg])) + 1) if len(reg) else []):
            first = rows[0]
            if end[first] - step == OPEN and prev_close == prev_close:
                gap = opn[first] / prev_close - 1
                gap_line[rows] = gap * 100
                side = -np.sign(gap)
                if lo <= abs(gap) <= hi and (side > 0 or self.allow_short) \
                        and (close[first] - prev_close) * side < 0:          # not already filled
                    for i in rows:
                        if end[i] >= CLOSE or end[i] > give_up or (close[i] - prev_close) * side >= 0:
                            break                                          # flat from this bar's close
                        pos[i] = side
            last = rows[end[rows] == CLOSE]
            prev_close = close[last[0]] if len(last) else np.nan
