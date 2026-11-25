"""Intraday momentum: the first half hour predicts the last half hour.

Research on US index ETFs (e.g. SPY) found that the market's move from the previous
close to 10:00 New York tends to continue in the LAST half hour of the day (15:30-16:00),
as big traders who must finish their orders by the close keep pushing the same way.

Rules (regular hours, New York time):
  * Signal: return from the previous day's regular close (or today's 9:30 open) to
    the close of the bar ending at 9:30 + `signal_minutes`.
  * Trade: at the close of the bar ending `minutes_before_close` before 16:00, go long if
    the signal is up (short if down and shorts are allowed); exit at 16:00.
  * Optional filters: a minimum signal size; the move of the hour before entry must agree.
One short trade a day, always flat overnight (no CFD swaps). Needs intraday candles (5m-30m).
Each bar only uses what is known at its close (the look-ahead check verifies it).
"""
import numpy as np
import pandas as pd

from quantlab.strategy import Param, Strategy

OPEN, CLOSE = 9 * 60 + 30, 16 * 60


class IntradayMomentum(Strategy):
    name = "Intraday momentum (first vs last half hour)"
    description = ("If the market rose from yesterday's close to 10:00, buy for the last half hour of the "
                   "day (sell short if it fell). One short trade a day, flat at the close.")
    params = {
        "signal_minutes": Param(30, 15, 120, "Signal: first minutes after the open", step=15,
                                help="The signal is the move up to 9:30 + this many minutes (30 = 10:00)."),
        "signal_from": Param("prev close", choices=("prev close", "open"), label="Signal measured from",
                             help="prev close: includes the overnight gap (as in the research). open: from 9:30 only."),
        "minutes_before_close": Param(30, 5, 120, "Enter this many minutes before the close", step=5,
                                      help="30 = hold 15:30-16:00. Longer holds catch more but dilute the effect."),
        "min_move_pct": Param(0.0, 0.0, 2.0, "Only trade if the signal moved at least (%)", step=0.05,
                              help="Skip quiet days: the effect is stronger after bigger moves (and costs matter less)."),
        "confirm": Param(False, label="The hour before entry must agree",
                         help="Yes: only trade if the price also moved the same way in the hour before the entry."),
        "allow_short": Param(True, label="Also trade down days",
                             help="Yes: short the last half hour after a down signal. No: long only."),
    }
    guide = """
## In short
On many days the market's direction early in the session **reappears in the last half hour**.
Funds rebalancing and traders finishing big orders by the close keep pushing the same way.
This strategy reads the move up to 10:00 and trades only the end of the day: one short trade,
always flat by the close, so no overnight risk or CFD swap fees.

## The rules
- **Signal**: return from yesterday's regular-hours close (or today's 9:30 open) to 10:00 (*Signal: first minutes*).
- **Entry**: at 15:30 (*minutes before close*), long after an up signal, short after a down signal (if allowed).
- **Exit**: at the 16:00 close.
- **Filters** (optional): skip days with a small signal; require the last hour before entry to move the same way.

## When it tends to work
- Busy, trending days in index ETFs/CFDs (SPY, QQQ / US500, US100), especially high-volatility periods.

## When it struggles
- Quiet days (small moves, costs dominate) and days that reverse in the afternoon.
- The effect is small per trade: tight spreads are essential.

## Watch out
- Needs intraday candles (5m to 30m). On daily candles it never trades.
- Look at "before costs" vs "after costs": with ~250 trades a year, costs matter.
"""

    def compute(self, bars):
        idx = bars.index
        n = len(bars)
        sig_line = np.full(n, np.nan)
        pos = np.zeros(n)
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None and n >= 2:
            self._trade_days(bars, pos, sig_line)
        self.plot("Signal (%)", pd.Series(sig_line, index=idx), pane="lower", style="step")
        self.plot("Zero", pd.Series(0.0, index=idx), pane="lower")
        return pd.Series(pos, index=idx)

    def _trade_days(self, bars, pos, sig_line):
        idx = bars.index
        ny = idx.tz_convert("America/New_York")
        step = int(pd.Series(idx[1:] - idx[:-1]).median().total_seconds() // 60) or 5
        end = (ny.hour * 60 + ny.minute + step).to_numpy()          # minute each bar ENDS
        regular = (end - step >= OPEN) & (end <= CLOSE)
        day = pd.factorize(pd.Index(ny.date))[0]
        close = bars["close"].to_numpy(dtype=float)
        opn = bars["open"].to_numpy(dtype=float)
        sig_end = OPEN + self.signal_minutes
        entry_at = CLOSE - self.minutes_before_close
        if entry_at <= sig_end:
            raise ValueError("the entry must come after the signal (fewer minutes before the close, or a shorter signal)")
        threshold = self.min_move_pct / 100
        reg = np.flatnonzero(regular)
        groups = np.split(reg, np.flatnonzero(np.diff(day[reg])) + 1) if len(reg) else []
        prev_close = np.nan
        for rows in groups:                                         # one trading day at a time
            if not len(rows):
                continue
            base = prev_close if self.signal_from == "prev close" else opn[rows[0]]
            at_sig = rows[end[rows] == sig_end]
            at_entry = rows[end[rows] == entry_at]
            # Only what is known at each bar: the signal from its own bar on; the trade from the
            # entry bar to the last bar before the close (no need to know the close exists)
            if len(at_sig) and base == base:
                s = close[at_sig[0]] / base - 1
                sig_line[rows[rows >= at_sig[0]]] = s * 100
                side = 1.0 if s > 0 else -1.0
                ok = bool(len(at_entry)) and abs(s) >= threshold and (side > 0 or self.allow_short)
                if ok and self.confirm:
                    hour_ago = rows[end[rows] == entry_at - 60]
                    ok = bool(len(hour_ago)) and np.sign(close[at_entry[0]] - close[hour_ago[0]]) == side
                if ok:
                    hold = rows[(end[rows] >= entry_at) & (end[rows] < CLOSE)]
                    pos[hold] = side                                # decided at 15:30's close, held to 16:00
            last = rows[end[rows] == CLOSE]
            prev_close = close[last[0]] if len(last) else np.nan   # half day / missing close: skip tomorrow
