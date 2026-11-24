"""Opening range breakout (ORB): trade the break of the first minutes' high or low.

Every trading day, regular hours only (9:30-16:00 New York):
  * Opening range: the high and low of the first `range_minutes` (e.g. 9:30-10:00).
  * Entry: the first time price trades ABOVE the range high (long) or BELOW the range low
    (short), at that level (a stop order); at the open of the bar if it gaps through.
  * Stop: the other side of the range, or its middle.
  * Target: entry +/- rr x the distance to the stop (0 = no target).
  * Exit at the stop / target, otherwise at the close of the last regular-hours bar.
  * One trade per day. If a bar touches both the entry and the stop, the stop is assumed
    (cautious). Size: risk `risk_pct` % of the account on the stop distance (max 100%).
Needs intraday candles shorter than the range (5m or 15m for a 30-minute range).
"""
import numpy as np
import pandas as pd

from quantlab.strategy import Param, Strategy

OPEN, CLOSE = 9 * 60 + 30, 16 * 60


class OpeningRangeBreakout(Strategy):
    name = "Opening range breakout"
    description = ("Intraday: trade the first break of the opening range's high or low, stop at the "
                   "other side (or middle), target at a reward:risk multiple, flat at the close.")
    params = {
        "range_minutes": Param(30, 5, 120, "Opening range (minutes)", step=5,
                               help="How long after 9:30 the range is measured. Use candles shorter than this (5m or 15m)."),
        "rr": Param(2.0, 0.0, 10.0, "Take profit (reward:risk, 0 = none)", step=0.5,
                    help="Target = entry + this x (entry - stop). 0: no target, hold to the stop or the close."),
        "stop_at": Param("other side", choices=("other side", "middle"), label="Stop at",
                         help="other side: stop at the opposite edge of the range (wider). middle: halfway (tighter, bigger size)."),
        "allow_short": Param(True, label="Also trade downside breaks",
                             help="Yes: short a break below the range low. No: long breaks only."),
        "risk_pct": Param(1.0, 0.1, 10.0, "Risk per trade (% of account)", step=0.1,
                          help="Size so the stop loses about this % of the account (max 100% invested)."),
    }
    guide = """
## In short
The first half hour of the US session sets a range. A break out of it often starts the day's
move. This strategy trades that **first break**, with a stop on the other side, and is always
**flat by the close**: no overnight risk, which suits prop-firm rules.

## The rules
- **Range**: high and low of the regular-hours bars in the first *Opening range* minutes (9:30 to 10:00 for 30).
- **Entry**: the first bar after the range that trades above the high (long) or below the low (short; if allowed). Filled at the range edge, or at the bar's open if it gapped past it.
- **Stop**: the other side of the range, or its middle. **Target**: *RR* x the risk (0 = none).
- **Exit**: at the stop or target during the day; otherwise at the close of the last regular-hours bar.
- One trade per day; pre-market and after-hours bars are ignored.
- If one bar touches both the entry and the stop, the stop is assumed (cautious).

## Settings
- **Opening range**: 15-60 minutes is typical. Needs candles shorter than the range: use 5m (or 15m for 30-60).
- **Stop at middle**: tighter stop, so a bigger position at the same risk: more wins cut off by noise, bigger wins.
- **RR**: 0 holds winners to the close (lets trend days run).

## When it tends to work
- Trend days in liquid index ETFs (SPY, QQQ, IWM) and big stocks.

## When it struggles
- Range days: the break fails and reverses into the stop. Costs on many small trades.

## Watch out
- Yahoo keeps only 60 days of 5m candles: connect Alpaca (Data page) for years of history.
- Lines on the chart: the opening range (high/low), and the trade's stop and target.
"""

    def compute(self, bars):
        idx = bars.index
        n = len(bars)
        rng_hi, rng_lo, stop_line, tgt_line = (np.full(n, np.nan) for _ in range(4))
        if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or n < 2:
            self._lines(bars, rng_hi, rng_lo, stop_line, tgt_line)
            return None                                  # daily candles: no opening range, no trades
        o, h, l, c = (bars[k].to_numpy(dtype=float).tolist() for k in ("open", "high", "low", "close"))
        minutes = (idx.hour * 60 + idx.minute).to_numpy()
        step = int(pd.Series(idx[1:] - idx[:-1]).median().total_seconds() // 60) or 5
        regular = (minutes >= OPEN) & (minutes < CLOSE)
        in_range = regular & (minutes + step <= OPEN + self.range_minutes)     # bar ENDS within the range
        day = pd.factorize(pd.Index(idx.date))[0]
        rr, risk_pct, shorts, mid = self.rr, self.risk_pct, self.allow_short, self.stop_at == "middle"

        reg_idx = np.flatnonzero(regular)
        if not len(reg_idx):
            self._lines(bars, rng_hi, rng_lo, stop_line, tgt_line)
            return None
        # Walk day by day through regular-hours bars only
        bounds = np.flatnonzero(np.diff(day[reg_idx])) + 1
        for chunk in np.split(reg_idx, bounds):
            r_bars = [i for i in chunk if in_range[i]]
            if not r_bars or minutes[r_bars[-1]] + step != OPEN + self.range_minutes:
                continue                                 # range incomplete (late open, missing bars)
            hi = max(h[i] for i in r_bars)
            lo = min(l[i] for i in r_bars)
            if hi <= lo:
                continue
            after = [i for i in chunk if not in_range[i]]
            if not after:
                continue
            trade = None
            for i in after:
                rng_hi[i], rng_lo[i] = hi, lo
                if trade is None:
                    side = entry = None
                    up, down = h[i] > hi, shorts and l[i] < lo
                    if up and down:                      # both edges in one bar: take the one nearer the open
                        side = 1 if abs(o[i] - hi) <= abs(o[i] - lo) else -1
                    elif up:
                        side = 1
                    elif down:
                        side = -1
                    if side:
                        entry = max(o[i], hi) if side > 0 else min(o[i], lo)
                        stop = (hi + lo) / 2 if mid else (lo if side > 0 else hi)
                        risk = abs(entry - stop)
                        if risk <= 0 or (stop - entry) * side >= 0:
                            break                         # gapped past the stop already: no trade today
                        size = min(1.0, risk_pct / 100 / (risk / entry))
                        target = entry + side * rr * risk if rr > 0 else None
                        trade = {"t": i, "side": side, "entry": entry, "stop": stop, "target": target, "size": size}
                        # the entry bar itself can also hit the stop (cautious)
                        if (side > 0 and l[i] <= stop) or (side < 0 and h[i] >= stop):
                            self._close(trade, idx, i, stop, "stop")
                            trade = "done"
                            continue
                elif trade != "done":
                    side, stop, target = trade["side"], trade["stop"], trade["target"]
                    px = None
                    if side > 0:
                        if o[i] <= stop: px, why = o[i], "stop (gap)"
                        elif l[i] <= stop: px, why = stop, "stop"
                        elif target is not None and h[i] >= target: px, why = max(o[i], target), "target"
                    else:
                        if o[i] >= stop: px, why = o[i], "stop (gap)"
                        elif h[i] >= stop: px, why = stop, "stop"
                        elif target is not None and l[i] <= target: px, why = min(o[i], target), "target"
                    if px is not None:
                        self._close(trade, idx, i, px, why)
                        trade = "done"
                        continue
                if isinstance(trade, dict):
                    stop_line[i] = trade["stop"]
                    tgt_line[i] = trade["target"] if trade["target"] is not None else np.nan
            if isinstance(trade, dict):                  # flat at the close of the last regular bar
                last = after[-1]
                if last + 1 < n or minutes[last] + step >= CLOSE:
                    self._close(trade, idx, last, c[last], "end of day")
                else:                                    # data ends mid-session: still open
                    self.trade(idx[trade["t"]], trade["entry"], trade["side"], trade["size"],
                               stop=trade["stop"], target=trade["target"])
        self._lines(bars, rng_hi, rng_lo, stop_line, tgt_line)
        return None

    def _close(self, trade, idx, i, price, why):
        self.trade(idx[trade["t"]], trade["entry"], trade["side"], trade["size"], idx[i], price,
                   trade["stop"], trade["target"], why)

    def _lines(self, bars, hi, lo, stop, tgt):
        self.plot("Range high", pd.Series(hi, index=bars.index), style="step")
        self.plot("Range low", pd.Series(lo, index=bars.index), style="step")
        self.plot("Stop", pd.Series(stop, index=bars.index), style="step")
        self.plot("Take profit", pd.Series(tgt, index=bars.index), style="step")
