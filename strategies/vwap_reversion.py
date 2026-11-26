"""VWAP reversion: fade big stretches away from the day's volume-weighted average price.

Regular hours (New York). VWAP = average price of today's trading so far, weighted by volume.
  * Stretch = (close - VWAP) / close.  Threshold = `band` x the average daily range of the
    previous 20 days (so it adapts to calm and wild markets).
  * After the first `start_minutes`: short when the close is more than the threshold above
    VWAP, buy when it is more than the threshold below.
  * Exit when a candle closes back at VWAP, and always before the close. Can trade again later.
Flat overnight (no CFD swaps). Needs 5m (or 15m) candles with volume.
"""
import numpy as np
import pandas as pd

from quantlab.strategy import Param, Strategy

from strategies.gap_fade import CLOSE, OPEN, session_layout


class VwapReversion(Strategy):
    name = "VWAP reversion (intraday)"
    description = ("Fade big moves away from the day's VWAP (volume-weighted average price); exit back at VWAP. "
                   "Flat at the close.")
    params = {
        "band": Param(0.25, 0.05, 1.0, "Stretch needed (x average daily range)", step=0.05,
                      help="0.25: enter when price is a quarter of a normal day's range away from VWAP."),
        "start_minutes": Param(30, 0, 180, "Start after (minutes from the open)", step=15,
                               help="Skip the opening rush, when VWAP is still settling."),
        "allow_short": Param(True, label="Also fade moves up (short)",
                             help="Yes: short stretches above VWAP too. No: only buy stretches below."),
    }
    guide = """
## In short
Big traders measure their fills against **VWAP**, the volume-weighted average price of the day.
When the price runs far from it without news, it often drifts back. This strategy fades
those stretches and exits at VWAP. Always flat by the close.

## The rules
- **VWAP**: from 9:30, sum(typical price x volume) / sum(volume).
- **Threshold**: *band* x the average high-low range of the previous 20 days.
- After *Start after* minutes: short above VWAP + threshold, buy below VWAP - threshold.
- **Exit**: a candle closes back at VWAP; always before 16:00. It may trade several times a day.

## Watch out
- Trend days run away from VWAP and keep going: those are the losses.
"""

    def compute(self, bars):
        idx, n = bars.index, len(bars)
        pos, vwap_line = np.zeros(n), np.full(n, np.nan)
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None and n >= 2:
            self._days(bars, pos, vwap_line)
        self.plot("VWAP", pd.Series(vwap_line, index=idx))
        return pd.Series(pos, index=idx)

    def _days(self, bars, pos, vwap_line):
        end, step, day, regular = session_layout(bars.index)
        h, l, c = (bars[k].to_numpy(float) for k in ("high", "low", "close"))
        v = bars["volume"].to_numpy(float) if "volume" in bars.columns else np.ones(len(bars))
        reg = np.flatnonzero(regular)
        groups = np.split(reg, np.flatnonzero(np.diff(day[reg])) + 1) if len(reg) else []
        ranges = []                                         # previous days' high-low range (fraction)
        start = OPEN + self.start_minutes
        for rows in groups:
            adr = np.mean(ranges[-20:]) if len(ranges) >= 5 else np.nan
            tp = (h[rows] + l[rows] + c[rows]) / 3
            vol = np.where(v[rows] > 0, v[rows], 0.0)
            cum_v = np.cumsum(vol)
            vwap = np.where(cum_v > 0, np.cumsum(tp * vol) / np.where(cum_v > 0, cum_v, 1),
                            np.cumsum(tp) / np.arange(1, len(rows) + 1))
            vwap_line[rows] = vwap
            if adr == adr:
                band = self.band * adr
                side = 0.0
                for k, i in enumerate(rows):
                    if end[i] >= CLOSE:
                        side = 0.0                          # flat at the close
                    else:
                        stretch = (c[i] - vwap[k]) / c[i]
                        if side and stretch * side >= 0:    # back at VWAP: exit
                            side = 0.0
                        elif not side and end[i] > start:
                            if stretch > band and self.allow_short:
                                side = -1.0
                            elif stretch < -band:
                                side = 1.0
                    pos[i] = side
            ranges.append((h[rows].max() - l[rows].min()) / c[rows[-1]])
