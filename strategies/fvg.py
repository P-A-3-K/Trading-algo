"""Fair Value Gap (FVG) retest, with swing stop and reward:risk take-profit.

The pattern (3 candles):
  * Bullish FVG: candle 1's high is BELOW candle 3's low. Candle 2 moved up so fast
    it left a price range nobody traded in: the zone from candle 1's high (bottom)
    to candle 3's low (top).
  * Bearish FVG: the mirror image, candle 1's low ABOVE candle 3's high.

The trade:
  * Signal: the FIRST day price comes back into the zone (bullish: low <= top;
    bearish: high >= bottom). Later touches of the same FVG are ignored.
  * Entry: at the NEXT day's open. Skipped if that open is already past the stop.
  * Stop-loss: the last swing low (long) / swing high (short) before the FVG.
    A swing low is a bar whose low is below the `swing` bars on each side; it is
    only known `swing` bars later, so only swings confirmed by the time the FVG
    is complete are used (no peeking).
  * Take-profit: entry + rr * (entry - stop) for longs, mirrored for shorts.
  * Exits happen AT the stop/target level during the day, or at the open if the
    price gaps past it. If one day touches both, we assume the stop came first
    (daily bars don't show the order; assuming the worst keeps results honest).
  * Size: risk `risk_pct` % of the account between entry and stop, capped at 100%
    of the account (no borrowing).
  * One trade at a time. FVGs touched while a trade is open are used up.
    An FVG that isn't touched within `expiry` bars is dropped.
  * Day trading (built-in setting, intraday only): enter only in regular hours
    (9:30-16:00 New York) and close any open trade at the last regular bar's close.
"""
import numpy as np
import pandas as pd

from quantlab.strategy import Param, Strategy


class FairValueGap(Strategy):
    name = "FVG retest"
    description = ("Trade the first retest of a fair value gap: enter at the next open, "
                   "stop at the last swing, take profit at a reward:risk multiple.")
    params = {
        "rr": Param(2.0, 0.5, 10.0, "Take profit (reward:risk)", step=0.5),
        "swing": Param(2, 1, 10, "Swing size (bars each side)"),
        "expiry": Param(20, 1, 250, "FVG expires after (bars)"),
        "risk_pct": Param(1.0, 0.1, 10.0, "Risk per trade (% of account)", step=0.1),
    }

    def compute(self, bars):
        o, h, l, c = (bars[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        dates, n, N = bars.index, len(bars), self.swing

        swing_lows, swing_highs = [], []      # (bar index, price), confirmed so far
        pending = []                           # untouched FVGs
        signal = None                          # FVG touched today -> enter at next open
        trade = None                           # the open trade, if any
        zone_top, zone_bot, stop_line, target_line = (np.full(n, np.nan) for _ in range(4))
        may_enter, may_hold = self.session_masks(bars)   # all True unless day trading

        for t in range(n):
            # 1. Enter at today's open if yesterday touched an FVG
            entered_today = False
            if signal is not None and trade is None and may_enter[t]:
                f, entry = signal, o[t]
                if (entry - f["stop"]) * f["side"] > 0:          # open still on the right side of the stop
                    risk = abs(entry - f["stop"])
                    size = min(1.0, self.risk_pct / 100 / (risk / entry))
                    trade = {"side": f["side"], "entry": entry, "date": dates[t], "size": size,
                             "stop": f["stop"], "target": entry + f["side"] * self.rr * risk}
                    entered_today = True
            signal = None

            # 2. Exit at the stop / target (stop first if a day touches both)
            if trade is not None:
                side, stop, target = trade["side"], trade["stop"], trade["target"]
                exit_price, reason = None, ""
                if side > 0:
                    if not entered_today and o[t] <= stop:
                        exit_price, reason = o[t], "stop (gap)"
                    elif l[t] <= stop:
                        exit_price, reason = stop, "stop"
                    elif not entered_today and o[t] >= target:
                        exit_price, reason = o[t], "target (gap)"
                    elif h[t] >= target:
                        exit_price, reason = target, "target"
                else:
                    if not entered_today and o[t] >= stop:
                        exit_price, reason = o[t], "stop (gap)"
                    elif h[t] >= stop:
                        exit_price, reason = stop, "stop"
                    elif not entered_today and o[t] <= target:
                        exit_price, reason = o[t], "target (gap)"
                    elif l[t] <= target:
                        exit_price, reason = target, "target"
                if exit_price is None and not may_hold[t]:       # day trading: flat at the close
                    exit_price, reason = c[t], "end of day"
                if exit_price is not None:
                    self.trade(trade["date"], trade["entry"], side, trade["size"], dates[t],
                               exit_price, stop, target, reason)
                    trade = None

            # 3. First touches of pending FVGs (made on earlier days)
            still = []
            for f in pending:
                touched = l[t] <= f["top"] if f["side"] > 0 else h[t] >= f["bottom"]
                if touched:
                    if trade is None and signal is None:
                        signal = f                    # newest touched first (see sort below)
                elif t - f["made"] <= self.expiry:
                    still.append(f)
            pending = still

            # 4. Swings confirmed today (the bar N days ago, compared with N bars each side)
            k = t - N
            if k - N >= 0:
                if l[k] < l[k - N:k].min() and l[k] < l[k + 1:t + 1].min():
                    swing_lows.append((k, l[k]))
                if h[k] > h[k - N:k].max() and h[k] > h[k + 1:t + 1].max():
                    swing_highs.append((k, h[k]))

            # 5. New FVG completed today (today = candle 3)
            if t >= 2:
                if h[t - 2] < l[t]:                                    # bullish
                    stop = next((p for i, p in reversed(swing_lows) if i <= t - 2), None)
                    if stop is not None and stop < h[t - 2]:
                        pending.insert(0, {"side": 1, "top": l[t], "bottom": h[t - 2],
                                           "stop": stop, "made": t})
                elif l[t - 2] > h[t]:                                  # bearish
                    stop = next((p for i, p in reversed(swing_highs) if i <= t - 2), None)
                    if stop is not None and stop > l[t - 2]:
                        pending.insert(0, {"side": -1, "top": l[t - 2], "bottom": h[t],
                                           "stop": stop, "made": t})

            # 6. Lines for the chart, as known at today's close
            if pending:
                zone_top[t], zone_bot[t] = pending[0]["top"], pending[0]["bottom"]
            if trade is not None:
                stop_line[t], target_line[t] = trade["stop"], trade["target"]

        if trade is not None:                                          # still open at the end
            self.trade(trade["date"], trade["entry"], trade["side"], trade["size"],
                       stop=trade["stop"], target=trade["target"])

        # Lines: the newest untouched FVG zone, and the open trade's stop / take-profit
        self.plot("FVG top", pd.Series(zone_top, index=dates), style="step")
        self.plot("FVG bottom", pd.Series(zone_bot, index=dates), style="step")
        self.plot("Stop", pd.Series(stop_line, index=dates), style="step")
        self.plot("Take profit", pd.Series(target_line, index=dates), style="step")
        return None                          # positions come from the recorded trades
