"""Swing structure: the chart's swing highs and lows, ranked by how prominent they are.

How a trader's eye reads structure, as rules:
  * Level 1: every swing. A swing high is a candle whose high is above the candle before it
    and not below the candle after it (a swing low is the mirror image).
  * Level 2: the same rule applied to the level-1 swings themselves: a level-1 swing high that
    is higher than the level-1 swing highs on both sides of it. Level 3 = the same on level 2,
    and so on. Each level keeps only the swings the smaller ones were leading to.
  * Taken: a swing high is "taken" when a later candle trades above it (a swing low: below).
    Its level line ends there.
  * Confirmation: on the chart swings sit on their own candle, but in real time a level-1
    swing is only known one candle later, and a level-k swing only once the next level-(k-1)
    swing on its right is known. swing_table() records that time for later research.
This draws only: it never trades.
"""
import numpy as np
import pandas as pd

from quantlab.strategy import Param, Strategy


def next_beyond(values: np.ndarray, higher: bool) -> np.ndarray:
    """For every candle, the first later candle that trades beyond it (strictly above its high /
    below its low); len(values) when that never happens."""
    n = len(values)
    out = np.full(n, n)
    stack = []
    v = values if higher else -values
    for i in range(n - 1, -1, -1):
        while stack and v[stack[-1]] <= v[i]:
            stack.pop()
        out[i] = stack[-1] if stack else n
        stack.append(i)
    return out


def pivot_levels(values: np.ndarray, higher: bool, max_level: int):
    """-> (swing candle indices, their level, the candle where each became known).

    Level 1 from every candle; level k+1 = pivots among the level-k swings."""
    v = values if higher else -values
    n = len(v)
    if n < 3:
        return np.array([], int), np.array([], int), np.array([], int)
    mid = np.arange(1, n - 1)
    is_p = (v[mid] > v[mid - 1]) & (v[mid] >= v[mid + 1])
    idx = mid[is_p]                                   # level-1 swings
    known = idx + 1                                   # confirmed at the next candle's close
    level = np.ones(len(idx), int)
    seq, seq_known, pos = idx, known, np.arange(len(idx))   # current level's swings (positions into idx)
    for lvl in range(2, max_level + 1):
        if len(seq) < 3:
            break
        sv = v[seq]
        j = np.arange(1, len(seq) - 1)
        up = (sv[j] > sv[j - 1]) & (sv[j] >= sv[j + 1])
        keep = j[up]
        if not len(keep):
            break
        level[pos[keep]] = lvl
        known[pos[keep]] = seq_known[keep + 1]        # known once the next swing on the right is
        seq, seq_known, pos = seq[keep], known[pos[keep]], pos[keep]
    return idx, level, known


def swing_table(bars: pd.DataFrame, max_level: int = 3) -> pd.DataFrame:
    """Every swing high and low: candle, price, level, when it became known, when it was taken."""
    rows = []
    for kind, col, higher in (("high", "high", True), ("low", "low", False)):
        vals = bars[col].to_numpy(float)
        idx, level, known = pivot_levels(vals, higher, max_level)
        taken = next_beyond(vals, higher)[idx] if len(idx) else np.array([], int)
        n = len(bars)
        for i, lv, k, t in zip(idx, level, known, taken):
            rows.append({"kind": kind, "i": int(i), "time": bars.index[i], "price": vals[i], "level": int(lv),
                         "known_i": int(min(k, n - 1)), "taken_i": int(t) if t < n else None})
    t = pd.DataFrame(rows, columns=["kind", "i", "time", "price", "level", "known_i", "taken_i"])
    return t.sort_values("i").reset_index(drop=True)


class SwingStructure(Strategy):
    name = "Swing structure (levels)"
    description = ("Marks swing highs and lows and ranks them by prominence: level 1 = every swing, "
                   "level 2 = swings of swings, level 3... Lines show untaken levels. Draws only, no trades.")
    params = {
        "max_level": Param(3, 1, 5, "Highest level",
                           help="How many times the swing rule is applied on top of itself. 3 is a good start."),
        "markers_from": Param(1, 1, 5, "Show markers from level",
                              help="1 = mark every swing. Raise it to hide the small ones."),
        "lines_from": Param(2, 1, 6, "Draw level lines from level",
                            help="Horizontal line from the swing until a candle trades through it. "
                                 "Higher = fewer, more important lines. Above 'Highest level' = no lines."),
    }
    guide = """
## In short
Marks the **swing highs and lows** the way you'd read structure by eye, and ranks them:
small wiggles are level 1, the swings those wiggles were leading to are level 2, and so on.
Higher levels are drawn **bigger and stronger**. A line runs from each important swing until
price **takes** it (trades through it). This only draws, it never trades.

## The rules
- **Level 1**: a candle whose high is above the previous candle's high and not below the next one's (swing high); the mirror image for swing lows.
- **Level 2**: the same rule on the level-1 swings: a level-1 high that's higher than the level-1 highs on both sides. **Level 3** = the same on level 2, etc.
- **Taken**: a later candle trades beyond the swing. Its line stops there.

## Watch out
- In real time a swing is only known later: level 1 one candle later, higher levels once the next swing of the level below appears on the right. The chart shows swings on their own candle, as the eye sees them afterwards.
- Works on any ticker and timeframe. On long intraday histories zoom in: there are many level-1 swings.
"""

    def validate(self):
        if self.markers_from > self.max_level:
            raise ValueError("'Show markers from level' can't be above 'Highest level'")

    def compute(self, bars):
        t = swing_table(bars, self.max_level)
        idx, last = bars.index, len(bars) - 1
        for r in t.itertuples():
            if r.level >= self.markers_from:
                self.mark(r.time, r.price, r.kind, r.level, f"{'H' if r.kind == 'high' else 'L'}{r.level}"
                          if r.level >= 3 else "")
            if r.level >= self.lines_from:
                end = last if r.taken_i is None or pd.isna(r.taken_i) else int(r.taken_i)
                self.level_line(r.time, idx[end], r.price, r.kind, r.level)
        return pd.Series(0.0, index=idx)
