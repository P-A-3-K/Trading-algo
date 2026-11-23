"""Multi-timeframe: higher-timeframe candles lined up with a strategy's own bars.

The trap this module exists for: a 4-hour candle from 08:00 to 12:00 is only known
at 12:00. A 5-minute bar at 09:00 must NOT see its close (or an RSI computed from
it): that would be knowing the next three hours. So:

  * closed (default): each bar sees the latest higher-timeframe candle that had
    FINISHED by the time the bar closed. At 09:00 that's the 04:00-08:00 candle.
  * forming: the candle that is still being built, from the bars so far
    (08:00 to 09:05). Also honest: it's what a live chart shows at that moment.

Candles are built from the strategy's own bars on the New York wall clock
(4h bins 0-4, 4-8, 8-12, 12-16, 16-20; days from midnight), the same way the app
builds its timeframes, so everything comes from one data source. A higher
timeframe that isn't actually higher than the bars just gives the bars themselves.
"""
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

RULES = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}
COLS = ["open", "high", "low", "close", "volume"]
_CACHE = {}


def _wall(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Clock time in New York with the time zone removed (DST can't shift the bins)."""
    return index.tz_localize(None) if index.tz is not None else index


def _bar_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(0)
    return pd.Series(index[1:] - index[:-1]).median()


def frame(bars: pd.DataFrame, tf: str):
    """(higher-timeframe candles, bin start per bar, bin end per bin) -- cached.

    Candles are indexed by their start (wall clock); ends are start + the bin length.
    """
    if tf not in RULES:
        raise ValueError(f"Unknown timeframe {tf!r}. Choose from: {', '.join(RULES)}")
    key = (id(bars), len(bars), bars.index[0] if len(bars) else None,
           bars.index[-1] if len(bars) else None, tf)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    rule = RULES[tf]
    wall = _wall(bars.index)
    df = pd.DataFrame(bars[COLS].to_numpy(dtype=float), index=wall, columns=COLS)
    htf = df.resample(rule, label="left", closed="left", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    htf = htf.dropna(subset=["open", "high", "low", "close"])
    ends = htf.index + to_offset(rule)
    # which bin each bar belongs to (by its start time)
    bin_of_bar = htf.index.searchsorted(wall, side="right") - 1
    out = (htf, bin_of_bar, ends)
    if len(_CACHE) > 16:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = out
    return out


def closed_position(bars: pd.DataFrame, tf: str) -> np.ndarray:
    """For each bar: the row number of the latest higher-timeframe candle that had
    finished when this bar closed (-1 = none yet)."""
    htf, _, ends = frame(bars, tf)
    known_at = _wall(bars.index) + _bar_step(bars.index)       # a bar is known at its close
    return ends.searchsorted(known_at, side="right") - 1


def higher(bars: pd.DataFrame, tf: str, forming: bool = False) -> pd.DataFrame:
    """Higher-timeframe open/high/low/close/volume for every bar (same index as bars).

    forming=False: the last CLOSED candle (NaN until the first one has closed).
    forming=True:  the candle still forming, built from the bars so far.
    """
    htf, bin_of_bar, _ = frame(bars, tf)
    if forming:
        df = pd.DataFrame(bars[COLS].to_numpy(dtype=float), index=bars.index, columns=COLS)
        g = df.groupby(bin_of_bar, sort=False)
        out = pd.DataFrame({"open": g["open"].transform("first"), "high": g["high"].cummax(),
                            "low": g["low"].cummin(), "close": df["close"],
                            "volume": g["volume"].cumsum()}, index=bars.index)
        return out
    pos = closed_position(bars, tf)
    vals = htf[COLS].to_numpy()
    out = np.full((len(bars), len(COLS)), np.nan)
    ok = pos >= 0
    out[ok] = vals[pos[ok]]
    return pd.DataFrame(out, index=bars.index, columns=COLS)


def on_higher(bars: pd.DataFrame, tf: str, func) -> pd.Series:
    """Compute an indicator on the higher-timeframe candles, then line it up with
    every bar, using only closed candles. Example:

        rsi_4h = on_higher(bars, "4h", lambda b: rsi(b.close, 14))

    func gets the closed higher-timeframe candles (columns open/high/low/close/
    volume) and returns a Series on the same index.
    """
    htf, _, _ = frame(bars, tf)
    values = pd.Series(func(htf), index=htf.index).to_numpy(dtype=float)
    pos = closed_position(bars, tf)
    out = np.full(len(bars), np.nan)
    ok = pos >= 0
    out[ok] = values[pos[ok]]
    return pd.Series(out, index=bars.index)
