"""Data checks: find gaps before they quietly corrupt a backtest.

Two kinds of gap:
  * missing days: the exchange was open, but the file has no row
  * stale days:   the row exists, but nothing traded (zero volume, flat price),
                  usually yesterday's price copied forward by the data vendor

Weekends and exchange holidays are NOT gaps: the market was closed.
That's why we need the real NYSE calendar, not just "every weekday".
"""
import pandas as pd
import pandas_market_calendars as mcal


def trading_days(start, end, exchange="NYSE") -> pd.DatetimeIndex:
    """Every day the exchange was open between start and end (inclusive)."""
    days = mcal.get_calendar(exchange).valid_days(start, end)
    return pd.DatetimeIndex(days.tz_localize(None).normalize(), name="Date")


def find_missing(index: pd.DatetimeIndex, exchange="NYSE") -> pd.DataFrame:
    """Trading days absent from `index`, grouped into runs of consecutive gaps.

    Returns one row per gap: first missing day, last missing day, length.
    """
    index = pd.DatetimeIndex(index).normalize()
    expected = trading_days(index.min(), index.max(), exchange)
    missing = expected.difference(index)
    if missing.empty:
        return pd.DataFrame(columns=["start", "end", "days"])
    # Consecutive missing *trading* days belong to the same gap.
    pos = expected.get_indexer(missing)
    run_id = (pd.Series(pos).diff() != 1).cumsum()
    runs = pd.Series(missing).groupby(run_id.values)
    return pd.DataFrame({"start": runs.min().values, "end": runs.max().values,
                         "days": runs.size().values})


def find_extra(index: pd.DatetimeIndex, exchange="NYSE") -> pd.DatetimeIndex:
    """Rows on days the exchange was closed (a weekend or holiday): bad data."""
    index = pd.DatetimeIndex(index).normalize()
    return index.difference(trading_days(index.min(), index.max(), exchange))


def find_stale(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Days that exist but show no trading: zero volume, or a frozen price
    (close unchanged and high == low)."""
    zero_volume = df["Volume"] == 0
    frozen = (df["Close"].diff() == 0) & (df["High"] == df["Low"])
    return df.index[zero_volume | frozen]


def gap_report(df: pd.DataFrame, name: str = "", exchange="NYSE") -> dict:
    """One-line health check for an OHLCV table indexed by date."""
    missing = find_missing(df.index, exchange)
    return {
        "ticker": name,
        "first": df.index.min().date(), "last": df.index.max().date(),
        "rows": len(df),
        "duplicates": int(df.index.duplicated().sum()),
        "missing_days": int(missing["days"].sum()),
        "longest_gap": int(missing["days"].max()) if len(missing) else 0,
        "extra_days": len(find_extra(df.index, exchange)),
        "stale_days": len(find_stale(df)),
    }


def fill_gaps(prices: pd.Series, max_fill: int = 3, exchange="NYSE") -> pd.Series:
    """Put prices on the full trading calendar, filling short gaps.

    Our policy (see lesson 3):
      * FORWARD-fill gaps of up to `max_fill` trading days: if we don't know
        today's price, the best honest guess is the last price we saw.
        That makes the return 0 on the missing days, and the next real
        price carries the whole move, so the total return is unchanged.
      * NEVER back-fill: that copies a future price into the past
        (look-ahead bias, the bug from lesson 1).
      * Gaps longer than `max_fill` stay as NaN, so you notice them and
        decide by hand (drop that period or that ticker).
    """
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    calendar = trading_days(prices.index.min(), prices.index.max(), exchange)
    on_calendar = prices.reindex(calendar)
    return on_calendar.ffill(limit=max_fill)
