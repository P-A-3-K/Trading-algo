"""Download daily US stock/ETF prices from Yahoo Finance and cache them as CSV.

Yahoo's chart endpoint is unofficial and for personal, non-commercial use
(see docs/data-sources.md). So we download once, save to data/, and read the
saved file afterwards instead of asking Yahoo again on every run.
"""
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
HEADERS = {"User-Agent": "Mozilla/5.0 (QuantLab learning project)"}
COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def parse_chart(payload: dict) -> pd.DataFrame:
    """Turn Yahoo's chart JSON into a tidy daily table indexed by date."""
    chart = payload["chart"]
    if chart.get("error"):
        raise ValueError(f"Yahoo error: {chart['error']}")
    result = chart["result"][0]
    quote = result["indicators"]["quote"][0]
    dates = [datetime.fromtimestamp(t, timezone.utc).date() for t in result["timestamp"]]
    df = pd.DataFrame({
        "Open": quote["open"], "High": quote["high"], "Low": quote["low"],
        "Close": quote["close"],
        "Adj Close": result["indicators"]["adjclose"][0]["adjclose"],
        "Volume": quote["volume"],
    }, index=pd.DatetimeIndex(dates, name="Date"))
    # Yahoo sometimes sends a row of nulls (e.g. a half-finished trading day).
    return df.dropna(subset=["Close", "Adj Close"]).sort_index()


def fetch(ticker: str, timeout: float = 30) -> pd.DataFrame:
    """Download the full daily history for one ticker."""
    params = {"period1": 0, "period2": int(time.time()), "interval": "1d",
              "events": "div,split", "includeAdjustedClose": "true"}
    resp = requests.get(URL.format(ticker=clean_ticker(ticker)), params=params,
                        headers=HEADERS, timeout=timeout)
    if resp.status_code == 404:
        raise ValueError(f"Yahoo Finance has no data for {ticker!r}. Check the symbol "
                         "(e.g. BRK-B, not BRK.B; ^GSPC for the S&P 500 index).")
    resp.raise_for_status()
    return parse_chart(resp.json())


TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-^=]{0,14}$")


def clean_ticker(ticker: str) -> str:
    """Upper-case a ticker and reject anything that isn't a plausible symbol.

    Tickers become file names and URLs, so this stops input like '../secret'
    from reaching files outside data/. Allowed: letters, digits, . - ^ =
    (e.g. BRK-B, ^GSPC, EURUSD=X), up to 15 characters.
    """
    t = str(ticker).strip().upper()
    if not TICKER_RE.match(t):
        raise ValueError(f"Not a valid ticker: {ticker!r}")
    return t


def csv_path(ticker: str) -> Path:
    return DATA_DIR / f"{clean_ticker(ticker)}.csv"


def cached_tickers() -> list:
    """Tickers already downloaded into data/, alphabetically."""
    return sorted(p.stem for p in DATA_DIR.glob("*.csv") if TICKER_RE.match(p.stem))


def download(ticker: str, refresh: bool = False) -> Path:
    """Save one ticker to data/<TICKER>.csv. Skips the download if cached."""
    path = csv_path(ticker)
    if path.exists() and not refresh:
        return path
    df = fetch(ticker)
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, float_format="%.6f")
    return path


def download_many(tickers, refresh=False, pause: float = 1.0) -> dict:
    """Download several tickers politely (a pause between requests).

    Returns {ticker: path or the error message}, so one bad ticker
    doesn't stop the rest.
    """
    results = {}
    for i, t in enumerate(tickers):
        was_cached = csv_path(t).exists() and not refresh
        try:
            results[t] = download(t, refresh=refresh)
        except Exception as exc:  # report and keep going
            results[t] = f"failed: {exc}"
        if not was_cached and i < len(tickers) - 1:
            time.sleep(pause)
    return results


def load(ticker: str, adjusted: bool = True) -> pd.Series:
    """Load a cached ticker as a price series (downloads it if missing).

    adjusted=True uses Adj Close, which folds in dividends and splits.
    Always backtest on adjusted prices.
    """
    df = pd.read_csv(download(ticker), parse_dates=["Date"], index_col="Date")
    col = "Adj Close" if adjusted else "Close"
    return df[col].rename(ticker.upper())


def load_bars(ticker: str, timeframe: str = "1d") -> pd.DataFrame:
    """Bars for strategies: open/high/low/close/volume.

    Daily ("1d"): adjusted for dividends and splits (the whole bar is scaled by
    Adj Close / Close, so every column is on the backtest's price basis).
    Intraday ("5m", "15m", "30m", "1h", "4h"): see load_intraday().
    """
    if timeframe != "1d":
        return load_intraday(ticker, timeframe)
    df = pd.read_csv(download(ticker), parse_dates=["Date"], index_col="Date")
    factor = df["Adj Close"] / df["Close"]
    bars = df[["Open", "High", "Low", "Close"]].mul(factor, axis=0)
    bars["Volume"] = df["Volume"]
    bars.columns = [c.lower() for c in bars.columns]
    return bars


# ---- Intraday candles (5m ... 4h), including pre-market and after-hours -------------------
#
# Yahoo only serves recent intraday history (5m: last 60 days, 1h: last 2 years), so
# every download is MERGED into the saved file: your history grows the longer you use it.
# 15m and 30m candles are built from 5m, and 4h from 1h, so all timeframes agree.

EXCHANGE_TZ = "America/New_York"
TIMEFRAMES = {           # name: (how it's built, label)
    "5m": ("yahoo:5m:60d", "5 minutes"),
    "15m": ("from:5m:15min", "15 minutes"),
    "30m": ("from:5m:30min", "30 minutes"),
    "1h": ("yahoo:1h:730d", "1 hour"),
    "4h": ("from:1h:4h", "4 hours"),
    "1d": ("daily", "1 day"),
}
INTRADAY_MAX_AGE = 15 * 60          # re-download intraday data older than 15 minutes


def check_timeframe(tf: str) -> str:
    if tf not in TIMEFRAMES:
        raise ValueError(f"Unknown timeframe {tf!r}. Choose from: {', '.join(TIMEFRAMES)}")
    return tf


def intraday_path(ticker: str, interval: str) -> Path:
    return DATA_DIR / interval / f"{clean_ticker(ticker)}.csv"


def parse_intraday(payload: dict) -> pd.DataFrame:
    """Yahoo intraday JSON -> OHLCV indexed by New York time (bar start)."""
    chart = payload["chart"]
    if chart.get("error"):
        raise ValueError(f"Yahoo error: {chart['error'].get('description', chart['error'])}")
    result = chart["result"][0]
    quote = result["indicators"]["quote"][0]
    idx = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(EXCHANGE_TZ)
    df = pd.DataFrame({"Open": quote["open"], "High": quote["high"], "Low": quote["low"],
                       "Close": quote["close"], "Volume": quote["volume"]},
                      index=pd.DatetimeIndex(idx, name="Datetime"))
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    # Yahoo appends a live quote stamped e.g. 19:59:59: not a real bar start.
    df = df[df.index.second == 0]
    df["Volume"] = df["Volume"].fillna(0)
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_intraday(ticker: str, interval: str, period: str, timeout: float = 30) -> pd.DataFrame:
    params = {"interval": interval, "range": period, "includePrePost": "true"}
    resp = requests.get(URL.format(ticker=clean_ticker(ticker)), params=params,
                        headers=HEADERS, timeout=timeout)
    if resp.status_code == 404:
        raise ValueError(f"Yahoo Finance has no data for {ticker!r}.")
    resp.raise_for_status()
    return parse_intraday(resp.json())


def read_intraday(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="Datetime")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(EXCHANGE_TZ)
    return df


def merge_bars(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Keep every bar ever downloaded; where both have a bar, the newer download wins."""
    both = pd.concat([old, new])
    return both[~both.index.duplicated(keep="last")].sort_index()


def download_intraday(ticker: str, interval: str, refresh: bool = False) -> Path:
    """Fetch recent bars from Yahoo and merge them into data/<interval>/<TICKER>.csv.

    Re-downloads when the file is older than 15 minutes. If Yahoo can't be
    reached, the saved file is used as it is (so it works offline too).
    """
    source = TIMEFRAMES[interval][0]
    _, yint, period = source.split(":")
    path = intraday_path(ticker, interval)
    fresh = path.exists() and time.time() - path.stat().st_mtime < INTRADAY_MAX_AGE
    if fresh and not refresh:
        return path
    try:
        new = fetch_intraday(ticker, yint, period)
    except requests.RequestException as e:
        if path.exists():
            print(f"  note: couldn't update {ticker} {interval} ({e}); using the saved data")
            return path
        raise
    merged = merge_bars(read_intraday(path), new) if path.exists() else new
    path.parent.mkdir(parents=True, exist_ok=True)
    out = merged.copy()
    out.index = out.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    out.index.name = "Datetime"
    out.to_csv(path, float_format="%.6f")
    return path


def resample_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Combine candles, aligned to the clock in New York time (bins start at midnight).

    4h bins: 0-4, 4-8, 8-12, 12-16, 16-20. Bins with no trades are dropped.
    """
    # Group on the New York WALL CLOCK (time zone removed), so daylight-saving
    # changes don't shift the bins by an hour; then put the time zone back.
    wall = df.tz_localize(None) if df.index.tz is not None else df
    agg = wall.resample(rule, label="left", closed="left", origin="start_day").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    agg = agg.dropna(subset=["Open", "High", "Low", "Close"])
    if df.index.tz is not None:
        agg.index = agg.index.tz_localize(df.index.tz, ambiguous="NaT", nonexistent="NaT")
        agg = agg[agg.index.notna()]
    return agg


def clean_spikes(df: pd.DataFrame, min_move: float = 0.02, factor: float = 10.0,
                 lookback: int = 50) -> pd.DataFrame:
    """Trim wicks that are bad prints (common in thin pre-market/after-hours data).

    A wick is a bad print when it sticks out beyond the bar's body (open/close) by
    more than `min_move` (2%) AND more than `factor` x the typical bar range of the
    previous `lookback` bars. It is trimmed back to the body; open and close stay.
    Only EARLIER bars are used to judge, so cleaning never peeks at the future.
    Returns a copy with an attribute .attrs["spikes_trimmed"].
    """
    df = df.copy()
    body_lo = df[["Open", "Close"]].min(axis=1)
    body_hi = df[["Open", "Close"]].max(axis=1)
    typical = ((df["High"] - df["Low"]) / df["Close"]).rolling(lookback, min_periods=10).median().shift(1)
    limit = (typical * factor).clip(lower=min_move).fillna(min_move * 2)
    bad_lo = (body_lo - df["Low"]) / df["Close"] > limit
    bad_hi = (df["High"] - body_hi) / df["Close"] > limit
    df.loc[bad_lo, "Low"] = body_lo[bad_lo]
    df.loc[bad_hi, "High"] = body_hi[bad_hi]
    df.attrs["spikes_trimmed"] = int(bad_lo.sum() + bad_hi.sum())
    return df


def load_intraday(ticker: str, timeframe: str) -> pd.DataFrame:
    """Intraday bars (pre-market and after-hours included), New York time.

    Prices are as traded: not adjusted for dividends (Yahoo doesn't adjust
    intraday data), which matters little over weeks or months.
    """
    check_timeframe(timeframe)
    source = TIMEFRAMES[timeframe][0]
    if source.startswith("from:"):
        _, base, rule = source.split(":")
        raw = clean_spikes(read_intraday(download_intraday(ticker, base)))
        df = resample_bars(raw, rule)
        df.attrs["spikes_trimmed"] = raw.attrs["spikes_trimmed"]
    else:
        df = clean_spikes(read_intraday(download_intraday(ticker, timeframe)))
    bars = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    bars.columns = [c.lower() for c in bars.columns]
    bars.attrs["spikes_trimmed"] = df.attrs.get("spikes_trimmed", 0)
    return bars


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] or ["SPY"]
    for t, result in download_many(tickers).items():
        print(f"{t:6s} {result}")
