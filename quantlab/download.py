"""Download daily US stock/ETF prices from Yahoo Finance and cache them as CSV.

Yahoo's chart endpoint is unofficial and for personal, non-commercial use
(see docs/data-sources.md). So we download once, save to data/, and read the
saved file afterwards instead of asking Yahoo again on every run.
"""
import re
import time
from datetime import datetime, timedelta, timezone

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
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
    # epoch + offset instead of datetime.fromtimestamp(): Windows can't convert
    # negative (pre-1970) timestamps, and indices like ^GSPC go back to 1927.
    dates = [(EPOCH + timedelta(seconds=t)).date() for t in result["timestamp"]]
    df = pd.DataFrame({
        "Open": quote["open"], "High": quote["high"], "Low": quote["low"],
        "Close": quote["close"],
        "Adj Close": result["indicators"]["adjclose"][0]["adjclose"],
        "Volume": quote["volume"],
    }, index=pd.DatetimeIndex(dates, name="Date"))
    # Yahoo sometimes sends a row of nulls (e.g. a half-finished trading day).
    return df.dropna(subset=["Close", "Adj Close"]).sort_index()


EARLIEST = -2208988800   # 1900-01-01: ask for everything Yahoo has


def fetch(ticker: str, timeout: float = 30) -> pd.DataFrame:
    """Download the full daily history for one ticker."""
    params = {"period1": EARLIEST, "period2": int(time.time()), "interval": "1d",
              "events": "div,split", "includeAdjustedClose": "true"}
    resp = requests.get(URL.format(ticker=clean_ticker(ticker)), params=params,
                        headers=HEADERS, timeout=timeout)
    if resp.status_code == 404:
        raise ValueError(f"Yahoo Finance has no data for {ticker!r}. Check the symbol "
                         "(e.g. BRK-B, not BRK.B; ^GSPC for the S&P 500 index).")
    resp.raise_for_status()
    return parse_chart(resp.json())


TICKER_RE = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,14}$")


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
        was_cached = False
        try:
            was_cached = csv_path(t).exists() and not refresh
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


def load_bars(ticker: str) -> pd.DataFrame:
    """Daily bars for strategies: open/high/low/close/volume, all adjusted.

    The whole bar is scaled by Adj Close / Close, so every column is on the
    same dividend- and split-adjusted basis the backtest uses.
    """
    df = pd.read_csv(download(ticker), parse_dates=["Date"], index_col="Date")
    factor = df["Adj Close"] / df["Close"]
    bars = df[["Open", "High", "Low", "Close"]].mul(factor, axis=0)
    bars["Volume"] = df["Volume"]
    bars.columns = [c.lower() for c in bars.columns]
    return bars


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    refresh = "--refresh" in args
    if "--popular" in args:
        from .universe import popular_tickers
        tickers = popular_tickers()
    else:
        tickers = [a for a in args if not a.startswith("--")] or ["SPY"]
    for t, result in download_many(tickers, refresh=refresh).items():
        print(f"{t:6s} {result}")
