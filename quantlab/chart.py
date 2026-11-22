"""Candlestick charts: turn a ticker's saved prices into an interactive page.

    python -m quantlab.chart SPY          -> reports/SPY.html (open in a browser)

The page is self-contained: the chart library (TradingView Lightweight
Charts, Apache-2.0) and the price data are written into the file, so it
works offline and can be emailed or published as-is.
"""
import json
from pathlib import Path

import pandas as pd

from . import download as dl

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent / "web"
REPORTS = ROOT / "reports"


def adjusted_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Scale Open/High/Low/Close by Adj Close / Close for each day.

    Yahoo's OHLC are split-adjusted but not dividend-adjusted, while our
    backtests use Adj Close. Scaling the whole bar by the same factor keeps
    the candle shapes and puts the chart on the same price basis as the
    backtest, so a buy arrow sits exactly on the price it traded at.
    """
    factor = df["Adj Close"] / df["Close"]
    out = df[["Open", "High", "Low", "Close"]].mul(factor, axis=0)
    out["Volume"] = df["Volume"]
    return out


def candles(df: pd.DataFrame, adjusted: bool = True) -> list:
    """Rows in the format the chart library wants: time + OHLC (+ volume)."""
    bars = adjusted_ohlc(df) if adjusted else df
    return [
        {"time": d.strftime("%Y-%m-%d"), "open": round(o, 4), "high": round(h, 4),
         "low": round(lo, 4), "close": round(c, 4), "volume": int(v)}
        for d, o, h, lo, c, v in zip(bars.index, bars["Open"], bars["High"],
                                     bars["Low"], bars["Close"], bars["Volume"])
    ]


def render(ticker: str, df: pd.DataFrame, adjusted: bool = True) -> str:
    """Build the HTML page for one ticker."""
    template = (WEB / "candles.html").read_text()
    library = (WEB / "vendor" / "lightweight-charts.standalone.production.js").read_text()
    payload = {"ticker": ticker.upper(), "adjusted": adjusted,
               "bars": candles(df, adjusted)}
    return (template
            .replace("__TICKER__", ticker.upper())
            .replace("/*__LIBRARY__*/", library)
            .replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":"))))


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
</head><body>
{body}
</body></html>
"""


def save(ticker: str, adjusted: bool = True, out_dir: Path = REPORTS) -> Path:
    """Write reports/<TICKER>.html (downloads the data first if needed)."""
    df = pd.read_csv(dl.download(ticker), parse_dates=["Date"], index_col="Date")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{ticker.upper()}.html"
    path.write_text(PAGE.format(body=render(ticker, df, adjusted)))
    return path


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:] or ["SPY"]:
        print(save(t))
