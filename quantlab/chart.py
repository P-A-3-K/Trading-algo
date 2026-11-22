"""Candlestick charts: turn a ticker's saved prices into an interactive page.

    python -m quantlab.chart SPY                         -> reports/SPY.html
    python -m quantlab.chart SPY SmaCrossover fast=20    -> candles + the strategy's lines

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


def line_points(values: pd.Series) -> list:
    """A plot's values as chart points; gaps (NaN, e.g. warm-up days) are skipped."""
    values = values.dropna()
    return [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), 6)}
            for d, v in zip(values.index, values.values)]


def plots_payload(plots: list) -> list:
    """Strategy plots (quantlab.strategy.Plot) in the format the page draws."""
    return [{"name": p.name, "pane": p.pane, "color": p.color,
             "points": line_points(p.values)} for p in plots]


def render(ticker: str, df: pd.DataFrame, adjusted: bool = True,
           plots: list = None, strategy: str = "") -> str:
    """Build the HTML page for one ticker, optionally with a strategy's lines.

    When plots are given, df must be on the same price basis as the strategy
    ran on (adjusted, as from download.load_bars) so the lines sit on the candles.
    """
    template = (WEB / "candles.html").read_text()
    library = (WEB / "vendor" / "lightweight-charts.standalone.production.js").read_text()
    payload = {"ticker": ticker.upper(), "adjusted": adjusted, "strategy": strategy,
               "bars": candles(df, adjusted), "plots": plots_payload(plots or [])}
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


def save(ticker: str, strategy=None, adjusted: bool = True, out_dir: Path = REPORTS) -> Path:
    """Write reports/<TICKER>[_<Strategy>].html (downloads the data if needed)."""
    df = pd.read_csv(dl.download(ticker), parse_dates=["Date"], index_col="Date")
    plots, label, name = [], "", ticker.upper()
    if strategy is not None:
        plots = strategy.run(dl.load_bars(ticker)).plots
        label, name = repr(strategy), f"{name}_{type(strategy).__name__}"
        adjusted = True          # strategies run on adjusted bars
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{name}.html"
    path.write_text(PAGE.format(body=render(ticker, df, adjusted, plots, label)))
    return path


def _parse_args(argv):
    """SPY [StrategyName [key=value ...]]"""
    from .strategies import BUILTIN
    ticker, strategy = (argv[0] if argv else "SPY"), None
    if len(argv) > 1:
        by_name = {cls.__name__.lower(): cls for cls in BUILTIN}
        cls = by_name.get(argv[1].lower())
        if cls is None:
            raise SystemExit(f"Unknown strategy {argv[1]!r}. Choose from: "
                             + ", ".join(c.__name__ for c in BUILTIN))
        settings = dict(a.split("=", 1) for a in argv[2:])
        strategy = cls(**settings)
    return ticker, strategy


if __name__ == "__main__":
    import sys
    print(save(*_parse_args(sys.argv[1:])))
