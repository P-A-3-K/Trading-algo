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


def trades_payload(fills: pd.DataFrame) -> list:
    """Fills (quantlab.backtest.trades) in the format the page draws as arrows."""
    if fills is None:
        return []
    return [{"time": d.strftime("%Y-%m-%d"), "side": s, "from": float(f), "to": float(t),
             "price": round(float(p), 4)}
            for d, s, f, t, p in zip(fills["date"], fills["side"], fills["from"],
                                     fills["to"], fills["price"])]


START_CAPITAL = 10_000


def equity_payload(strategy_returns: pd.Series, benchmark_returns: pd.Series,
                   start: float = START_CAPITAL) -> dict:
    """Growth of $10,000 for the strategy and buy & hold, plus their drawdowns.

    Drawdown = how far the account is below its own previous peak, in %.
    It's the number that tells you how painful a strategy is to live with.
    """
    def curve(rets):
        eq = start * (1 + rets.fillna(0.0)).cumprod()
        dd = eq / eq.cummax() - 1
        return eq, dd
    s_eq, s_dd = curve(strategy_returns)
    b_eq, b_dd = curve(benchmark_returns)
    pts = lambda x, nd: [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), nd)}
                         for d, v in zip(x.index, x.values)]
    return {"start": start,
            "strategy": pts(s_eq, 2), "benchmark": pts(b_eq, 2),
            "strategy_dd": pts(s_dd * 100, 3), "benchmark_dd": pts(b_dd * 100, 3)}


def render(ticker: str, df: pd.DataFrame, adjusted: bool = True,
           plots: list = None, strategy: str = "", fills: pd.DataFrame = None,
           equity: dict = None) -> str:
    """Build the HTML page for one ticker, optionally with a strategy's lines.

    When plots are given, df must be on the same price basis as the strategy
    ran on (adjusted, as from download.load_bars) so the lines sit on the candles.
    """
    template = (WEB / "candles.html").read_text()
    library = (WEB / "vendor" / "lightweight-charts.standalone.production.js").read_text()
    payload = {"ticker": ticker.upper(), "adjusted": adjusted, "strategy": strategy,
               "bars": candles(df, adjusted), "plots": plots_payload(plots or []),
               "trades": trades_payload(fills), "equity": equity}
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
    plots, fills, equity, label, name = [], None, None, "", ticker.upper()
    if strategy is not None:
        from .backtest import run_backtest, trades
        bars = dl.load_bars(ticker)
        out = strategy.run(bars)
        plots, fills = out.plots, trades(out.positions, bars["close"])
        strat = run_backtest(bars["close"], out.positions)
        bench = run_backtest(bars["close"], pd.Series(1.0, index=bars.index))
        equity = equity_payload(strat.returns, bench.returns)
        label, name = repr(strategy), f"{name}_{type(strategy).__name__}"
        adjusted = True          # strategies run on adjusted bars
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{name}.html"
    path.write_text(PAGE.format(body=render(ticker, df, adjusted, plots, label, fills, equity)))
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
