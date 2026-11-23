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


EPOCH = pd.Timestamp("1970-01-01")


def tkey(d: pd.Timestamp):
    """Time value for the chart library.

    Daily bars: "YYYY-MM-DD". Intraday bars (time-zone aware, New York): seconds
    since 1970 of the New York WALL-CLOCK time, so the chart (which shows UTC)
    displays exchange time: a 09:30 bar shows as 09:30.
    """
    if d.tzinfo is None:
        return d.strftime("%Y-%m-%d")
    return int((d.tz_localize(None) - EPOCH).total_seconds())


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
        {"time": tkey(d), "open": round(o, 4), "high": round(h, 4),
         "low": round(lo, 4), "close": round(c, 4), "volume": int(v)}
        for d, o, h, lo, c, v in zip(bars.index, bars["Open"], bars["High"],
                                     bars["Low"], bars["Close"], bars["Volume"])
    ]


def line_points(values: pd.Series) -> list:
    """A plot's values as chart points.

    Leading/trailing gaps (warm-up days) are dropped; gaps in the middle become
    empty points so the chart breaks the line instead of joining across them
    (e.g. a stop-loss line that only exists while a trade is open).
    """
    valid = values.notna()
    if not valid.any():
        return []
    first, last = valid.idxmax(), valid[::-1].idxmax()
    span = values.loc[first:last]
    return [{"time": tkey(d)} if v != v else
            {"time": tkey(d), "value": round(float(v), 6)}
            for d, v in zip(span.index, span.values)]


def plots_payload(plots: list) -> list:
    """Strategy plots (quantlab.strategy.Plot) in the format the page draws."""
    return [{"name": p.name, "pane": p.pane, "color": p.color, "style": p.style,
             "points": line_points(p.values)} for p in plots]


def trades_payload(fills: pd.DataFrame) -> list:
    """Fills (quantlab.backtest.trades) in the format the page draws as arrows."""
    if fills is None:
        return []
    return [{"time": tkey(d), "side": s, "from": float(f), "to": float(t),
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
    pts = lambda x, nd: [{"time": tkey(d), "value": round(float(v), nd)}
                         for d, v in zip(x.index, x.values)]
    return {"start": start,
            "strategy": pts(s_eq, 2), "benchmark": pts(b_eq, 2),
            "strategy_dd": pts(s_dd * 100, 3), "benchmark_dd": pts(b_dd * 100, 3)}


# (key, label, explanation, format, which is better)
STAT_ROWS = [
    ("final_value", "Final value", "What $10,000 grew to", "money", "higher"),
    ("cagr", "Return per year", "Compound annual growth (CAGR), after costs", "pct", "higher"),
    ("cagr_gross", "Return per year before costs", "The same trades if trading were free", "pct", "higher"),
    ("volatility", "Volatility", "How much it swings, per year", "pct", "lower"),
    ("sharpe", "Sharpe ratio", "Return per unit of risk; above 1 is good", "num", "higher"),
    ("sortino", "Sortino ratio", "Like Sharpe, but only falls count as risk", "num", "higher"),
    ("max_drawdown", "Worst drawdown", "Biggest fall from a peak", "pct", "higher"),
    ("longest_drawdown_days", "Longest underwater", "Most time spent below a previous peak", "days", "lower"),
    ("calmar", "Calmar ratio", "Return per year ÷ worst drawdown", "num", "higher"),
    ("time_in_market", "Time in market", "Share of days holding a position", "pct", None),
    ("fills", "Trades", "Buy and sell orders", "int", None),
    ("costs_paid", "Costs paid", "Commission, spread, slippage and impact, in $", "money", "lower"),
    ("win_rate", "Win rate", "Share of round trips that made money", "pct", None),
    ("avg_win", "Average win", "Mean return of winning round trips", "pct", None),
    ("avg_loss", "Average loss", "Mean return of losing round trips", "pct", None),
    ("avg_days_held", "Average hold", "Trading days per round trip", "days", None),
]


def stats_payload(strategy: dict, benchmark: dict, start: float = START_CAPITAL) -> list:
    """Rows for the stats table: strategy next to buy & hold.

    Trade statistics (win rate etc.) are only shown for the strategy:
    buy & hold makes one trade, so they mean nothing there.
    """
    trade_only = {"win_rate", "avg_win", "avg_loss", "avg_days_held"}
    rows = []
    for key, label, note, fmt, better in STAT_ROWS:
        get = lambda st: start * (1 + st["total_return"]) if key == "final_value" else st.get(key)
        bench = None if key in trade_only else get(benchmark)
        val = lambda v: None if v is None or v != v else float(v)   # NaN -> None
        rows.append({"key": key, "label": label, "note": note, "fmt": fmt, "better": better,
                     "strategy": val(get(strategy)), "benchmark": val(bench)})
    return rows


def luck_payload(strategy_returns: pd.Series, benchmark_returns: pd.Series,
                 samples: int = 5000, block: int = None) -> dict:
    """Block bootstrap of strategy vs buy & hold, for the 'Could it be luck?' panel.

    Blocks: about a month of daily bars, or one trading day of intraday bars.
    """
    from .bootstrap import bootstrap, histogram
    from .metrics import bars_per_day, periods_per_year
    bpd = bars_per_day(strategy_returns.index)
    block = block or (21 if bpd <= 1 else int(round(bpd)))
    st, be, beat = bootstrap(strategy_returns.to_numpy(), benchmark_returns.to_numpy(),
                             samples, block, ppy=periods_per_year(strategy_returns))
    f = lambda d: {k: float(v) for k, v in d.items()}
    return {"samples": samples, "block": block, "strategy": f(st.summary()),
            "benchmark": f(be.summary()), "beats": beat, "hist": histogram(st, be)}


def render(ticker: str, df: pd.DataFrame, adjusted: bool = True,
           plots: list = None, strategy: str = "", fills: pd.DataFrame = None,
           equity: dict = None, stats: list = None) -> str:
    """Build the HTML page for one ticker, optionally with a strategy's lines.

    When plots are given, df must be on the same price basis as the strategy
    ran on (adjusted, as from download.load_bars) so the lines sit on the candles.
    """
    payload = {"ticker": ticker.upper(), "adjusted": adjusted, "strategy": strategy,
               "bars": candles(df, adjusted), "plots": plots_payload(plots or []),
               "trades": trades_payload(fills), "equity": equity,
               "stats": stats}
    return render_payload(payload)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
</head><body>
{body}
</body></html>
"""


def cost_summary(bt, start: float = START_CAPITAL, part: slice = slice(None)) -> dict:
    """Return per year before costs, and total $ paid in costs (from a BacktestResult).

    part: only these bars (e.g. the test period), as if $10,000 started there."""
    from .metrics import cagr
    r = bt.returns.iloc[part]
    paid = (bt.costs if bt.costs is not None else pd.Series(0.0, index=bt.returns.index)).iloc[part]
    equity_before = start * (1 + r).cumprod().shift(1).fillna(1.0)
    return {"cagr_gross": cagr(r + paid), "costs_paid": float((paid * equity_before).sum())}


def when(ts) -> str:
    """A bar's time as text: '2024-03-01' for daily bars, '2024-03-01 09:30' intraday."""
    ts = pd.Timestamp(ts)
    return ts.strftime("%Y-%m-%d %H:%M") if (ts.hour or ts.minute) else str(ts.date())


def split_index(index: pd.DatetimeIndex, test_from) -> int:
    """Position of the first bar at or after `test_from` (text like '2015-06-01 09:30')."""
    ts = pd.Timestamp(test_from)
    if index.tz is not None and ts.tz is None:
        ts = ts.tz_localize(index.tz)
    elif index.tz is None and ts.tz is not None:
        ts = ts.tz_localize(None)
    return int(index.searchsorted(ts))


MIN_PERIOD_BARS = 20


def build_payload(ticker: str, strategy=None, adjusted: bool = True,
                  timeframe: str = "1d", costs=None, test_from=None) -> dict:
    """Everything the chart page shows, for one ticker, timeframe and optional strategy.

    test_from: where the optimizer's test (out-of-sample) period started, when the
    settings were opened from a sweep. Stats and the luck check are then also given
    for the training part and the test part separately: only the test part is
    honest, because the settings were picked on the training part.

    Used both for saved report files and by the web app (quantlab.app).
    """
    from .metrics import bars_per_day
    ticker = dl.clean_ticker(ticker)
    dl.check_timeframe(timeframe)
    intraday = timeframe != "1d"
    if intraday:
        bars = dl.load_bars(ticker, timeframe)
        df = bars.rename(columns=str.capitalize)
        df["Adj Close"] = df["Close"]          # intraday prices are as traded
        adjusted = False
    else:
        df = pd.read_csv(dl.download(ticker), parse_dates=["Date"], index_col="Date")
    bpd = bars_per_day(df.index)
    plots, fills, equity, stats, luck, label = [], None, None, None, None, ""
    periods, split = None, None
    if strategy is not None:
        from .backtest import fills_of, run_backtest, simulate
        from .metrics import full_stats
        if not intraday:
            bars = dl.load_bars(ticker)
        out = strategy.run(bars)
        plots, fills = out.plots, fills_of(bars, out)
        from .costs import CostModel
        costs = costs or CostModel()
        prepared = costs.prepare(bars)
        strat = simulate(bars, out, costs=prepared)
        bench = run_backtest(bars["close"], pd.Series(1.0, index=bars.index), costs=prepared)
        equity = equity_payload(strat.returns, bench.returns)
        def period_stats(part=slice(None)):
            r, held = strat.returns.iloc[part], strat.positions.iloc[part]
            n_fills = int(fills["date"].between(r.index[0], r.index[-1]).sum()) if len(fills) else 0
            s_st = {**full_stats(r, held, n_fills), **cost_summary(strat, part=part)}
            b_st = {**full_stats(bench.returns.iloc[part], bench.positions.iloc[part], 1),
                    **cost_summary(bench, part=part)}
            for st in (s_st, b_st):                # bar counts -> trading days
                for k in ("longest_drawdown_days", "avg_days_held"):
                    if st.get(k) is not None:
                        st[k] = st[k] / bpd
            return stats_payload(s_st, b_st)

        stats = period_stats()
        luck = luck_payload(strat.returns, bench.returns)
        if test_from:
            cut = split_index(bars.index, test_from)
            if cut < MIN_PERIOD_BARS or len(bars) - cut < MIN_PERIOD_BARS:
                raise ValueError(f"The test period starting {test_from} leaves under {MIN_PERIOD_BARS} bars "
                                 "on one side; open the chart without it.")
            idx = bars.index
            split = {"time": tkey(idx[cut]), "label": when(idx[cut]),
                     "train": [when(idx[0]), when(idx[cut - 1])], "test": [when(idx[cut]), when(idx[-1])]}
            periods = {"all": {"stats": stats, "luck": luck}}
            for name, part in (("train", slice(None, cut)), ("test", slice(cut, None))):
                periods[name] = {"stats": period_stats(part),
                                 "luck": luck_payload(strat.returns.iloc[part], bench.returns.iloc[part])}
        label = repr(strategy)
        adjusted = adjusted if intraday else True  # daily strategies run on adjusted bars
    return {"ticker": ticker, "adjusted": adjusted, "strategy": label,
            "costs": costs.describe() if (strategy is not None and costs) else "",
            "timeframe": timeframe, "intraday": intraday, "bars_per_day": bpd,
            "is_index": ticker.startswith("^"),
            "bars": candles(df, adjusted), "plots": plots_payload(plots),
            "trades": trades_payload(fills), "equity": equity, "stats": stats, "luck": luck,
            "split": split, "periods": periods}


def render_payload(payload: dict, page_title: str = None) -> str:
    """Fill the page template with a payload from build_payload()."""
    template = (WEB / "candles.html").read_text(encoding="utf-8")
    library = (WEB / "vendor" / "lightweight-charts.standalone.production.js").read_text(encoding="utf-8")
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return (template
            .replace("__TICKER__", page_title or payload["ticker"])
            .replace("/*__LIBRARY__*/", library)
            .replace("/*__DATA__*/null", data))


def save(ticker: str, strategy=None, adjusted: bool = True, out_dir: Path = REPORTS) -> Path:
    """Write reports/<TICKER>[_<Strategy>].html (downloads the data if needed)."""
    payload = build_payload(ticker, strategy, adjusted)
    name = payload["ticker"] + (f"_{type(strategy).__name__}" if strategy is not None else "")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{name}.html"
    path.write_text(PAGE.format(body=render_payload(payload)), encoding="utf-8")
    return path


def _parse_args(argv):
    """SPY [StrategyName [key=value ...]]"""
    from . import strategies
    ticker, strategy = (argv[0] if argv else "SPY"), None
    if len(argv) > 1:
        try:
            cls = strategies.get(argv[1])
        except ValueError as e:
            raise SystemExit(str(e))
        settings = dict(a.split("=", 1) for a in argv[2:])
        strategy = cls(**settings)
    return ticker, strategy


if __name__ == "__main__":
    import sys
    print(save(*_parse_args(sys.argv[1:])))
