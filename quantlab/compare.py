"""Compare strategies side by side on one ticker and timeframe (A/B tests).

    python -m quantlab.compare GLD 1h "MtfRsi htf=4h" "MtfRsi htf=1d" RsiReversion
    python -m quantlab.compare GLD 15m "MtfRsi flat_at_close=1" "RsiReversion flat_at_close=1" --hist 5y

Each strategy is a name plus optional settings (name key=value ...), in quotes.
Prints return after and before costs, Sharpe, worst drawdown, time in market,
round trips, win rate and costs, plus buy & hold. Realistic costs.
--hist: intraday history to use (60d, 6m, 1y, 2y, 5y, all; default all).
--from / --to: only part of the history, e.g. --from 2022-01-01 --to 2024-06-01.
Costs (default: realistic): --cost realistic|flat5|none|expensive, or adjust the realistic
model with --spread 0.2 (half-spread in bp, regular hours), --slip 0.02 (share of the bar's
range), --commission 0.5 (bp per trade).
"""
import argparse

import pandas as pd

from . import download as dl
from . import strategies


def parse_strategy(text: str):
    """'MtfRsi htf=1d buy_below=25' -> a strategy object."""
    name, *pairs = text.split()
    settings = dict(p.split("=", 1) for p in pairs)
    return strategies.get(name)(**settings)


def compare(ticker: str, timeframe: str, specs: list, history: str = "all", start=None,
            costs=None, end=None) -> pd.DataFrame:
    from .backtest import run_backtest, simulate
    from .chart import cost_summary
    from .costs import CostModel
    from .metrics import full_stats
    bars = dl.load_bars(ticker, timeframe, history if timeframe != "1d" else None)
    if start:
        ts = pd.Timestamp(start)
        if bars.index.tz is not None and ts.tz is None:
            ts = ts.tz_localize(bars.index.tz)
        bars = bars[bars.index >= ts]
    if end:
        ts = pd.Timestamp(end)
        if bars.index.tz is not None and ts.tz is None:
            ts = ts.tz_localize(bars.index.tz)
        bars = bars[bars.index < ts]
    costs = costs or CostModel()
    prepared = costs.prepare(bars)
    rows = []
    for spec in specs:
        strat = parse_strategy(spec) if isinstance(spec, str) else spec
        bt = simulate(bars, strat.run(bars), costs=prepared)
        st, cs = full_stats(bt.returns, bt.positions), cost_summary(bt)
        rows.append({"strategy": spec if isinstance(spec, str) else repr(spec),
                     "after costs": st["cagr"], "before costs": cs["cagr_gross"],
                     "sharpe": st["sharpe"], "worst DD": st["max_drawdown"],
                     "in market": st["time_in_market"], "round trips": st["round_trips"],
                     "win rate": st["win_rate"], "costs $": cs["costs_paid"]})
    bh = run_backtest(bars["close"], pd.Series(1.0, index=bars.index), costs=prepared).returns
    st = full_stats(bh)
    rows.append({"strategy": "Buy & hold", "after costs": st["cagr"], "before costs": None,
                 "sharpe": st["sharpe"], "worst DD": st["max_drawdown"], "in market": 1.0,
                 "round trips": None, "win rate": None, "costs $": None})
    out = pd.DataFrame(rows).set_index("strategy")
    note = costs.describe()
    if costs.bar_spread and "spread" in bars.columns:
        note = "real spread of every candle (from the data) instead of the fixed spread; " + note
    sl = costs.swap_long_bps if costs.swap_long_bps is not None else bars.attrs.get("swap_long_bps")
    ss = costs.swap_short_bps if costs.swap_short_bps is not None else bars.attrs.get("swap_short_bps")
    if sl or ss:
        note += f"; overnight swaps: long {sl or 0:.2f} bp, short {ss or 0:.2f} bp per night"
    out.attrs.update(start=bars.index[0], end=bars.index[-1], bars=len(bars), costs=note,
                     source=bars.attrs.get("source", "yahoo" if timeframe != "1d" else "daily"))
    return out


def show(df: pd.DataFrame) -> str:
    pct = lambda v: "" if v is None or v != v else f"{v * 100:.1f}%"
    fmt = {"after costs": pct, "before costs": pct, "worst DD": pct, "in market": pct, "win rate": pct,
           "sharpe": lambda v: "" if v is None or v != v else f"{v:.2f}",
           "round trips": lambda v: "" if v is None or v != v else f"{int(v)}",
           "costs $": lambda v: "" if v is None or v != v else f"${v:,.0f}"}
    table = df.copy().astype(object)
    for c, f in fmt.items():
        table[c] = [f(v) for v in df[c]]
    a = df.attrs
    head = f"{a.get('start')} -> {a.get('end')}  ({a.get('bars'):,} bars, {a.get('source')})" + \
        (f"\ncosts: {a['costs']}" if a.get("costs") else "")
    return head + "\n" + table.to_string()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare strategies side by side (A/B test).")
    ap.add_argument("ticker")
    ap.add_argument("timeframe", help="5m 15m 30m 1h 4h 1d")
    ap.add_argument("strategies", nargs="+", help='e.g. "MtfRsi htf=1d" RsiReversion')
    ap.add_argument("--hist", default="all", help="intraday history: 60d 6m 1y 2y 5y all")
    ap.add_argument("--from", dest="start", default=None, help="start date, e.g. 2020-01-01")
    ap.add_argument("--to", dest="end", default=None, help="end date (not included), e.g. 2024-06-01")
    ap.add_argument("--cost", default="realistic", help="realistic, flat5, none or expensive")
    ap.add_argument("--spread", type=float, help="half-spread in bp (regular hours)")
    ap.add_argument("--slip", type=float, help="slippage as a share of the bar's range, e.g. 0.02")
    ap.add_argument("--commission", type=float, help="commission in bp per trade")
    a = ap.parse_args(argv)
    dl.check_timeframe(a.timeframe)
    from dataclasses import replace
    from .costs import PRESETS
    if a.cost not in PRESETS:
        ap.error(f"--cost must be one of {', '.join(PRESETS)}")
    model = PRESETS[a.cost]
    tweaks = {k: v for k, v in (("spread_bps", a.spread), ("slippage_range", a.slip),
                                 ("commission_bps", a.commission)) if v is not None}
    if tweaks:
        model = replace(model, **tweaks)
    print(show(compare(a.ticker, a.timeframe, a.strategies, a.hist, a.start, model, a.end)))


if __name__ == "__main__":
    main()
