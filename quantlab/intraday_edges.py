"""Intraday edge finder: textbook day-trading strategies on liquid ETFs, judged on two periods.

    python -m quantlab.intraday_edges                 # SPY QQQ IWM DIA EWG EWJ GLD SLV USO FXE FXY
    python -m quantlab.intraday_edges SPY QQQ GLD     # only these
    python -m quantlab.intraday_edges --only ORB      # only strategies whose name contains "ORB"

Why ETFs: FTMO keeps only ~1 year of 5-minute history, Alpaca has 5-minute candles since 2016
for ETFs that mirror FTMO's markets (SPY ~ US500, QQQ ~ US100, GLD ~ gold, EWG ~ GER40...).
Every strategy here is flat by the close, so FTMO's overnight swaps don't apply; costs are
FTMO-like (half-spread 0.5 bp + slippage of 5% of a 5-minute candle's range), also shown with
low costs (0.25 bp + 2%) and none, to see how much the verdict depends on execution.
Same verdict as the daily edge finder: A = 2017-2024 vs B = 2025-today, and a portfolio of all
markets (equal risk, weights from A only). Needs Alpaca keys (Data page in the app).
"""
import sys

import pandas as pd

from . import download as dl
from . import edges as E

A_START = pd.Timestamp("2017-01-01")

STRATEGIES = {
    "ORB 30m": ("OpeningRangeBreakout", {"range_minutes": 30, "rr": 0.0}),
    "First/last half hour": ("IntradayMomentum", {}),
    "Late-day trend": ("IntradayMomentum", {"signal_from": "open", "signal_minutes": 300,
                                            "minutes_before_close": 60}),
    "Gap fade": ("GapFade", {}),
    "VWAP reversion": ("VwapReversion", {}),
}

GROUPS = {
    "US indices": ["SPY", "QQQ", "IWM", "DIA"],
    "Other indices": ["EWG", "EWJ"],
    "Metals": ["GLD", "SLV"],
    "Energy": ["USO"],
    "FX": ["FXE", "FXY"],
}


def cost_models() -> dict:
    from .costs import CostModel, flat
    ftmo_like = CostModel(spread_bps=0.5, extended_spread_bps=0.5, slippage_range=0.05, impact=0,
                          bar_spread=False, swap_long_bps=0.0, swap_short_bps=0.0)
    low = CostModel(spread_bps=0.25, extended_spread_bps=0.25, slippage_range=0.02, impact=0,
                    bar_spread=False, swap_long_bps=0.0, swap_short_bps=0.0)
    return {"full": ftmo_like, "low costs": low, "no costs": flat(0)}


def load_alpaca(ticker: str) -> pd.DataFrame:
    bars = dl.load_bars(ticker, "5m", "all")
    if bars.index[0] > pd.Timestamp("2019-01-01", tz=bars.index.tz):
        raise ValueError(f"only {bars.index[0]:%Y-%m-%d} onward: connect Alpaca (Data page) for history since 2016")
    return bars


def run(symbols=None, loader=load_alpaca, progress=print, only: str = None):
    """only: run just the strategies whose name contains this text (e.g. "ORB")."""
    chosen = {k: v for k, v in STRATEGIES.items() if not only or only.lower() in k.lower()} or STRATEGIES
    return E.run(symbols, loader=loader, progress=progress, strategy_set=chosen, costs=cost_models(),
                 groups=GROUPS, a_start=A_START, unit="candles")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return
    print("Loading 5-minute candles (first run downloads years of history per ETF: a few minutes each)...")
    only = None
    if "--only" in args:
        i = args.index("--only")
        only, args = args[i + 1], args[:i] + args[i + 2:]
    markets, port, groups = run([a.upper() for a in args] or None, only=only)
    if markets.empty:
        print("  No markets loaded. Are your Alpaca keys set (Data page)?")
        return
    print()
    print(E.show(markets, port, groups))
    path = dl.DATA_DIR / "edges" / "intraday_edges.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    markets.to_csv(path, index=False)
    print(f"\n  Saved per-market results in {path}")


if __name__ == "__main__":
    main()
