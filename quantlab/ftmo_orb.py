"""FTMO ORB check: does the opening range breakout's edge survive FTMO's REAL costs?

    python -m quantlab.ftmo_orb                  # all pairs below
    python -m quantlab.ftmo_orb US100.cash EURUSD

For every FTMO market (MT5 open, logged into FTMO) next to the ETF that stands in for it:
  * edge per trade  = average gross result of an ORB trade on the ETF, 2017-today (Alpaca),
    in basis points (bp) of the position: the long-run edge, before costs;
  * cost per trade  = what a round trip really costs on FTMO: the spread of every candle
    from FTMO's own data (2026), plus slippage of 0%, 2% or 5% of a 5-minute candle's range;
  * FTMO 2026       = the same ORB on FTMO's own candles, after real costs (one year: noisy).
A market is worth trading when its edge per trade is clearly bigger than its cost per trade.
"""
import sys
import warnings

import numpy as np
import pandas as pd

from . import download as dl

# FTMO symbol -> (ETF stand-in, ORB session)
PAIRS = {
    "US500.cash": ("SPY", "US"), "US100.cash": ("QQQ", "US"), "US30.cash": ("DIA", "US"),
    "US2000.cash": ("IWM", "US"), "GER40.cash": ("EWG", "US"), "XAUUSD": ("GLD", "US"),
    "XAGUSD": ("SLV", "US"), "USOIL.cash": ("USO", "US"), "EURUSD": ("FXE", "US"),
    "USDJPY": ("FXY", "US"), "JP225.cash": ("EWJ", "US"),
}
SLIPS = (0.0, 0.02, 0.05)


def orb(session: str):
    from . import strategies
    return strategies.get("OpeningRangeBreakout")(range_minutes=30, rr=0.0, session=session)


def trade_table(out) -> pd.DataFrame:
    """One row per ORB trade: side, size (share of the account) and gross result in bp of the position."""
    rows = [{"size": t.size, "gross_bp": t.side * (t.exit_price / t.entry_price - 1) * 1e4}
            for t in (out.trades or []) if t.exit_price is not None]
    return pd.DataFrame(rows, columns=["size", "gross_bp"])


def edge_per_trade(bars: pd.DataFrame, session: str) -> dict:
    t = trade_table(orb(session).run(bars))
    return {"trades": len(t), "edge bp": float(t["gross_bp"].mean()) if len(t) else np.nan}


def ftmo_costs(bars: pd.DataFrame, session: str) -> dict:
    """Real FTMO cost per round trip (bp of the position) at each slippage, and the year's Sharpe."""
    from .backtest import simulate
    from .costs import CostModel
    out = orb(session).run(bars)
    t = trade_table(out)
    res = {"FTMO trades": len(t), "FTMO edge bp": float(t["gross_bp"].mean()) if len(t) else np.nan}
    sp = bars["spread"].astype(float) / 2 / bars["close"].astype(float) * 1e4 if "spread" in bars.columns else None
    res["half spread bp"] = float(sp[sp > 0].median()) if sp is not None and (sp > 0).any() else np.nan
    for slip in SLIPS:
        bt = simulate(bars, out, costs=CostModel(slippage_range=slip, impact=0))
        size = t["size"].sum()
        res[f"cost bp {slip:.0%}"] = float(bt.costs.sum() / size * 1e4) if size else np.nan
        r = bt.returns.groupby(pd.DatetimeIndex(bt.returns.index).date).apply(lambda x: (1 + x).prod() - 1)
        res[f"sharpe {slip:.0%}"] = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
    return res


def check(symbols=None, mt5_loader=None, etf_loader=None, progress=print) -> pd.DataFrame:
    def quiet(fn, *a):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*a)
    mt5_loader = mt5_loader or (lambda s: quiet(dl.load_bars, "MT5:" + s, "5m"))
    etf_loader = etf_loader or (lambda s: quiet(dl.load_bars, s, "5m", "all"))
    rows = []
    for sym in symbols or PAIRS:
        etf, session = PAIRS.get(sym, (None, "US"))
        row = {"market": sym, "ETF": etf or ""}
        try:
            row.update(ftmo_costs(mt5_loader(sym), session))
        except Exception as e:
            progress(f"  skip {sym}: {str(e).splitlines()[0][:90]}")
            continue
        if etf:
            try:
                e = edge_per_trade(etf_loader(etf), session)
                row.update({"ETF trades": e["trades"], "ETF edge bp": e["edge bp"]})
            except Exception as e:
                progress(f"  {etf}: {str(e).splitlines()[0][:90]}")
        progress(f"  {sym:12s} done")
        rows.append(row)
    return pd.DataFrame(rows).set_index("market") if rows else pd.DataFrame()


def show(t: pd.DataFrame) -> str:
    cols = ["ETF", "ETF trades", "ETF edge bp", "FTMO trades", "FTMO edge bp", "half spread bp"] + \
           [f"cost bp {s:.0%}" for s in SLIPS] + [f"sharpe {s:.0%}" for s in SLIPS]
    t = t.reindex(columns=cols)
    out = t.copy().astype(object)
    for c in cols[1:]:
        out[c] = ["" if v != v else (f"{int(v)}" if "trades" in c else f"{v:.2f}") for v in t[c]]
    if "ETF edge bp" in t and t["ETF edge bp"].notna().any():
        out["edge / cost (2%)"] = [("" if (e != e or c != c or c <= 0) else f"{e / c:.1f}x")
                                   for e, c in zip(t["ETF edge bp"], t["cost bp 2%"])]
    return ("Per ORB trade, in bp of the position. ETF = long-run edge (2017-today, before costs); "
            "FTMO = 2026 on FTMO's own candles; cost = real spread + slippage 0/2/5% of a 5m range.\n"
            + out.to_string())


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return
    print("Running ORB on FTMO's candles (MT5) and on the stand-in ETFs (Alpaca)...")
    t = check(args or None)
    if t.empty:
        print("  Nothing loaded. Is MT5 open and logged into FTMO?")
        return
    print()
    print(show(t))
    path = dl.DATA_DIR / "edges" / "ftmo_orb.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(path)
    print(f"\n  Saved in {path}")


if __name__ == "__main__":
    main()
