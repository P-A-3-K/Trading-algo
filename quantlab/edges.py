"""Edge finder: textbook strategies on every FTMO market, judged on two periods.

    python -m quantlab.edges                     # FTMO's FX, metals, energy, indices, crypto, softs
    python -m quantlab.edges EURUSD XAUUSD ...   # only these MT5 symbols

Fixed, well-known settings (no tuning, so nothing is fitted to the past); daily candles from
MT5 with the broker's real spreads and overnight swaps. Every strategy is judged on:
  A = 2019-2024 (the long run) and B = 2025 to today (does it still work?).
The headline is each strategy as a PORTFOLIO of all markets (equal risk per market, weights
from period A only, scaled to 10% yearly volatility in A), because trend following earns
its keep across many unrelated markets, not on any single one.
Results are also saved in data/edges/edges.csv.
"""
import sys
import warnings

import numpy as np
import pandas as pd

from . import download as dl
from . import strategies

A_START, B_START = pd.Timestamp("2019-01-01"), pd.Timestamp("2025-01-01")
TARGET_VOL = 0.10
MIN_BARS = 500

STRATEGIES = {                                          # name -> (strategy class, settings)
    "Momentum 12m": ("TimeSeriesMomentum", {"lookback_months": 12, "allow_short": True}),
    "Momentum 3m weekly": ("TimeSeriesMomentum", {"lookback_months": 3, "rebalance": "weekly", "allow_short": True}),
    "Donchian 55/20": ("Breakout", {"entry": 55, "exit": 20, "allow_short": True}),
    "Donchian 20/10": ("Breakout", {"entry": 20, "exit": 10, "allow_short": True}),
}

# FTMO symbol names (missing ones are skipped; see: python -m quantlab.mt5 list)
GROUPS = {
    "FX": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY",
           "EURGBP", "AUDJPY", "EURCHF", "EURAUD", "GBPAUD", "AUDNZD", "CADJPY", "USDMXN", "USDZAR"],
    "Metals": ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD", "XCUUSD"],
    "Energy": ["USOIL.cash", "UKOIL.cash", "NATGAS.cash", "HEATOIL.c"],
    "Indices": ["US500.cash", "US100.cash", "US30.cash", "GER40.cash", "UK100.cash", "JP225.cash",
                "EU50.cash", "FRA40.cash", "HK50.cash", "AUS200.cash"],
    "Crypto": ["BTCUSD", "ETHUSD"],
    "Softs/grains": ["COCOA.c", "COFFEE.c", "CORN.c", "COTTON.c", "SOYBEAN.c", "WHEAT.c", "SUGAR.c"],
}


def group_of(symbol: str) -> str:
    for g, names in GROUPS.items():
        if symbol.lower() in (n.lower() for n in names):
            return g
    return "Other"


def weekdays_only(r: pd.Series) -> pd.Series:
    """Weekend returns (crypto) folded into Monday, so every market has the same calendar."""
    idx = pd.DatetimeIndex(r.index)
    day = idx.where(idx.day_of_week < 5, idx + pd.offsets.BDay(1)).normalize()
    return (1 + r).groupby(day).prod() - 1


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return {"sharpe": np.nan, "cagr": np.nan, "dd": np.nan}
    eq = (1 + r).cumprod()
    years = len(r) / 252
    return {"sharpe": float(r.mean() / r.std() * np.sqrt(252)),
            "cagr": float(eq.iloc[-1] ** (1 / years) - 1),
            "dd": float((eq / eq.cummax() - 1).min())}


def load_mt5(symbol: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dl.load_bars("MT5:" + symbol, "1d")


def cost_models() -> dict:
    """The same backtest three ways, to see what eats the returns."""
    from .costs import CostModel, flat
    return {"full": CostModel(),                                         # spread, slippage, swaps
            "no swaps": CostModel(swap_long_bps=0.0, swap_short_bps=0.0),
            "no costs": flat(0)}


def market_returns(bars: pd.DataFrame, costs=None) -> dict:
    """Daily returns (after the given costs; default: all) of every strategy on one market."""
    from .backtest import simulate
    from .costs import CostModel
    prepared = (costs or CostModel()).prepare(bars)
    out = {}
    for name, (cls, kw) in STRATEGIES.items():
        strat = strategies.get(cls)(**kw)
        bt = simulate(bars, strat.run(bars), costs=prepared)
        r = bt.returns.copy()
        r.index = pd.DatetimeIndex(r.index).tz_localize(None) if getattr(r.index, "tz", None) else pd.DatetimeIndex(r.index)
        out[name] = weekdays_only(r)
    return out


def cost_facts(bars: pd.DataFrame) -> dict:
    """The broker's swaps (bp per night) and median half-spread (bp) for this market."""
    half = np.nan
    if "spread" in bars.columns:
        sp = bars["spread"].astype(float) / 2 / bars["close"].astype(float) * 10_000
        half = float(sp[sp > 0].median()) if (sp > 0).any() else np.nan
    return {"swap long bp": bars.attrs.get("swap_long_bps"), "swap short bp": bars.attrs.get("swap_short_bps"),
            "half spread bp": half}


def data_problems(bars: pd.DataFrame) -> str:
    jumps = (bars["close"].pct_change().abs() > 0.25).sum()
    span = (bars.index[-1] - bars.index[0]).days * 252 / 365
    notes = []
    if len(bars) < 0.8 * span:
        notes.append(f"gaps ({len(bars)} of ~{int(span)} days)")
    if jumps:
        notes.append(f"{jumps} jumps >25% (contract rolls?)")
    return ", ".join(notes)


def run(symbols=None, loader=load_mt5, progress=print):
    """-> (per-market table, portfolio table, group table)."""
    symbols = symbols or [s for names in GROUPS.values() for s in names]
    rets, rows = {}, []
    for sym in symbols:
        try:
            bars = loader(sym)
        except Exception as e:
            progress(f"  skip {sym}: {str(e).splitlines()[0][:90]}")
            continue
        idx = pd.DatetimeIndex(bars.index)
        idx = idx.tz_localize(None) if idx.tz is not None else idx
        bars = bars[idx >= A_START - pd.Timedelta(days=400)]          # a year of warm-up before A
        if len(bars) < MIN_BARS:
            progress(f"  skip {sym}: only {len(bars)} daily candles")
            continue
        problems = data_problems(bars)
        progress(f"  {sym:12s} {len(bars):5d} days" + (f"   WARNING: {problems}" if problems else ""))
        variants = {k: market_returns(bars, m) for k, m in cost_models().items()}
        rets[sym] = variants
        facts = cost_facts(bars)
        for name, r in variants["full"].items():
            a, b = stats(r[(r.index >= A_START) & (r.index < B_START)]), stats(r[r.index >= B_START])
            g = variants["no costs"][name]
            ga, gb = stats(g[(g.index >= A_START) & (g.index < B_START)]), stats(g[g.index >= B_START])
            rows.append({"market": sym, "group": group_of(sym), "strategy": name,
                         "sharpe A": a["sharpe"], "sharpe B": b["sharpe"],
                         "return/yr A": a["cagr"], "return/yr B": b["cagr"],
                         "no-cost sharpe A": ga["sharpe"], "no-cost sharpe B": gb["sharpe"],
                         **facts, "note": problems})
    markets = pd.DataFrame(rows)
    if markets.empty:
        return markets, pd.DataFrame(), pd.DataFrame()
    port_rows, group_rows, check_rows = [], [], []
    for name in STRATEGIES:
        legs = {s: rets[s]["full"][name] for s in rets}
        port_rows.append({"strategy": name, **portfolio(legs)})
        check = {"strategy": name}
        for k in cost_models():
            p = portfolio({s: rets[s][k][name] for s in rets})
            check[f"A {k}"], check[f"B {k}"] = p["sharpe A"], p["sharpe B"]
        check_rows.append(check)
        for g in sorted({group_of(s) for s in legs}):
            sub = {s: r for s, r in legs.items() if group_of(s) == g}
            p = portfolio(sub)
            group_rows.append({"strategy": name, "group": g, "markets": len(sub),
                               "sharpe A": p["sharpe A"], "sharpe B": p["sharpe B"]})
    port = pd.DataFrame(port_rows).set_index("strategy")
    port.attrs["cost_check"] = pd.DataFrame(check_rows).set_index("strategy")
    return markets, port, pd.DataFrame(group_rows)


def portfolio(legs: dict) -> dict:
    """Equal-risk mix (weights from period A), scaled to TARGET_VOL in A; stats for A and B."""
    df = pd.DataFrame(legs).sort_index()
    df = df[df.index >= A_START].fillna(0.0)
    a_part = df[df.index < B_START]
    vol = a_part.std().replace(0, np.nan)
    if a_part.empty or vol.isna().all():
        return {k: np.nan for k in ("sharpe A", "sharpe B", "return/yr A", "return/yr B", "worst DD A",
                                    "worst DD B", "B markets > 0")} | {"markets": len(legs)}
    w = (1 / vol).fillna(0)
    w = w / w.sum()
    mix = (df * w).sum(axis=1)
    mix = mix * TARGET_VOL / (mix[mix.index < B_START].std() * np.sqrt(252))
    a, b = stats(mix[mix.index < B_START]), stats(mix[mix.index >= B_START])
    positive_b = sum(1 for s in df.columns if stats(df.loc[df.index >= B_START, s])["sharpe"] > 0)
    return {"markets": len(legs), "sharpe A": a["sharpe"], "sharpe B": b["sharpe"],
            "return/yr A": a["cagr"], "return/yr B": b["cagr"], "worst DD A": a["dd"], "worst DD B": b["dd"],
            "B markets > 0": f"{positive_b}/{len(legs)}"}


def show(markets, port, groups) -> str:
    pct = lambda v: "" if v != v else f"{v * 100:.1f}%"
    num = lambda v: "" if v != v else f"{v:.2f}"
    out = []
    if not port.empty:
        t = port.copy().astype(object)
        for c in ("sharpe A", "sharpe B"):
            t[c] = [num(v) for v in port[c]]
        for c in ("return/yr A", "return/yr B", "worst DD A", "worst DD B"):
            t[c] = [pct(v) for v in port[c]]
        out += [f"PORTFOLIO of all markets (equal risk, {TARGET_VOL:.0%} volatility in A). "
                f"A = {A_START.year}-{B_START.year - 1}, B = {B_START.year}-today", t.to_string(), ""]
        chk = port.attrs.get("cost_check")
        if chk is not None:
            out += ["What the costs take (portfolio Sharpe: no costs -> spread+slippage only -> also swaps):",
                    chk[[f"{p} {k}" for p in ("A", "B") for k in ("no costs", "no swaps", "full")]].map(num).to_string(), ""]
    if not groups.empty:
        g = groups.pivot(index="group", columns="strategy", values=["sharpe A", "sharpe B"])
        g = g.swaplevel(axis=1).sort_index(axis=1, level=0)
        out += ["Sharpe by market group (A | B):", g.map(num).to_string(), ""]
    if not markets.empty:
        m = markets.pivot(index="market", columns="strategy", values=["sharpe A", "sharpe B"])
        m = m.swaplevel(axis=1).sort_index(axis=1, level=0)
        out += ["Sharpe per market (A | B):", m.map(num).to_string()]
    return "\n".join(out)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return
    print("Loading daily candles from MT5 (first run downloads each market)...")
    markets, port, groups = run(args or None)
    if markets.empty:
        print("  No markets loaded. Is MT5 open and logged in?")
        return
    print()
    print(show(markets, port, groups))
    path = dl.DATA_DIR / "edges" / "edges.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    markets.to_csv(path, index=False)
    print(f"\n  Saved per-market results in {path}")


if __name__ == "__main__":
    main()
