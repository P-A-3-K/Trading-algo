"""Market scan: the same parameter sweep on many tickers x timeframes.

    from quantlab.scan import scan
    res = scan(SmaCrossover, {"fast": (5, 100, 5), "slow": (20, 300, 20)},
               markets=[("SPY", "1d"), ("QQQ", "1d"), ("SPY", "1h")])

For every market (one ticker on one timeframe) it runs optimize.sweep, which
splits that market's history into in-sample (IS) and out-of-sample (OOS).

The honest question is NOT "which market had the best result?", because with
many markets and many settings the top result is mostly luck. Instead:
  * each market's settings are picked on its IS part only, and judged on OOS;
  * markets are compared on those OOS numbers;
  * settings are also compared ACROSS markets: a setting that is good on most
    markets is far more believable than one that shines on a single market.
"""
import time

import numpy as np
import pandas as pd

from . import optimize

MAX_MARKETS = 60              # tickers x timeframes
MAX_BACKTESTS = 100_000       # markets x combinations (~3 ms each)
MIN_BARS = 200                # less history than this: skip the market
THIN_TEST_DAYS = 60           # fewer OOS trading days: results flagged as unreliable

# Rank-by choices: which column picks a market's settings, and which way is better
BETTER = {"sharpe": "higher", "cagr": "higher", "final": "higher", "max_dd": "higher", "trades": "lower"}


class Cancelled(Exception):
    """Raised from the progress callback to stop a scan."""


def _num(v):
    """JSON-safe float: NaN/inf -> None."""
    if v is None:
        return None
    v = float(v)
    return None if np.isnan(v) or np.isinf(v) else v


def _when(ts) -> str:
    from .chart import when
    return when(ts)


def _pick(table: pd.DataFrame, rank: str) -> pd.Series:
    """The row with the best IN-SAMPLE score (ties: better IS Sharpe)."""
    col = f"is_{rank}"
    t = table.dropna(subset=[col])
    if t.empty:
        return None
    asc = BETTER.get(rank, "higher") == "lower"
    return t.sort_values([col, "is_sharpe"], ascending=[asc, False]).iloc[0]


def summarize_market(res, rank: str, params: list) -> dict:
    """One market's sweep reduced to what the scan compares."""
    t = res.table
    best = _pick(t, rank)
    bench = res.benchmark
    oos_sharpe = t["oos_sharpe"].dropna()
    bh_oos = bench.get("oos_sharpe", np.nan)
    rep = optimize.overfitting_report(res, "sharpe")
    return {
        "settings": {k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in params} if best is not None else None,
        **{f"best_{k}": _num(best[k]) if best is not None else None
           for k in ("is_sharpe", "is_cagr", "is_final", "is_max_dd", "is_trades",
                     "oos_sharpe", "oos_cagr", "oos_final", "oos_max_dd", "oos_trades", "oos_win_rate")},
        "bh_is_sharpe": _num(bench.get("is_sharpe")), "bh_oos_sharpe": _num(bh_oos),
        "bh_is_cagr": _num(bench.get("is_cagr")), "bh_oos_cagr": _num(bench.get("oos_cagr")),
        "median_oos_sharpe": _num(oos_sharpe.median()) if len(oos_sharpe) else None,
        # share of ALL settings that beat buy & hold out-of-sample (Sharpe): is the strategy
        # good here in general, or did only a few lucky settings work?
        "share_beat_bh": _num((oos_sharpe > bh_oos).mean()) if len(oos_sharpe) and not np.isnan(bh_oos) else None,
        "rank_corr": _num(rep.get("rank_corr")) if rep else None,
        "combos": int(len(t)),
    }


def scan(cls, ranges: dict, markets: list, split: float = 0.7, rank: str = "sharpe",
         costs=None, load=None, progress=None) -> dict:
    """Sweep `ranges` on every (ticker, timeframe) in `markets`.

    load: function(ticker, timeframe) -> bars (default: quantlab.download.load_bars,
          which downloads anything missing).
    progress: function(done, total, label); may raise Cancelled to stop.
    Returns plain data (JSON-ready): one summary per market, plus every setting's
    results across all markets.
    """
    if load is None:
        from .download import load_bars as load
    from .metrics import bars_per_day
    if rank not in BETTER:
        raise ValueError(f"unknown rank {rank!r}")
    markets = list(dict.fromkeys((t, tf) for t, tf in markets))   # drop repeats, keep order
    if not markets:
        raise ValueError("Pick at least one ticker and one timeframe.")
    if len(markets) > MAX_MARKETS:
        raise ValueError(f"{len(markets)} markets is too many (max {MAX_MARKETS}). "
                         "Use fewer tickers or timeframes.")
    combos = optimize.grid(cls, ranges)
    total = len(markets) * len(combos)
    if total > MAX_BACKTESTS:
        raise ValueError(f"{len(markets)} markets x {len(combos):,} combinations = {total:,} backtests "
                         f"(max {MAX_BACKTESTS:,}). Use bigger steps, or fewer tickers or timeframes.")
    params = [k for k, (a, b, _) in ranges.items() if b > a] or list(ranges)[:1]
    t0 = time.time()
    out, per_setting = [], {}
    for m, (ticker, tf) in enumerate(markets):
        label = f"{ticker} {tf}"
        base = m * len(combos)
        if progress:
            progress(base, total, f"Loading {label}")
        row = {"ticker": ticker, "tf": tf, "error": None}
        try:
            bars = load(ticker, tf)
            if len(bars) < MIN_BARS:
                raise ValueError(f"only {len(bars)} bars of history (need {MIN_BARS})")
            step = (lambda i, n, _b=base, _l=label: progress(_b + i, total, f"Testing {_l}")) if progress else None
            res = optimize.sweep(bars, cls, ranges, split, costs=costs, progress=step)
            if res.table.empty:
                raise ValueError("no valid combinations" + (f": {res.skipped[0][1]}" if res.skipped else ""))
        except Cancelled:
            raise
        except Exception as e:                 # bad ticker, no data, download failed...
            row["error"] = str(e) or type(e).__name__
            out.append(row)
            continue
        bpd = bars_per_day(bars.index)
        cut = bars.index.get_loc(res.split_date)
        oos_days = (len(bars) - cut) / bpd
        row.update(summarize_market(res, rank, params))
        row.update({"bars": len(bars), "start": _when(res.start), "end": _when(res.end),
                    "split": _when(res.split_date), "oos_days": round(oos_days),
                    "thin": oos_days < THIN_TEST_DAYS})
        out.append(row)
        # Every setting's IS / OOS Sharpe on this market, for the across-markets table
        bh = res.benchmark.get("oos_sharpe", np.nan)
        for r in res.table.itertuples(index=False):
            d = r._asdict()
            key = tuple(d[k] for k in params)
            s = per_setting.setdefault(key, {"is": [], "oos": [], "beat": 0, "n": 0})
            s["is"].append(d["is_sharpe"]); s["oos"].append(d["oos_sharpe"])
            s["n"] += 1
            s["beat"] += int(not np.isnan(bh) and d["oos_sharpe"] > bh)
    if progress:
        progress(total, total, "Done")

    ok = [r for r in out if not r["error"]]
    settings = []
    for key, s in per_setting.items():
        if s["n"] < max(1, len(ok)):          # only settings valid on every market
            continue
        settings.append({**{k: (v.item() if hasattr(v, "item") else v) for k, v in zip(params, key)},
                         "median_is_sharpe": _num(np.nanmedian(s["is"])),
                         "median_oos_sharpe": _num(np.nanmedian(s["oos"])),
                         "worst_oos_sharpe": _num(np.nanmin(s["oos"])),
                         "beat_bh": s["beat"], "markets": s["n"]})
    # Chosen like everything else: by IN-SAMPLE results, across all markets
    settings.sort(key=lambda r: -(r["median_is_sharpe"] if r["median_is_sharpe"] is not None else -1e9))
    return {"markets": out, "settings": settings[:25], "settings_count": len(settings),
            "params": params, "combos": len(combos), "rank": rank, "split": split,
            "backtests": len(ok) * len(combos), "seconds": time.time() - t0}
