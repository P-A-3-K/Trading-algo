"""Check a strategy file from the terminal before (or instead of) opening the app.

    python -m quantlab.check strategies/my_idea.py            # tests on SPY
    python -m quantlab.check strategies/my_idea.py QQQ        # tests on another ticker

For every Strategy class in the file it: loads it, shows its settings, checks it
doesn't look into the future, runs a backtest with the default settings and
compares it with buy & hold. Exit code 1 if anything is wrong.
"""
import sys
from pathlib import Path

import pandas as pd


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    path, ticker = Path(argv[0]), (argv[1] if len(argv) > 1 else "SPY")
    if not path.is_file():
        print(f"No such file: {path}")
        return 2

    from . import discovery
    from .backtest import fills_of, run_backtest, simulate
    from .download import load_bars
    from .metrics import full_stats

    print(f"Checking {path} ...")
    try:
        found = discovery._load(path.resolve(), str(path))
    except RuntimeError as e:
        print(f"  FAIL  the file doesn't load: {e}")
        return 1
    if not found:
        print("  FAIL  no Strategy class found. Is it `class Something(Strategy):`?")
        return 1

    ok = True
    bars = None
    for f in found:
        cls = f.cls
        print(f"\n{cls.__name__}: \"{cls.name}\"")
        for k, p in cls.params.items():
            print(f"    setting {k:12s} default {p.default:<6} allowed {p.min}-{p.max}  ({p.label})")
        if f.problem:
            ok = False
            print(f"  FAIL  {f.problem}")
            continue
        print("  ok    loads, runs on test data, doesn't look into the future")
        if path.name.startswith("_"):
            print("  note  files starting with _ are hidden from the app: copy it to a new name")
        if bars is None:
            bars = load_bars(ticker)
        try:
            strategy = cls()
            out = strategy.run(bars)
        except Exception as e:
            ok = False
            print(f"  FAIL  crashed on {ticker}: {type(e).__name__}: {e}")
            continue
        bt = simulate(bars, out)
        bh = run_backtest(bars["close"], pd.Series(1.0, index=bars.index))
        n = len(fills_of(bars, out))
        s, b = full_stats(bt.returns, bt.positions, n), full_stats(bh.returns, bh.positions, 1)
        span = f"{bars.index[0].date()} to {bars.index[-1].date()}"
        print(f"\n  {repr(strategy)} on {ticker}, {span}:")
        rows = [("Return per year", "cagr", "{:.1%}"), ("Sharpe ratio", "sharpe", "{:.2f}"),
                ("Worst drawdown", "max_drawdown", "{:.1%}"), ("Time in market", "time_in_market", "{:.0%}")]
        print(f"    {'':18s}{'strategy':>10s}{'buy & hold':>12s}")
        for label, key, fmt in rows:
            print(f"    {label:18s}{fmt.format(s[key]):>10s}{fmt.format(b[key]):>12s}")
        wr = f"{s['win_rate']:.0%}" if s["win_rate"] is not None else "-"
        print(f"    {'Trades':18s}{n:>10d}{1:>12d}    win rate {wr}")
        if n == 0:
            print("  note  it never traded with the default settings: check your signal conditions")
        if out.plots:
            print(f"    lines drawn: {', '.join(p.name for p in out.plots)}")
        from .bootstrap import bootstrap
        bs, bb, beat = bootstrap(bt.returns.to_numpy(), bh.returns.to_numpy())
        S = bs.summary()
        print(f"  luck  in 5,000 reshuffled histories: Sharpe {S['p5']:.2f} to {S['p95']:.2f} "
              f"(90% range), above 0 in {S['prob_positive']:.0%}, beats buy & hold in {beat:.0%}")
    print("\nAll good: refresh the app to use it." if ok else "\nFix the problems above and run the check again.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
