"""Lesson 4: write your own strategy as a plug-in.

Run:  python lessons/04_strategy_class.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from quantlab.backtest import backtest
from quantlab.download import load_bars
from quantlab.strategies import BuyAndHold, SmaCrossover
from quantlab.strategy import Param, Strategy, check_no_lookahead


# ---- A brand-new strategy: this is all it takes -------------------------------
class Breakout(Strategy):
    name = "Donchian breakout"
    description = ("Buy when the price closes at a new N-day high; sell when it "
                   "closes at a new M-day low. A classic trend-following rule.")
    params = {"entry": Param(55, 5, 250, "Buy on a new high over (days)"),
              "exit": Param(20, 5, 250, "Sell on a new low over (days)")}

    def compute(self, bars):
        # shift(1): compare today's close with the PREVIOUS N days, not including today
        upper = bars.high.rolling(self.entry).max().shift(1)
        lower = bars.low.rolling(self.exit).min().shift(1)
        self.plot(f"{self.entry}-day high", upper)
        self.plot(f"{self.exit}-day low", lower)
        signal = pd.Series(float("nan"), index=bars.index)
        signal[bars.close > upper] = 1.0      # breakout: get in
        signal[bars.close < lower] = 0.0      # breakdown: get out
        return signal.ffill().fillna(0.0)     # hold the last decision


# ---- A broken one: it peeks at tomorrow's price -------------------------------
class Cheater(Strategy):
    name = "Cheater"
    description = "Buys if TOMORROW goes up. Impossible in real life."
    params = {}

    def compute(self, bars):
        return (bars.close.pct_change().shift(-1) > 0).astype(float)


bars = load_bars("SPY")

print("=== What the app will read from each strategy ===")
for cls in (SmaCrossover, Breakout):
    info = cls.info()
    print(f"- {info['name']}: {info['description']}")
    for k, p in info["params"].items():
        print(f"    {k:6s} default {p['default']:>4}  range {p['min']}-{p['max']}  ({p['label']})")

print("\n=== Bad settings get a clear message instead of a silent wrong result ===")
for bad in ({"entry": 2}, {"entyr": 30}, {"entry": "fifty"}):
    try:
        Breakout(**bad)
    except ValueError as e:
        print(f"  Breakout(**{bad}) -> {e}")

print("\n=== Backtests on SPY (5 bp costs) ===")
rows = {}
for s in (BuyAndHold(), SmaCrossover(), Breakout(), Breakout(entry=20, exit=10)):
    r = backtest(s, bars)
    rows[repr(s)] = {**r.summary(), "lines drawn": ", ".join(p.name for p in r.plots) or "-"}
df = pd.DataFrame(rows).T
for col, f in {"CAGR": "{:.1%}", "Sharpe": "{:.2f}", "MaxDrawdown": "{:.1%}", "trades": "{:.0f}"}.items():
    df[col] = df[col].map(f.format)
print(df[["CAGR", "Sharpe", "MaxDrawdown", "trades", "lines drawn"]].to_string(), "\n")

print("=== The look-ahead detector ===")
for s in (Breakout(), Cheater()):
    leaks = check_no_lookahead(s, bars)
    verdict = "honest" if not leaks else f"PEEKS AT THE FUTURE on {len(leaks)} of 8 test dates"
    print(f"  {s.name:18s} {verdict}")
r = backtest(Cheater(), bars).summary()
print(f"\n  The cheater 'earns' {r['CAGR']:.0%} a year with a Sharpe of {r['Sharpe']:.1f}.\n"
      "  check_no_lookahead() runs the strategy on data cut off at a date and checks\n"
      "  the position on that date doesn't change once the future is added. The app\n"
      "  will run this on every strategy you add.")
