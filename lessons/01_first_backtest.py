"""Lesson 1: your first backtest, and why you should distrust it.

Run:  python -m lessons.01_first_backtest   (from the repo root)
  or: python lessons/01_first_backtest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from quantlab.backtest import run_backtest
from quantlab.data import synthetic_prices
from quantlab.strategies import buy_and_hold, mean_reversion, sma_crossover


def show(results: dict):
    df = pd.DataFrame({k: v.summary() for k, v in results.items()}).T
    fmt = {"CAGR": "{:.1%}", "Volatility": "{:.1%}", "Sharpe": "{:.2f}",
           "MaxDrawdown": "{:.1%}", "TimeInMarket": "{:.0%}", "trades": "{:.0f}"}
    for col, f in fmt.items():
        df[col] = df[col].map(f.format)
    print(df.to_string(), "\n")


prices = synthetic_prices(seed=42)
strategies = {
    "Buy & Hold": buy_and_hold(prices),
    "SMA 50/200": sma_crossover(prices),
    "MeanRev 20": mean_reversion(prices),
}

print("=== Part A: honest backtest on random-walk prices (5 bp costs) ===")
show({name: run_backtest(prices, pos) for name, pos in strategies.items()})

print("=== Part B: look-ahead bias, the #1 backtesting bug ===")
print("Strategy: 'be long today if today's return is positive'.")
print("You only know today's return at the close, so you can't have held it.")
rets = prices.pct_change().fillna(0.0)
peek = (rets > 0).astype(float)
honest = run_backtest(prices, peek)
cheat = run_backtest(prices, peek.shift(-1).fillna(0.0))  # undoes the engine's shift(1)
show({"Honest (trade tomorrow)": honest, "Cheating (trade today)": cheat})
print("If a backtest ever looks this good, assume a bug before assuming genius.\n")

print("=== Part C: is SMA's result skill or luck? 20 different random markets ===")
sharpes = []
for seed in range(20):
    p = synthetic_prices(seed=seed)
    sharpes.append(run_backtest(p, sma_crossover(p)).summary()["Sharpe"])
s = pd.Series(sharpes)
print(f"SMA Sharpe across 20 random worlds: mean {s.mean():.2f}, "
      f"min {s.min():.2f}, max {s.max():.2f}")
print("On a random walk there is no edge to find, so the spread above is "
      "pure luck.\nAny single backtest of yours sits somewhere inside a "
      "spread like this.")
