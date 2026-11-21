"""Lesson 2: real market data, and why you backtest on ADJUSTED prices.

Run:  python lessons/02_real_data.py
First run downloads SPY and AAPL into data/; later runs use the saved files.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from quantlab import download as dl
from quantlab.backtest import run_backtest
from quantlab.strategies import buy_and_hold, mean_reversion, sma_crossover


def table(results: dict):
    df = pd.DataFrame({k: v.summary() for k, v in results.items()}).T
    fmt = {"CAGR": "{:.1%}", "Volatility": "{:.1%}", "Sharpe": "{:.2f}",
           "MaxDrawdown": "{:.1%}", "TimeInMarket": "{:.0%}", "trades": "{:.0f}"}
    for col, f in fmt.items():
        df[col] = df[col].map(f.format)
    print(df.to_string(), "\n")


dl.download_many(["SPY", "AAPL"])

print("=== Part A: a split looks like a crash in truly raw prices ===")
yahoo_close = dl.load("AAPL", adjusted=False)
# Yahoo's "Close" is already split-adjusted (but NOT dividend-adjusted).
# Rebuild what an unadjusted feed shows: before AAPL's 4-for-1 split on
# 2020-08-31, the shares really traded at 4x the price Yahoo now reports.
split_day = pd.Timestamp("2020-08-31")
truly_raw = yahoo_close.where(yahoo_close.index >= split_day, yahoo_close * 4)
print(f"Real price on 2020-08-28: ${truly_raw['2020-08-28']:.2f} "
      f"(Yahoo shows ${yahoo_close['2020-08-28']:.2f})")
print(f"Unadjusted feed, split day:     {truly_raw.pct_change()[split_day]:+.1%}  <- fake crash")
print(f"Split-adjusted close, same day: {yahoo_close.pct_change()[split_day]:+.1%}  <- what happened")
print("Lesson: always check what a data column means. Yahoo's Close fixes splits;\n"
      "only Adj Close also adds back dividends.\n")

print("=== Part B: raw prices silently delete dividends (SPY, buy & hold) ===")
spy_raw = dl.load("SPY", adjusted=False)
spy = dl.load("SPY")
table({"Raw close": run_backtest(spy_raw, buy_and_hold(spy_raw), cost_bps=0),
       "Adjusted close": run_backtest(spy, buy_and_hold(spy), cost_bps=0)})
print("The gap is the dividends. Over 30 years they are a big part of the return.\n")

print(f"=== Part C: Lesson 1's strategies on real SPY, "
      f"{spy.index[0].year}-{spy.index[-1].year} (5 bp costs) ===")
table({"Buy & Hold": run_backtest(spy, buy_and_hold(spy)),
       "SMA 50/200": run_backtest(spy, sma_crossover(spy)),
       "MeanRev 20": run_backtest(spy, mean_reversion(spy))})
print("Compare with Lesson 1: real markets have drift (they tend to go up),\n"
      "so time out of the market costs return. Look at drawdown too, not just CAGR.")
