"""Lesson 3: find gaps in price data, and decide whether to fill or drop.

Run:  python lessons/03_data_checks.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from quantlab import download as dl
from quantlab.checks import fill_gaps, find_missing, find_stale, gap_report

tickers = ["SPY", "QQQ", "AAPL"]
dl.download_many(tickers)
tables = {t: pd.read_csv(dl.csv_path(t), parse_dates=["Date"], index_col="Date")
          for t in tickers}

print("=== Part A: health check of our real data ===")
print(pd.DataFrame([gap_report(df, t) for t, df in tables.items()]).to_string(index=False))
print("\nYahoo's daily data for big US tickers is clean: no missing days.")
stale = find_stale(tables["AAPL"])
print(f"But AAPL has {len(stale)} stale days: {[d.date() for d in stale]}")
print(tables["AAPL"].loc["1981-08-06":"1981-08-12",
                          ["Open", "High", "Low", "Close", "Volume"]].to_string())
print("Zero volume and a frozen price: nobody traded, the vendor copied the\n"
      "previous price. Harmless here, but a strategy that 'traded' that day\n"
      "would be trading on a price that never existed.\n")

print("=== Part B: weekends and holidays are not gaps ===")
week = tables["SPY"].loc["2025-12-22":"2026-01-05"].index
print("SPY rows over Christmas/New Year 2025:", [d.strftime("%a %d") for d in week])
print("Dec 25 and Jan 1 are NYSE holidays, so they are correctly absent.\n"
      "A naive 'every weekday' check would wrongly call them gaps.\n")

print("=== Part C: a damaged copy of SPY (we delete some days on purpose) ===")
spy = dl.load("SPY")["2024-01-01":"2024-12-31"]
holes = pd.to_datetime(["2024-03-05", "2024-07-15", "2024-07-16",
                        "2024-10-01", "2024-10-02", "2024-10-03",
                        "2024-10-04", "2024-10-07"])
damaged = spy.drop(holes)
print("Gaps found:")
print(find_missing(damaged.index).to_string(index=False), "\n")

filled = fill_gaps(damaged, max_fill=3)
print("After fill_gaps(max_fill=3):")
print(f"  short gaps (1-2 days) forward-filled; {int(filled.isna().sum())} days of "
      f"the 5-day gap left empty on purpose.")
total = lambda s: s.dropna().iloc[-1] / s.dropna().iloc[0] - 1
print(f"  total return, true data:   {total(spy):.2%}")
print(f"  total return, filled data: {total(filled):.2%}  (same: filling moves "
      "returns between days, it doesn't invent them)")
bad = damaged.reindex(spy.index).bfill()
print(f"  back-filled instead: the price for 2024-03-05 becomes the 03-06 price "
      f"{bad['2024-03-05']:.2f} vs the real {spy['2024-03-05']:.2f}:\n"
      "  a future price leaking into the past. Never back-fill.\n")

print("=== The policy ===")
print("1. Run gap_report on every new download.")
print("2. Forward-fill gaps of up to 3 trading days.")
print("3. Longer gaps: investigate, then drop that period or that ticker.")
print("4. Stale days: keep the row, but don't trust a trade made on that day.")
