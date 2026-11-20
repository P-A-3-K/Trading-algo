"""Strategies turn prices into a *position*: the fraction of capital held.

 1.0 = fully long, 0.0 = flat (cash), -1.0 = fully short.
A strategy decides its position using data up to and including day t.
The backtester is responsible for making sure that position only earns
returns from day t+1 onward (no peeking into the future).
"""
import pandas as pd


def buy_and_hold(prices: pd.Series) -> pd.Series:
    """The benchmark every strategy must beat."""
    return pd.Series(1.0, index=prices.index)


def sma_crossover(prices: pd.Series, fast=50, slow=200) -> pd.Series:
    """Trend following: long when the fast average is above the slow one."""
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    return (fast_ma > slow_ma).astype(float)


def mean_reversion(prices: pd.Series, lookback=20, entry_z=1.0) -> pd.Series:
    """Buy when price is unusually far below its recent average."""
    mean = prices.rolling(lookback).mean()
    std = prices.rolling(lookback).std()
    z = (prices - mean) / std
    return (z < -entry_z).astype(float)
