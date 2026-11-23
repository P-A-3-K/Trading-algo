"""Common indicators, for strategies to import:  from quantlab.indicators import rsi, sma, ema

All of them use only the current and earlier values (no look-ahead)."""
import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple moving average of the last n values."""
    return series.rolling(int(n)).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    """Exponential moving average: recent values count more."""
    return series.ewm(span=int(n), adjust=False).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI, 0-100: average gain vs average loss over `period` bars."""
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)
