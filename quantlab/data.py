"""Market data: load real prices from CSV, or generate synthetic ones.

Synthetic data matters for learning: on random-walk prices *no* strategy
has a real edge, so any profit a backtest shows there is luck or a bug.
"""
import numpy as np
import pandas as pd


def synthetic_prices(n_days=2520, start_price=100.0, annual_drift=0.07,
                     annual_vol=0.20, seed=42):
    """Geometric Brownian motion: the textbook model of a stock price.

    2520 trading days ~= 10 years (252 trading days per year).
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    # Log-returns are normally distributed under GBM.
    log_rets = rng.normal((annual_drift - 0.5 * annual_vol**2) * dt,
                          annual_vol * np.sqrt(dt), n_days)
    prices = start_price * np.exp(np.cumsum(log_rets))
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.Series(prices, index=dates, name="close")


def load_csv(path, date_col="Date", price_col="Adj Close"):
    """Load a CSV of daily prices, e.g. one saved by quantlab.download.

    Defaults to "Adj Close" (dividends and splits folded in), which is
    what returns should be computed from.
    """
    df = pd.read_csv(path, parse_dates=[date_col], index_col=date_col)
    return df[price_col].sort_index().rename("close")
