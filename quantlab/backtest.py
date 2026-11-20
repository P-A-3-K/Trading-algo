"""The backtest engine: simulate what a strategy would have earned."""
from dataclasses import dataclass

import pandas as pd

from . import metrics


@dataclass
class BacktestResult:
    equity: pd.Series       # portfolio value over time (starts at 1.0)
    returns: pd.Series      # daily strategy returns, after costs
    positions: pd.Series    # position actually held each day
    trades: int             # number of position changes

    def summary(self) -> dict:
        s = metrics.summarize(self.returns)
        s["trades"] = self.trades
        return s


def run_backtest(prices: pd.Series, positions: pd.Series,
                 cost_bps: float = 5.0) -> BacktestResult:
    """Vectorized daily backtest.

    cost_bps: cost per unit of position change, in basis points
    (1 bp = 0.01%). Covers commission + bid/ask spread + slippage.
    """
    asset_returns = prices.pct_change().fillna(0.0)

    # THE most important line in any backtester: the signal computed at
    # the close of day t can only be traded for day t+1's return.
    # Remove .shift(1) and you get "look-ahead bias" -- fantasy profits.
    held = positions.shift(1).fillna(0.0)

    turnover = held.diff().abs().fillna(held.abs())
    costs = turnover * cost_bps / 10_000

    strat_returns = held * asset_returns - costs
    equity = (1 + strat_returns).cumprod()
    trades = int((turnover > 0).sum())
    return BacktestResult(equity, strat_returns, held, trades)
