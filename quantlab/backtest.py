"""The backtest engine: simulate what a strategy would have earned."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics


@dataclass
class BacktestResult:
    equity: pd.Series       # portfolio value over time (starts at 1.0)
    returns: pd.Series      # daily strategy returns, after costs
    positions: pd.Series    # position actually held each day
    trades: int             # number of position changes
    plots: list = None      # lines the strategy asked to draw (if any)
    strategy: str = ""

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


def backtest(strategy, bars: pd.DataFrame, cost_bps: float = 5.0) -> BacktestResult:
    """Run a Strategy plug-in on daily bars and simulate it."""
    out = strategy.run(bars)
    result = simulate(bars, out, cost_bps)
    result.plots = out.plots
    result.strategy = repr(strategy)
    return result


def trades(positions: pd.Series, prices: pd.Series) -> pd.DataFrame:
    """Every change of position, as a list of fills.

    Fill timing matches run_backtest: the position decided with day t's close
    is entered AT day t's close (it starts earning from day t+1). So the fill
    date is the signal date and the fill price is that day's close.

    side: "buy" (more long), "sell" (less long), "short" (more short),
          "cover" (less short). A flip from long to short is two rows.
    """
    pos = positions.astype(float).fillna(0.0)
    prev = pos.shift(1).fillna(0.0)
    rows = []
    for date in pos.index[pos.ne(prev)]:
        a, b = prev[date], pos[date]
        legs = []
        if a > 0 and b < a:                       # reduce or close a long
            legs.append(("sell", a, max(b, 0.0)))
        if a < 0 and b > a:                       # reduce or close a short
            legs.append(("cover", a, min(b, 0.0)))
        if b > 0 and b > max(a, 0.0):             # open or add to a long
            legs.append(("buy", max(a, 0.0), b))
        if b < 0 and b < min(a, 0.0):             # open or add to a short
            legs.append(("short", min(a, 0.0), b))
        for side, frm, to in legs:
            rows.append({"date": date, "side": side, "from": frm, "to": to,
                         "price": float(prices[date])})
    return pd.DataFrame(rows, columns=["date", "side", "from", "to", "price"])


# ---- Trade-based strategies (exact entry/exit prices) -------------------------------------

def held_positions(index: pd.DatetimeIndex, trades: list) -> pd.Series:
    """Position held at each day's close: side * size while a trade is open.

    A trade entered on day d (e.g. at the open) is held at d's close; on its exit
    day it is no longer held at the close.
    """
    held = pd.Series(0.0, index=index)
    for t in trades:
        mask = index >= t.entry_date
        if t.exit_date is not None:
            mask &= index < t.exit_date
        held[mask] = t.side * t.size
    return held


def simulate_trades(bars: pd.DataFrame, trades: list, cost_bps: float = 5.0) -> BacktestResult:
    """Daily returns from trades filled at exact prices.

    Entry day earns close/entry, days in between close/previous close, the exit
    day exit/previous close (or exit/entry if both happen the same day). An open
    trade is valued at the last close. Costs are charged on entry and exit.
    """
    close = bars["close"].to_numpy(dtype=float)
    idx = bars.index
    r = np.zeros(len(idx))
    cost = cost_bps / 10_000
    for t in trades:
        i0 = idx.get_loc(t.entry_date)
        w = t.side * t.size
        r[i0] -= t.size * cost
        if t.exit_date is not None and idx.get_loc(t.exit_date) == i0:
            r[i0] += w * (t.exit_price / t.entry_price - 1) - t.size * cost
            continue
        r[i0] += w * (close[i0] / t.entry_price - 1)
        i1 = idx.get_loc(t.exit_date) if t.exit_date is not None else len(idx)
        if i1 > i0 + 1:
            r[i0 + 1:i1] += w * (close[i0 + 1:i1] / close[i0:i1 - 1] - 1)
        if t.exit_date is not None:
            r[i1] += w * (t.exit_price / close[i1 - 1] - 1) - t.size * cost
    returns = pd.Series(r, index=idx)
    fills = sum(1 if t.exit_date is None else 2 for t in trades)
    return BacktestResult((1 + returns).cumprod(), returns, held_positions(idx, trades), fills)


def trade_fills(trades: list) -> pd.DataFrame:
    """Trades as a list of fills (same format as trades()), with exact prices."""
    rows = []
    for t in trades:
        w = t.side * t.size
        rows.append({"date": t.entry_date, "side": "buy" if t.side > 0 else "short",
                     "from": 0.0, "to": w, "price": t.entry_price})
        if t.exit_date is not None:
            rows.append({"date": t.exit_date, "side": "sell" if t.side > 0 else "cover",
                         "from": w, "to": 0.0, "price": t.exit_price})
    return pd.DataFrame(rows, columns=["date", "side", "from", "to", "price"])


def simulate(bars: pd.DataFrame, out, cost_bps: float = 5.0) -> BacktestResult:
    """Backtest a StrategyResult: trade-based if it has trades, else position-based."""
    if out.trades is not None:
        return simulate_trades(bars, out.trades, cost_bps)
    return run_backtest(bars["close"], out.positions, cost_bps)


def fills_of(bars: pd.DataFrame, out) -> pd.DataFrame:
    """All fills of a StrategyResult, for charts and stats."""
    if out.trades is not None:
        return trade_fills(out.trades)
    return trades(out.positions, bars["close"])
