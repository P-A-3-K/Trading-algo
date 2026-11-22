"""Performance metrics. Returns alone are meaningless without risk."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate."""
    total = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    return total ** (1 / years) - 1


def annual_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe(returns: pd.Series, risk_free=0.0) -> float:
    """Return per unit of risk. >1 is good, >2 on real data is suspicious."""
    excess = returns - risk_free / TRADING_DAYS
    if excess.std() == 0:
        return 0.0
    return excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough loss. This is what makes people quit a strategy."""
    equity = (1 + returns).cumprod()
    return (equity / equity.cummax() - 1).min()


def summarize(returns: pd.Series) -> dict:
    return {
        "CAGR": cagr(returns),
        "Volatility": annual_volatility(returns),
        "Sharpe": sharpe(returns),
        "MaxDrawdown": max_drawdown(returns),
        "TimeInMarket": float((returns != 0).mean()),
    }


def sortino(returns: pd.Series) -> float:
    """Like Sharpe, but only counts DOWNSIDE volatility as risk.

    Big up-days don't hurt you, so they shouldn't count against a strategy.
    """
    downside = returns[returns < 0]
    dd = np.sqrt((downside ** 2).sum() / len(returns)) if len(returns) else 0.0
    if dd == 0:
        return 0.0
    return returns.mean() / dd * np.sqrt(TRADING_DAYS)


def calmar(returns: pd.Series) -> float:
    """Yearly return divided by the worst drawdown: reward per unit of pain."""
    mdd = abs(max_drawdown(returns))
    return cagr(returns) / mdd if mdd else 0.0


def longest_drawdown(returns: pd.Series) -> int:
    """Most trading days spent below a previous peak (peak to full recovery,
    or to the end of the data if it never recovered)."""
    equity = (1 + returns).cumprod()
    under = equity < equity.cummax()
    longest = run = 0
    for flag in under.values:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return int(longest)


def round_trips(returns: pd.Series, held: pd.Series) -> pd.DataFrame:
    """Split the backtest into trades: each stretch of days with a position.

    `held` is the position actually held each day (BacktestResult.positions).
    A trip's return compounds the strategy's daily returns over the stretch,
    costs included, plus the cost of the exit day. Returns one row per trip.
    """
    held = held.fillna(0.0)
    in_trade = held != 0
    trip_id = (in_trade != in_trade.shift(fill_value=False)).cumsum()[in_trade]
    rows = []
    for _, days in trip_id.groupby(trip_id):
        start, end = days.index[0], days.index[-1]
        span = returns.loc[start:end]
        exit_pos = returns.index.get_loc(end) + 1        # exit cost lands the next day
        if exit_pos < len(returns) and held.iloc[exit_pos] == 0:
            span = pd.concat([span, returns.iloc[exit_pos:exit_pos + 1]])
        rows.append({"entry": start, "exit": end, "days": len(days),
                     "return": float((1 + span).prod() - 1)})
    return pd.DataFrame(rows, columns=["entry", "exit", "days", "return"])


def full_stats(returns: pd.Series, held: pd.Series = None, fills: int = None) -> dict:
    """Everything the stats table shows, for one return series."""
    eq = (1 + returns).cumprod()
    stats = {
        "total_return": float(eq.iloc[-1] - 1),
        "cagr": cagr(returns),
        "volatility": annual_volatility(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar(returns),
        "longest_drawdown_days": longest_drawdown(returns),
    }
    if held is not None:
        trips = round_trips(returns, held)
        wins, losses = trips[trips["return"] > 0], trips[trips["return"] <= 0]
        stats.update({
            "time_in_market": float((held.fillna(0) != 0).mean()),
            "fills": fills,
            "round_trips": len(trips),
            "win_rate": len(wins) / len(trips) if len(trips) else None,
            "avg_win": float(wins["return"].mean()) if len(wins) else None,
            "avg_loss": float(losses["return"].mean()) if len(losses) else None,
            "avg_days_held": float(trips["days"].mean()) if len(trips) else None,
        })
    return stats
