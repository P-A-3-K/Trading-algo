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
