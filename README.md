# QuantLab: learn algo trading by building a backtester

Each lesson adds one feature to the software and teaches the idea behind it.

```bash
pip install pandas numpy pytest
python lessons/01_first_backtest.py
pytest -q
```

## Layout
| File | What it teaches |
|---|---|
| `quantlab/data.py` | Prices, returns, random walks (synthetic data = a "no edge" control group) |
| `quantlab/strategies.py` | A strategy is a function: prices -> position (1 long, 0 flat, -1 short) |
| `quantlab/backtest.py` | Simulation, `shift(1)` against look-ahead bias, transaction costs |
| `quantlab/metrics.py` | CAGR, volatility, Sharpe, max drawdown |

## Curriculum
**Phase 1: Foundations (the engine)**
1. ✅ First backtest: positions, returns, costs, look-ahead bias, luck vs. skill
2. Real data: load actual stock/ETF/crypto history, adjusted prices, survivorship bias
3. Better metrics: Sortino, Calmar, win rate, turnover, benchmark-relative returns, charts

**Phase 2: Strategy families**

4. Trend following / momentum (time-series and cross-sectional)
5. Mean reversion and pairs trading (cointegration, z-scores)
6. Multi-asset portfolios: combining signals, rebalancing

**Phase 3: Not fooling yourself (the "quant" part)**

7. Overfitting: parameter sweeps, train/test splits, walk-forward testing
8. Statistical significance: bootstrap, deflated Sharpe, multiple-testing
9. Realistic execution: slippage models, order types, an event-driven engine

**Phase 4: Risk and going live**

10. Position sizing: volatility targeting, Kelly, stop-losses
11. Risk management: drawdown limits, correlation, regime changes
12. Paper trading: connecting to a broker API, with no real money
