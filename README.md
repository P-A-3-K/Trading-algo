# QuantLab: learn algo trading by building a backtester

Each lesson adds one feature to the software and teaches the idea behind it.

```bash
pip install pandas numpy requests pandas_market_calendars pytest
python -m quantlab.app                  # the QuantLab web app: http://127.0.0.1:8000
                                        #   /optimize = parameter sweep with in/out-of-sample check
python lessons/01_first_backtest.py
python lessons/02_real_data.py          # downloads SPY & AAPL into data/ on first run
python -m quantlab.download SPY QQQ     # download any tickers
python lessons/03_data_checks.py        # find gaps in the data
python -m quantlab.chart SPY            # candlestick chart -> reports/SPY.html
python lessons/04_strategy_class.py     # write your own strategy plug-in
pytest -q
```

## Layout
| File | What it teaches |
|---|---|
| `quantlab/download.py` | Download real prices from Yahoo, cache them in `data/` |
| `quantlab/checks.py` | Find missing and stale trading days; fill short gaps safely |
| `quantlab/optimize.py` | Parameter sweep: every combination of settings, scored in-sample and out-of-sample; overfitting report |
| `quantlab/app.py` | Local web server: chart page per ticker/strategy (set in the address), JSON API |
| `quantlab/chart.py` | Interactive candlestick chart page (TradingView Lightweight Charts, bundled in `quantlab/web/vendor`) |
| `quantlab/data.py` | Prices, returns, random walks (synthetic data = a "no edge" control group) |
| `quantlab/strategy.py` | The Strategy plug-in class: name, settings, `compute()` -> positions, `plot()` lines; look-ahead detector |
| `quantlab/strategies.py` | Built-in strategies: buy & hold, SMA crossover, mean reversion |
| `quantlab/backtest.py` | Simulation, `shift(1)` against look-ahead bias, transaction costs |
| `quantlab/metrics.py` | CAGR, volatility, Sharpe, max drawdown |

## Curriculum
**Phase 1: Foundations (the engine)**
1. ✅ First backtest: positions, returns, costs, look-ahead bias, luck vs. skill
2. ✅ Real data: download US stocks from Yahoo, cache as CSV, adjusted vs. raw prices
3. 🔶 Data checks: missing days, stale days, fill vs. drop (gaps ✅, bad prices and aligning assets next)
4. Better metrics: Sortino, Calmar, win rate, turnover, benchmark-relative returns, charts

**Phase 2: Strategy families**

5. Trend following / momentum (time-series and cross-sectional)
6. Mean reversion and pairs trading (cointegration, z-scores)
7. Multi-asset portfolios: combining signals, rebalancing

**Phase 3: Not fooling yourself (the "quant" part)**

8. Overfitting: parameter sweeps, train/test splits, walk-forward testing
9. Statistical significance: bootstrap, deflated Sharpe, multiple-testing
10. Realistic execution: slippage models, order types, an event-driven engine

**Phase 4: Risk and going live**

11. Position sizing: volatility targeting, Kelly, stop-losses
12. Risk management: drawdown limits, correlation, regime changes
13. Paper trading: connecting to a broker API, with no real money
