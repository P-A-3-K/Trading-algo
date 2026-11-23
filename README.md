# QuantLab: learn algo trading by building a backtester

Each lesson adds one feature to the software and teaches the idea behind it.

## Quick start

**Windows:** double-click **`start.bat`**. **Mac/Linux:** run `./start.sh`.

The first time, it creates a private Python environment (`.venv`) and installs the
packages (about a minute). After that it starts in a second or two. Your browser
opens QuantLab at http://127.0.0.1:8000; close the window or press Ctrl+C to stop.

To get the newest version and start: double-click **`update.bat`** (or `./update.sh`).

**On your phone:** double-click **`start-phone.bat`** (or `./start-phone.sh`). The window shows
the address to open in your phone's browser, e.g. `http://192.168.1.23:8000`. The phone must
be on the same Wi-Fi (or use [Tailscale](https://tailscale.com) to reach it from anywhere).
There's no password, so only do this on your home network.

**Timeframes:** pick 5m, 15m, 30m, 1h, 4h or 1 day in the app. Intraday candles include
pre-market and after-hours (4:00-20:00 New York). Free intraday history is short (5m: last
60 days, 1h: last 2 years), so QuantLab keeps every download and your history grows.
Any strategy can be set to **day trade** (flat outside 9:30-16:00) on intraday timeframes.
**Costs:** pick Realistic (default), Flat 0.05%, No costs, Expensive or Custom; the stats table
shows return before costs and the $ paid, so you see how much trading costs eat.

**Which ticker and timeframe suit a strategy?** On the Optimize page pick the method
**Many markets: tickers × timeframes**, add tickers (or a whole group) and tick timeframes. The
same sweep runs on every combination. Each market's settings are picked on its in-sample part
and judged on the rest, so the ticker × timeframe heatmap compares honest, out-of-sample numbers.
The "settings across all markets" table shows which settings hold up almost everywhere.

Needs Python 3.10+ ([python.org](https://www.python.org/downloads/), tick
"Add python.exe to PATH" on Windows) and, for updates, Git.

## Command line

```bash
pip install pandas numpy requests pandas_market_calendars pytest
python -m quantlab.app                  # the QuantLab web app: http://127.0.0.1:8000
                                        #   /optimize = parameter sweep: one split, walk-forward,
                                        #   or many markets (tickers x timeframes)
python lessons/01_first_backtest.py
python lessons/02_real_data.py          # downloads SPY & AAPL into data/ on first run
python -m quantlab.download SPY QQQ     # download any tickers
python lessons/03_data_checks.py        # find gaps in the data
python -m quantlab.chart SPY            # candlestick chart -> reports/SPY.html
python lessons/04_strategy_class.py     # write your own strategy plug-in
python -m quantlab.check strategies/my_idea.py   # test a strategy file (see strategies/README.md)
pytest -q
```

## Layout
| File | What it teaches |
|---|---|
| `quantlab/download.py` | Download real prices from Yahoo, cache them in `data/` |
| `quantlab/checks.py` | Find missing and stale trading days; fill short gaps safely |
| `quantlab/optimize.py` | Parameter sweep: every combination of settings, scored in-sample and out-of-sample; overfitting report; walk-forward (re-optimize each window, trade the next) |
| `quantlab/scan.py` | Market scan: the same sweep on many tickers × timeframes; settings picked in-sample per market, compared out-of-sample; settings ranked across all markets |
| `quantlab/bootstrap.py` | Block bootstrap: 5,000 reshuffled histories, range of likely Sharpe, chance of beating buy & hold |
| `quantlab/costs.py` | Cost model: commission, spread (wider outside regular hours), slippage scaled by volatility, square-root market impact |
| `quantlab/app.py` | Local web server: chart page per ticker/strategy (set in the address), JSON API |
| `quantlab/chart.py` | Interactive candlestick chart page (TradingView Lightweight Charts, bundled in `quantlab/web/vendor`) |
| `quantlab/data.py` | Prices, returns, random walks (synthetic data = a "no edge" control group) |
| `quantlab/strategy.py` | The Strategy plug-in class: name, settings, `compute()` -> positions, `plot()` lines; look-ahead detector |
| `quantlab/strategies.py` | Built-in strategies: buy & hold, SMA crossover, mean reversion |
| `strategies/` | **Your strategies.** Start from `_template.py`, guide in `strategies/README.md`. Any `Strategy` class in a `.py` file here appears in the app automatically (`quantlab/discovery.py`) |
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
