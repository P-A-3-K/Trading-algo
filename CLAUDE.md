# QuantLab: notes for Claude

Read **`docs/RESEARCH_LOG.md`** at the start of a session: it is the project's memory (what was
tested, the numbers, what failed and why, decisions, the open next step). Add to it whenever
something meaningful is learned or decided, in the same session, and commit it.

## The project
- A local Python web app for learning algo trading by building it: candlestick charts, plug-in
  strategies, backtests with realistic costs, optimizer / walk-forward, many-markets scan,
  portfolio, prop-firm challenge simulation, MT5 connector, a live trader and an MT5 Expert Advisor.
- The user's goal: pass a **prop firm challenge** (FTMO 2-Step Swing, 100K, EUR account,
  automated trading) with a strategy that has a real, still-working edge.
- The user is learning: explain the reasoning before building something complex, give step by
  step instructions (Windows PowerShell, `C:\Users\<user>\Trading-algo`, `.venv\Scripts\python ...`),
  keep answers concrete. Short messages like "yes, start X" mean: go ahead and build it.
- The user's PC runs Windows with MT5 (FTMO-Demo and MetaQuotes-Demo accounts in one terminal).
  This cloud container has no MT5 and no Alpaca data: anything needing them is a command the user
  runs and pastes back.

## Code map
- `quantlab/app.py` web server (stdlib ThreadingHTTPServer) + `quantlab/web/*.html` pages.
- `quantlab/strategy.py` Strategy/Param/Trade, `check_no_lookahead`; strategies in `strategies/`
  (auto-discovered) and `quantlab/strategies.py` (built-ins: BuyAndHold, SmaCrossover, MeanReversion).
- `quantlab/backtest.py` engines (`run_backtest`, `simulate_trades`, `simulate`), `quantlab/costs.py`
  CostModel (spread, per-candle MT5 spread, slippage = share of a 5-minute candle's range, impact, swaps).
- `quantlab/download.py` Yahoo/Alpaca/MT5/CSV loading; tickers `MT5:US100.cash`, `CSV:GER40`.
- `quantlab/mt5.py` MT5 connector (`download`, `specs`, `list`); falls back to saved candles.
- `quantlab/prop.py` prop-firm rules, simulation, challenge plans; `quantlab/portfolio.py`.
- `quantlab/compare.py` CLI A/B backtests; `quantlab/scan.py` many-markets sweep.
- `quantlab/edges.py` daily edge finder (FTMO markets), `quantlab/intraday_edges.py` (ETFs, Alpaca),
  `quantlab/ftmo_orb.py` ORB edge per trade vs FTMO's real cost per trade.
- `quantlab/live.py` Python live trader for MT5; `mql5/QuantLabORB.mq5` the same rules as an
  Expert Advisor (for MetaQuotes VPS), `quantlab/mql5.py` holiday calendar + install/compile.

## Rules we keep
- Judge every idea on **two periods**: the long run AND 2025-today. Textbook settings first;
  tuning only after an idea passes, and never on the period used to judge it.
- Always include realistic costs: FTMO spreads and **swaps** (swaps kill multi-day holds on FTMO).
- No look-ahead: strategies decide at a candle's close using only data up to it.
- Tests: `python -m pytest -q` (all must pass before committing). Open text files with
  `encoding="utf-8"` (a test enforces it).
- Demo accounts only unless the user explicitly says otherwise (`allow_real` / `AllowReal` false).
- Never put the user's personal details, passwords or API keys in the repo or in requests.
  Alpaca keys live in `alpaca_keys.json` (git-ignored).
- Work on branch `claude/algo-trading-strategy-software-rmm6zn`; commit and push when done.
