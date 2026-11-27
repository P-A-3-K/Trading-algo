# Research log

The project's memory: what we built, what we tested (with numbers), what failed and why, and
what's next. Newest status first, then the history. Sharpe = yearly, after costs unless noted.
"A" = the long run, "B" = 2025 to today.

---

## Current status (2026-11-27)

**Goal:** pass an FTMO 2-Step Swing challenge (100K, EUR) with an automated strategy that has a
real edge in BOTH the long run and 2025-2026.

**Where we are:** no strategy is ready for a paid challenge yet. The only surviving candidate is
**ORB (opening range breakout, 30-minute range, no target, US session 9:30 New York) on a
portfolio of GER40.cash + US100.cash + XAUUSD**, whose edge per trade is only 2-4x FTMO's
cost per trade. The unknown that decides it is **real slippage** of stop orders.

**Next step (agreed direction, not started):**
1. ~~Build a fill report~~ done: `python -m quantlab.fills` (asked vs got price for every robot
   fill, slippage in bp, cost per round trip vs the backtest's assumption).
2. Run `QuantLabORB` on an FTMO demo on GER40.cash, US100.cash, XAUUSD, all `Session = US`,
   hosted on MetaQuotes VPS (~$15/month; the user can't keep their PC on).
   2026-11-27: started on the user's PC first (3 charts, Session US, weight 0.33, plan size 1,
   start balance 80000) to check everything runs; VPS after that.
3. After ~2 months (~120 trades): if real costs are at or below the "2% slippage" level, rerun
   the prop check / challenge plan on the 3-market portfolio and decide on the challenge.

**Infrastructure that is ready:** MT5 data + costs, edge finders, prop simulation, live Python
trader, MQL5 EA (compiled OK on the user's PC, verified in Strategy Tester, matches QuantLab).

---

## Key findings (the lessons)

1. **FTMO swaps kill multi-day holds.** Long US indices pay ~9.2-9.4%/yr, GER40 ~7.3%, gold
   ~7.5%; most FX pairs charge on both sides. A strategy in a position most nights loses
   0.5-1.0 Sharpe to swaps (daily edge finder). On FTMO, only intraday (flat at close) is viable
   unless an edge is huge.
2. **The famous index-timing edges are fading.** ORB on QQQ: Sharpe 1.23 (2021-24) -> 0.49 (2025)
   -> -0.48 (FTMO US100, 2026). IndexDip and DE40 ORB also weak in 2025-2026.
3. **Single-market results are noise; portfolios across unrelated markets are what's real.**
   ORB on 11 ETFs before costs: Sharpe 1.49 (2017-24) and 1.16 (2025+); single markets -0.3 to 0.8.
4. **For ORB, costs decide everything.** Same 11-ETF portfolio: no costs 1.49 / 1.16, low costs
   (0.25 bp + 2% slippage) 1.00 / 0.65, full (0.5 bp + 5%) 0.43 / 0.05.
5. **Simple trend following has no edge on FTMO's markets 2019-2024 even before costs**
   (portfolio Sharpe -0.18 to -0.65 gross). Professional trend funds earned largely in bond/rate
   futures, which FTMO doesn't offer.
6. Spreads matter little for daily strategies; slippage and swaps are the big costs.

---

## Experiments (newest first)

### FTMO ORB check (`python -m quantlab.ftmo_orb`), 2026-11-26
Edge per ORB trade (ETF stand-in, 2017-today, gross) vs FTMO's real cost per trade
(per-candle spread + 2% slippage):

| FTMO market | ETF | edge bp | cost bp | edge/cost | FTMO's own gross edge bp |
|---|---|---|---|---|---|
| GER40.cash (US session) | EWG | 3.73 | 0.96 | 3.9x | 1.20 (2026) |
| US100.cash | QQQ | 2.76 | 1.05 | 2.6x | 0.61 (2026) |
| XAUUSD | GLD | 2.99 | 1.53 | 2.0x | 0.72 (FTMO history) |
| US500.cash | SPY | 1.41 | 1.12 | 1.3x | 1.62 |
| EURUSD | FXE | 1.39 | 1.26 | 1.1x | 0.68 (~22 years) |
| USDJPY | FXY | 1.60 | 1.59 | 1.0x | 0.39 |
| US30 / JP225 / US2000 / silver / oil | | | | 0.8x / 0.9x / 0.5x / 0.1x / 0.2x | |

FTMO half-spreads: US100 0.28 bp, US500 0.39, GER40 0.32, gold 0.47, US2000 1.67, silver 4.5,
oil 4.1. Caveat: EWG is a US-listed ETF (also moves with EUR/USD), an imperfect GER40 stand-in.
FTMO keeps 5m history only for 2026 on indices; FX/metals go back many years.

### Intraday edge finder (`python -m quantlab.intraday_edges`), 2026-11-26
11 ETFs (SPY QQQ IWM DIA EWG EWJ GLD SLV USO FXE FXY), Alpaca 5m 2017-today, flat at close.
Portfolio Sharpe A / B, no costs -> full costs:
- ORB 30m: 1.49 / 1.16 -> 0.43 / 0.05 (low costs 1.00 / 0.65). **Only real edge.**
- First/last half hour: -0.37 / 0.48 -> -3.24 / -3.02. Dead.
- Late-day trend (14:30 signal, hold 15:00-16:00): 0.09 / 0.47 -> -1.77 / -1.32. Dead.
- Gap fade (0.2-1.5% gaps, exit at fill or 12:00): -0.03 / -0.90 -> -0.99 / -1.90. Dead.
- VWAP reversion (0.25 x ADR band): -0.29 / -0.29 -> -1.96 / -1.69. Dead.

### Daily edge finder (`python -m quantlab.edges`), 2026-11-26
46 FTMO markets (FX, metals, energy, indices, crypto, softs), daily, real spreads and swaps,
textbook settings, long/short. Portfolio Sharpe A (2019-24) / B (2025+):

| Strategy | A no costs | A full | B no costs | B full |
|---|---|---|---|---|
| Momentum 12m | -0.18 | -0.94 | 0.36 | -0.62 |
| Momentum 3m weekly | -0.65 | -1.37 | -0.15 | -1.03 |
| Donchian 55/20 | -0.43 | -0.91 | -0.09 | -0.71 |
| Donchian 20/10 | -0.31 | -0.88 | -0.19 | -1.36 |

Only metals positive in B (gold's 2025-26 bull run; negative in A). Verdict: no.

### IndexDip on FTMO daily data, 2026-11-26
Before costs, 2018-24 vs 2025+: US500 +6.0% vs +3.4%/yr, US100 +13.9% vs +2.2%, GER40 +2.1% vs
-4.5%. Swaps (~2%/yr at its time in market) leave ~0 or negative in 2025+. Verdict: no.

### ORB year by year, 2026-11-25
- QQQ (spread 0.3 bp): 2021-24 Sharpe 1.23 (+14.6%/yr), 2025 0.49 (+5.1%).
- FTMO US100 2026 (real spreads): -0.48 (-5.6%/yr, -1.3% before costs).
- DE40 Europe session (MetaQuotes data): 2021-24 0.47, 2025 -0.40, 2026 0.23.
- So the US100 + DE40 ORB portfolio was ~flat in 2025-2026. **Challenge plan on it cancelled.**

### MQL5 EA verification, 2026-11-25
`QuantLabORB` compiled with 0 errors. Strategy Tester US100.cash May-Jul 2026, Weight 1:
59 trades, 39% winners, -8% then the total-loss guard stopped it (as designed). QuantLab on the
same period: 50 trades, about -10%. Times (16:30-17:00 server = 9:30-10:00 NY), OCO, stops,
sizing and the 15:58 flatten all verified in the journal.

### Earlier (before 2026-11-25), short version
- Portfolio QQQ ORB + DE40 ORB 50/50: Sharpe 0.89 (2016-2026 overall), correlation -0.01,
  drawdown -12.6%. Challenge plan chosen: 2x size in the challenge (1x after -4%), 0.75x funded:
  funded within 24 months 73%, net about EUR 6.7k. **Superseded:** the edge was weak in 2025-26.
- QQQ ORB rr=0, 5 years, FTMO-like costs: Sharpe 0.92, but negative in 2026.
- Failed: intraday momentum (first/last half hour), FVG (walk-forward), trend legs on CFDs
  (swaps), JP225 ORB, US2000 ORB, UK100 (MetaQuotes data too patchy: 479 of 1,208 days).
- GLD SMA 25/150 and 4h scans, MTF RSI, FVG with trend filter: nothing robust out of sample.

---

## Decisions

- **Prop firm:** FTMO 2-Step Swing, 100K (researched FTMO and The5ers). Swing allows weekend
  holds; swaps still apply. FTMO Futures: no swaps but must be flat by 16:10 NY, end-of-day
  trailing drawdown (~4-6%), NinjaTrader/Tradovate/TradingView (no MT5): not useful for now.
- **Run live trading in the cloud:** the user can't keep their PC on, so the MQL5 EA runs on
  MetaQuotes' built-in VPS. Activate the VPS only once a strategy is worth running.
- **Data:** FTMO 5m history only covers 2026 for indices, so long-run intraday tests use Alpaca
  ETFs (5m since 2016) as stand-ins; daily FTMO history goes back to 2017-2018.
- **Dukascopy:** its data API sits behind a web firewall; we don't circumvent it. Use the CSV
  importer (`python -m quantlab.importer`) with manual exports instead.

---

## Fixes worth remembering
- Slippage was 5% of the WHOLE candle's range, overstating costs on daily/hourly candles ~10x;
  now 5% of a 5-minute candle's range on every timeframe (2026-11-26).
- MT5 loading now falls back to saved candles when the connected broker doesn't know the
  symbol (e.g. DE40 saved from MetaQuotes while logged into FTMO).
- EA: automatic magic number is unique per symbol, so several charts can run at once.

## How to run things (user's PC, PowerShell, in the repo folder)
- Update: `git pull`. App: `start.bat`.
- MT5 data: `.venv\Scripts\python -m quantlab.mt5 download US100.cash` (MT5 open, logged in).
- A/B test: `.venv\Scripts\python -m quantlab.compare MT5:US100.cash 5m "OpeningRangeBreakout rr=0" --from 2026-01-01`
- Fill report (real slippage of the robot's trades): `.venv\Scripts\python -m quantlab.fills [--days 30]`
- Edge finders: `python -m quantlab.edges`, `python -m quantlab.intraday_edges [--only ORB]`,
  `python -m quantlab.ftmo_orb`.
- EA: `.venv\Scripts\python -m quantlab.mql5 install` (copies + compiles into MT5).
- Python live trader (needs PC on): `python -m quantlab.live init|run --dry-run|status|stop`.
