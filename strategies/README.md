# Your strategies

Every `.py` file in this folder is scanned by the app. Any `Strategy` class in it
appears in the **Strategy** menu (under "Your strategies"), gets its own settings
boxes, chart lines, buy/sell arrows, stats, and an **Optimize** page. No other
code to change, no restart: save the file and refresh the browser.

## Make a new strategy in 5 steps

1. **Copy the template**: `_template.py` → `my_idea.py` (any name not starting with `_`).
   ```
   copy strategies\_template.py strategies\my_idea.py      (Windows)
   cp strategies/_template.py strategies/my_idea.py        (Mac/Linux)
   ```
2. **Rename** the class (`class MyStrategy` → `class MyIdea`) and set `name`, `description` and `guide`
   (the text in the app's **ⓘ How it works** panel). Add `help="..."` to each `Param` for its hover tip.
3. **Write your idea** in `params` (the settings) and `compute()` (the positions).
4. **Check it** in the terminal:
   ```
   python -m quantlab.check strategies/my_idea.py        (add a ticker to test on it: ... QQQ)
   ```
5. **Refresh the app.** If something's wrong, a red box above the chart says what and where.

## Cheat sheet

| You want | Write |
|---|---|
| Today's close / high / low / open / volume | `bars.close`, `bars.high`, `bars.low`, `bars.open`, `bars.volume` |
| N-day moving average | `bars.close.rolling(n).mean()` |
| Exponential moving average | `bars.close.ewm(span=n).mean()` |
| N-day highest high, **not counting today** | `bars.high.rolling(n).max().shift(1)` |
| N-day return (momentum) | `bars.close.pct_change(n)` |
| Volatility (daily std over N days) | `bars.close.pct_change().rolling(n).std()` |
| Yesterday's value | `x.shift(1)` |
| A setting | `self.my_setting` (from `params`) |
| Draw a line on the candles | `self.plot("SMA 50", series)` |
| Draw in the panel below | `self.plot("RSI", series, pane="lower")` |
| Draw a level (stop, zone edge) as flat steps | `self.plot("Stop", series, style="step")` |
| RSI / moving averages ready-made | `from quantlab.indicators import rsi, sma, ema` |
| Higher-timeframe candles (closed only) | `h4 = self.higher(bars, "4h")` then `h4.close`, `h4.high`... |
| An indicator on a higher timeframe | `self.on_higher(bars, "4h", lambda b: rsi(b.close, 14))` |
| A pick-one setting (menu) | `Param("4h", choices=("1h", "4h", "1d"), label="Higher timeframe")` |
| Always invested / always in cash | `pd.Series(1.0, index=bars.index)` / `0.0` |
| Buy on X, sell on Y, hold in between | `s = pd.Series(float("nan"), index=bars.index); s[X] = 1.0; s[Y] = 0.0; return s.ffill().fillna(0.0)` |

**Timeframes:** the same code runs on daily or intraday candles (5m ... 4h). Settings count
**bars**, so `rolling(20)` is 20 days on daily candles but 100 minutes on 5m. Intraday bars
are indexed by New York time and include pre-market and after-hours.

**Day trading:** every strategy automatically gets a "Day trade: flat outside 9:30-16:00"
setting on intraday timeframes; you don't need to code anything for position-based
strategies. Trade-based strategies (`self.trade`) can use
`may_enter, may_hold = self.session_masks(bars)` to respect it inside their loop
(see `fvg.py`); otherwise trades are cut at the close automatically.

**Positions:** `1` = fully invested, `0` = cash, `-1` = fully short, `0.5` = half.
The position you return for a day is decided at that day's close and earns from the next day.

## Multi-timeframe strategies

Let a higher timeframe set the direction and the chart's own timeframe time the entries:

```python
from quantlab.indicators import rsi

def compute(self, bars):                      # e.g. on 15-minute candles
    trend_up = self.on_higher(bars, "4h", lambda b: rsi(b.close, 14)) > 50
    dip = rsi(bars.close, 14) < 30
    ...
```

- `self.higher(bars, tf)` gives the higher timeframe's open/high/low/close/volume on every bar;
  `self.on_higher(bars, tf, func)` computes an indicator on those candles and lines it up.
- **Only finished candles:** at 09:00 a strategy sees the 04:00-08:00 4h candle, never the one
  that ends at 12:00. (`self.higher(bars, tf, forming=True)` gives the candle being built so far,
  from the bars up to now: also honest.)
- Don't do it yourself with `bars.close.resample("4h").last()` + forward fill: that leaks the
  future into the past, and the app's look-ahead check will block the strategy.
- Candles are built from the chart's own bars (New York clock; 4h bins 0-4-8-12-16-20; days from
  midnight, pre/after-market included). Make the timeframe a setting with
  `Param("4h", choices=(...))` so the optimizer can test several (tick-boxes on the Optimize page).
- Examples: `mtf_rsi.py`, and `fvg.py`'s trend filter.

## Strategies with stops, targets or exact entry prices

Positions are traded at the **close** by default. For orders at exact prices
(enter at the next open, exit at a stop or take-profit level), record each trade
instead and `return None` from `compute()`:

```python
self.trade(entry_date, entry_price, side=1, size=0.5,        # side 1 long / -1 short; size 0-1
           exit_date=exit_date, exit_price=exit_price,        # leave out while still open
           stop=stop, target=target, exit_reason="target")
```

The backtester then values the account from those exact prices, and the chart puts
the arrows at them. Only use prices a real order could get: today's open, or a level
inside today's high-low range (if both your stop and target are inside the same day's
range, assume the stop: daily bars don't show which came first). See `fvg.py`.

## The rules that keep backtests honest

- Use only **today's and past** data. Never `shift(-1)`, `rolling(..., center=True)`,
  or whole-series numbers like `close.max()` / `close.mean()` (they include the future).
- The app runs a **look-ahead check** on every strategy and blocks ones that fail.
  It compares positions *and* plotted lines with and without the future, so plot
  the indicators you trade on: that makes the check much stronger.
- A great result on one ticker with hand-picked settings means little. Use the
  **Optimize** page and look at the *out-of-sample* column before believing it.

## Files here

| File | What it is |
|---|---|
| `_template.py` | Fully commented starter (class `MyStrategy`, a simple RSI idea). Hidden from the app because of the `_`. |
| `donchian_breakout.py` | Example: buy new N-day highs, sell new M-day lows. |
| `macd.py` | MACD crossover: long (or short) while MACD is above (below) its signal line. |
| `rsi.py` | RSI reversion: buy oversold, sell the bounce; optional shorts. |
| `fvg.py` | Fair value gap retest: next-open entry, swing stop, reward:risk target, risk-% sizing, long & short, optional higher-timeframe trend filter. Uses `self.trade(...)`. |
| `index_dip.py` | Index dip-buying: IBS or RSI(2) dips above the 200-day average, sell the bounce (daily, SPY/QQQ...). |
| `orb.py` | Opening range breakout: trade the first break of the first 30 minutes' range, stop at the other side, flat at the close (intraday). |
| `ts_momentum.py` | Time-series momentum: hold while the past 12 months' return is positive, checked monthly. Best across many asset classes on the Portfolio page. |
| `mtf_rsi.py` | Multi-timeframe RSI: buy short-term dips only while a higher timeframe's RSI shows an up-trend. |
| `README.md` | This guide. |
