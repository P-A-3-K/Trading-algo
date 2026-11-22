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
2. **Rename** the class (`class RsiReversion` → `class MyIdea`) and set `name` and `description`.
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
| Always invested / always in cash | `pd.Series(1.0, index=bars.index)` / `0.0` |
| Buy on X, sell on Y, hold in between | `s = pd.Series(float("nan"), index=bars.index); s[X] = 1.0; s[Y] = 0.0; return s.ffill().fillna(0.0)` |

**Positions:** `1` = fully invested, `0` = cash, `-1` = fully short, `0.5` = half.
The position you return for a day is decided at that day's close and earns from the next day.

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
| `_template.py` | Fully commented starter (RSI mean reversion). Hidden from the app because of the `_`. |
| `donchian_breakout.py` | Example: buy new N-day highs, sell new M-day lows. |
| `fvg.py` | Fair value gap retest: next-open entry, swing stop, reward:risk target, risk-% sizing, long & short. Uses `self.trade(...)`. |
| `README.md` | This guide. |
