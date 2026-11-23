"""STARTER TEMPLATE: copy this file to make your own strategy.

    1. Copy it:      strategies/_template.py  ->  strategies/my_idea.py
                     (any name not starting with "_"; files starting with "_" are ignored)
    2. Rename the class (MyStrategy -> MyIdea) and change `name`, `description` and `guide`.
    3. Change `params` and `compute()` to your idea.
    4. Check it:     python -m quantlab.check strategies/my_idea.py
    5. Refresh the app in your browser: it appears in the Strategy menu
       under "Your strategies", with its own settings boxes and an Optimize page.

This example is a complete, working strategy: RSI mean reversion (a simpler version of
strategies/rsi.py).
RSI (Relative Strength Index, 0-100) measures how strongly the price has been
rising vs falling lately. Below ~30 is "oversold" (fallen a lot), above ~70 is
"overbought". The idea: buy oversold dips, sell once the bounce is done.
"""
import pandas as pd

from quantlab.strategy import Param, Strategy


class MyStrategy(Strategy):                # <- rename: the class name is the strategy's id
    # ---- Shown in the app ------------------------------------------------------------
    name = "My strategy (template)"         # <- the name in the Strategy menu
    description = ("Buy when RSI drops below the 'buy below' level (oversold), "
                   "sell when it rises above the 'sell above' level.")
    # The "How it works" panel in the app. Formatting: "## Heading" lines,
    # "- " bullet lines, **bold**, and a blank line starts a new paragraph.
    guide = """
## In short
Describe the idea in one or two sentences.

## The rules
- When it buys.
- When it sells.

## When it tends to work
- ...

## When it struggles
- ...
"""

    # ---- Settings ----------------------------------------------------------------------
    # Each Param becomes a number box in the app and a range on the Optimize page.
    #   Param(default, min, max, label)        whole numbers: write the default as 14
    #   Param(1.5, 0.1, 4.0, label, step=0.1)  decimals: write the default as 1.5
    #   Param(False, label="Also go short")    yes/no: a Yes/No menu in the app
    #   Param("4h", choices=("1h", "4h", "1d"), label="Higher timeframe")   a pick-one menu
    #   help="..."                             one line shown when hovering the setting
    # Multi-timeframe: self.higher(bars, "4h") / self.on_higher(bars, "4h", func): see README.md
    # Read them in compute() as self.<name>, e.g. self.period.
    params = {
        "period": Param(14, 2, 100, "RSI length (bars)", help="How many bars RSI looks back."),
        "buy_below": Param(30, 5, 50, "Buy when RSI below", help="Oversold level that triggers a buy."),
        "sell_above": Param(55, 30, 95, "Sell when RSI above", help="Level where the bounce is taken."),
    }

    # ---- Optional: check settings against each other ------------------------------------
    # Raise ValueError with a clear message; the app shows it, and the optimizer
    # skips those combinations.
    def validate(self):
        if self.buy_below >= self.sell_above:
            raise ValueError("'buy below' must be lower than 'sell above'")

    # ---- The strategy itself ------------------------------------------------------------
    def compute(self, bars: pd.DataFrame) -> pd.Series:
        """Return your position for every day.

        bars: one row per trading day, oldest first, adjusted for dividends & splits.
              Columns: bars.open, bars.high, bars.low, bars.close, bars.volume

        Return a pandas Series on the same dates with values from -1 to 1:
              1 = fully invested (long), 0 = in cash, -1 = fully short,
              0.5 = half invested, and so on. NaN counts as 0 (cash).

        The position on day t is decided at the close of day t and starts
        earning from day t+1; the backtester handles that delay for you.
        """
        # 1. Indicators: use only today's and PAST values (rolling, ewm, shift(1)...)
        rsi = self.rsi(bars.close, self.period)

        # 2. Lines to draw: pane="price" draws over the candles, pane="lower"
        #    in a separate panel below them (good for 0-100 oscillators like RSI)
        self.plot(f"RSI {self.period}", rsi, pane="lower")
        self.plot("Buy level", pd.Series(float(self.buy_below), index=bars.index), pane="lower")
        self.plot("Sell level", pd.Series(float(self.sell_above), index=bars.index), pane="lower")

        # 3. Signals -> positions. A common pattern for "enter on X, exit on Y":
        #    1 on entry days, 0 on exit days, NaN otherwise; then ffill() holds the
        #    last decision until the next signal.
        signal = pd.Series(float("nan"), index=bars.index)
        signal[rsi < self.buy_below] = 1.0       # oversold: buy
        signal[rsi > self.sell_above] = 0.0      # bounce done: sell
        return signal.ffill().fillna(0.0)

    # ---- Helpers: any method you like ---------------------------------------------------
    @staticmethod
    def rsi(close: pd.Series, period: int) -> pd.Series:
        """Wilder's RSI: average gain vs average loss over `period` days, scaled 0-100."""
        change = close.diff()
        gain = change.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-change.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        return 100 - 100 / (1 + gain / loss)


# ---- MISTAKES THAT MAKE BACKTESTS LIE (the app blocks the worst ones) ----------------------
#  * .shift(-1), or anything that reads a LATER row: that's tomorrow's price.
#  * rolling(..., center=True): a centred window includes future days.
#  * Using the whole series at once, e.g. close / close.max() or (close - close.mean()):
#    the max and mean include the future. Use rolling(...) or expanding() instead.
#  * Comparing today's close with a window that includes today, when you meant
#    "the previous N days": add .shift(1) to the window (see donchian_breakout.py).
