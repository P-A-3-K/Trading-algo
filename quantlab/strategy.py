"""The Strategy class: the plug-in format every QuantLab strategy follows.

A strategy is a small class with:
  * name, description   shown in the app
  * params              its settings, each with a default and allowed range
  * compute(bars)       returns the position for every day
                        (1 = long, 0 = cash, -1 = short, or anything between)
  * self.plot(...)      lines it wants drawn on the chart (e.g. moving averages)

Example:

    class SmaCrossover(Strategy):
        name = "SMA crossover"
        description = "Long when the fast average is above the slow one."
        params = {"fast": Param(50, 2, 400, "Fast average (days)"),
                  "slow": Param(200, 5, 500, "Slow average (days)")}

        def compute(self, bars):
            fast = bars.close.rolling(self.fast).mean()
            slow = bars.close.rolling(self.slow).mean()
            self.plot(f"SMA {self.fast}", fast)
            self.plot(f"SMA {self.slow}", slow)
            return (fast > slow).astype(float)

    result = SmaCrossover(fast=20).run(bars)
    result.positions, result.plots

`bars` is a DataFrame of daily open/high/low/close/volume (adjusted prices),
indexed by date. Use only data up to each day: check_no_lookahead() tests it.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Param:
    """One setting of a strategy, e.g. the length of a moving average."""
    default: float
    min: Optional[float] = None
    max: Optional[float] = None
    label: str = ""
    step: Optional[float] = None   # app input step; defaults to 1 for whole numbers
    intraday_only: bool = False    # only shown/used on intraday timeframes
    help: str = ""                 # one-line explanation, shown when hovering the setting
    choices: Optional[tuple] = None  # a pick-one setting, e.g. ("1h", "4h", "1d"): a menu in the app

    @property
    def is_choice(self) -> bool:
        return self.choices is not None

    @property
    def is_int(self) -> bool:
        return (not self.is_choice and isinstance(self.default, (int, np.integer))
                and not isinstance(self.default, bool))

    @property
    def is_bool(self) -> bool:
        return isinstance(self.default, bool)

    def coerce(self, name: str, value):
        """Convert and range-check a value, with a clear error message."""
        if self.is_choice:                        # pick-one settings
            v = str(value).strip()
            if v not in self.choices:
                raise ValueError(f"{name} must be one of {', '.join(self.choices)}; got {value!r}")
            return v
        if self.is_bool:                          # yes/no settings
            if isinstance(value, str):
                v = value.strip().lower()
                if v in ("1", "true", "yes", "on"):
                    return True
                if v in ("0", "false", "no", "off", ""):
                    return False
                raise ValueError(f"{name} must be yes or no, got {value!r}")
            return bool(value)
        try:
            value = int(value) if self.is_int else float(value)
        except (TypeError, ValueError):
            kind = "a whole number" if self.is_int else "a number"
            raise ValueError(f"{name} must be {kind}, got {value!r}") from None
        if self.min is not None and value < self.min:
            raise ValueError(f"{name} must be at least {self.min}, got {value}")
        if self.max is not None and value > self.max:
            raise ValueError(f"{name} must be at most {self.max}, got {value}")
        return value

    def describe(self) -> dict:
        kind = "choice" if self.is_choice else "bool" if self.is_bool else "int" if self.is_int else "float"
        out = {"default": self.default, "min": self.min, "max": self.max,
               "label": self.label, "step": self.step or (1 if self.is_int else None),
               "type": kind, "intraday_only": self.intraday_only, "help": self.help}
        if self.is_choice:
            out["choices"] = list(self.choices)
        return out


# Settings every strategy gets automatically (on top of its own `params`)
BUILTIN_PARAMS = {
    "flat_at_close": Param(False, label="Day trade: flat outside 9:30-16:00", intraday_only=True,
                           help="Yes: hold positions only 9:30-16:00 New York and close everything "
                                "at the last regular-hours bar. Candles from all hours still feed "
                                "the signals."),
}


def regular_hours(index) -> np.ndarray:
    """True for intraday bars that start within regular hours (9:30-16:00 New York)."""
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        return np.ones(len(index), dtype=bool)
    minutes = index.hour * 60 + index.minute
    return np.asarray((minutes >= 9 * 60 + 30) & (minutes < 16 * 60))


def hold_allowed(index) -> np.ndarray:
    """For day trading: may a position be held from this bar's close into the next bar?

    Only if the next bar (this bar's start + one bar length: the schedule, which is
    known in advance) starts in regular hours on the same day. Uses no later data,
    so it behaves the same in a live run as in a backtest.
    """
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or len(index) == 0:
        return np.ones(len(index), dtype=bool)
    step = pd.Series(index[1:] - index[:-1]).median() if len(index) > 1 else pd.Timedelta("5min")
    nxt = index + step
    return regular_hours(nxt) & np.asarray(nxt.date == index.date)


@dataclass
class Plot:
    """A line a strategy wants on the chart."""
    name: str
    values: pd.Series
    pane: str = "price"            # "price" draws over the candles, "lower" below them
    color: Optional[str] = None    # None lets the app pick
    style: str = "line"            # "line", or "step" for levels (stops, zones): flat with jumps


@dataclass
class Trade:
    """One trade with exact prices, for strategies that use orders (stops, targets...).

    entry/exit happen AT the given prices on the given days (e.g. at the open,
    or at a stop level during the day). exit_date None = still open at the end.
    side: 1 = long, -1 = short. size: fraction of the account, 0 to 1.
    """
    entry_date: pd.Timestamp
    entry_price: float
    side: int
    size: float = 1.0
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    exit_reason: str = ""


@dataclass
class StrategyResult:
    positions: pd.Series
    plots: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    trades: Optional[list] = None     # set for trade-based strategies (self.trade)


class Strategy:
    name: str = "Unnamed strategy"
    description: str = ""
    # Longer explanation for the app's "How it works" panel. Light formatting:
    # "## Heading" lines, "- " bullet lines, **bold**, blank line = new paragraph.
    guide: str = ""
    params: dict = {}

    @classmethod
    def all_params(cls) -> dict:
        """The strategy's own settings plus the built-in ones (e.g. flat_at_close)."""
        return {**cls.params, **BUILTIN_PARAMS}

    def __init__(self, **settings):
        unknown = set(settings) - set(self.all_params())
        if unknown:
            raise ValueError(f"{self.name}: unknown setting(s) {sorted(unknown)}; "
                             f"available: {sorted(self.all_params())}")
        self.settings = {k: p.coerce(k, settings.get(k, p.default))
                         for k, p in self.all_params().items()}
        self.validate()
        self._plots, self._trades = [], []

    # Settings read like attributes: self.fast, self.slow, ...
    def __getattr__(self, item):
        settings = self.__dict__.get("settings", {})
        if item in settings:
            return settings[item]
        raise AttributeError(item)

    def validate(self):
        """Override to check settings against each other (e.g. fast < slow)."""

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("A strategy must implement compute(bars)")

    # ---- Multi-timeframe (see quantlab/mtf.py) ----------------------------------------
    def higher(self, bars: pd.DataFrame, tf: str, forming: bool = False) -> pd.DataFrame:
        """Higher-timeframe candles ("15m", "30m", "1h", "4h", "1d") for every bar.

        By default each bar sees the last CLOSED higher candle (no peeking at a
        candle that's still forming). forming=True: the candle being built so far.
            h4 = self.higher(bars, "4h");  trend_up = h4.close > h4.open
        """
        from .mtf import higher
        return higher(bars, tf, forming)

    def on_higher(self, bars: pd.DataFrame, tf: str, func) -> pd.Series:
        """An indicator computed on closed higher-timeframe candles, for every bar.
            rsi_4h = self.on_higher(bars, "4h", lambda b: rsi(b.close, 14))
        """
        from .mtf import on_higher
        return on_higher(bars, tf, func)

    def plot(self, name: str, values: pd.Series, pane: str = "price", color: str = None,
             style: str = "line"):
        """Ask the app to draw a line. Call it from compute().

        style="step" draws levels (a stop, a zone edge) as flat lines that jump,
        instead of joining one level to the next with a slope.
        """
        if pane not in ("price", "lower"):
            raise ValueError("pane must be 'price' or 'lower'")
        if style not in ("line", "step"):
            raise ValueError("style must be 'line' or 'step'")
        self._plots.append(Plot(name, values, pane, color, style))

    def trade(self, entry_date, entry_price, side, size=1.0, exit_date=None, exit_price=None,
              stop=None, target=None, exit_reason=""):
        """Record a trade with exact prices (for strategies with stops/targets).

        A strategy that records trades can return None from compute(): the
        position held at each day's close is worked out from the trades.
        """
        if side not in (1, -1):
            raise ValueError("side must be 1 (long) or -1 (short)")
        if not 0 < size <= 1 + 1e-9:
            raise ValueError("size must be above 0 and at most 1")
        if (exit_date is None) != (exit_price is None):
            raise ValueError("give both exit_date and exit_price, or neither")
        self._trades.append(Trade(entry_date, float(entry_price), side, float(size), exit_date,
                                  None if exit_price is None else float(exit_price),
                                  stop, target, exit_reason))

    def run(self, bars: pd.DataFrame) -> StrategyResult:
        """Compute positions and plots, and check the output is usable."""
        self._plots, self._trades = [], []
        positions = self.compute(bars)
        trades = None
        if self._trades or positions is None:
            from .backtest import held_positions
            trades = sorted(self._trades, key=lambda t: t.entry_date)
            for a, b in zip(trades, trades[1:]):
                if a.exit_date is None or b.entry_date < a.exit_date:
                    raise ValueError(f"{self.name}: trades overlap ({a.entry_date.date()} and "
                                     f"{b.entry_date.date()}); one trade at a time")
            trades = self._day_trade_trades(bars, trades)
            positions = held_positions(bars.index, trades)
        elif self.day_trading(bars) and isinstance(positions, pd.Series) \
                and positions.index.equals(bars.index):
            # flat outside regular hours: no position carried into a non-regular bar
            positions = positions.where(hold_allowed(bars.index), 0.0)
        if not isinstance(positions, pd.Series):
            raise TypeError(f"{self.name}: compute() must return a pandas Series")
        if not positions.index.equals(bars.index):
            raise ValueError(f"{self.name}: positions must have one value per bar "
                             "(same dates as the input)")
        positions = positions.astype(float).fillna(0.0)   # warm-up NaN -> in cash
        if (positions.abs() > 1 + 1e-9).any():
            raise ValueError(f"{self.name}: positions must be between -1 and 1")
        return StrategyResult(positions.rename(self.name), list(self._plots),
                              dict(self.settings), trades)

    # ---- Day trading (built-in "flat_at_close" setting) -----------------------------
    def day_trading(self, bars: pd.DataFrame) -> bool:
        """True when flat_at_close is on AND the bars are intraday."""
        if not self.settings.get("flat_at_close"):
            return False
        idx = bars.index
        return isinstance(idx, pd.DatetimeIndex) and idx.tz is not None

    def session_masks(self, bars: pd.DataFrame):
        """(may_enter, may_hold) arrays for trade-based strategies.

        may_enter[t]: a position may be opened during bar t (regular hours).
        may_hold[t]: a position may be carried from bar t's close into bar t+1.
        Both all-True unless day trading.
        """
        if not self.day_trading(bars):
            n = len(bars)
            return np.ones(n, dtype=bool), np.ones(n, dtype=bool)
        return regular_hours(bars.index), hold_allowed(bars.index)

    def _day_trade_trades(self, bars, trades):
        """Safety net for trade-based strategies that ignore day trading themselves:
        drop entries outside regular hours, close open trades at the last regular bar."""
        if not self.day_trading(bars):
            return trades
        rth, hold = regular_hours(bars.index), hold_allowed(bars.index)
        idx, close = bars.index, bars["close"].to_numpy()
        out = []
        for t in trades:
            i0 = idx.get_loc(t.entry_date)
            if not rth[i0]:
                continue
            j = i0 + int(np.argmin(hold[i0:])) if not hold[i0:].all() else None
            end = idx.get_loc(t.exit_date) if t.exit_date is not None else None
            if j is not None and (end is None or end > j):
                t = Trade(t.entry_date, t.entry_price, t.side, t.size, idx[j], float(close[j]),
                          t.stop, t.target, "end of day")
            out.append(t)
        return out

    @classmethod
    def info(cls) -> dict:
        """What the app needs to list the strategy and build its settings form."""
        import inspect
        guide = cls.guide or inspect.cleandoc(cls.__doc__ or "")
        return {"name": cls.name, "description": cls.description, "guide": guide.strip(),
                "params": {k: p.describe() for k, p in cls.all_params().items()}}

    def __repr__(self):
        shown = {k: v for k, v in self.settings.items()
                 if k not in BUILTIN_PARAMS or v != BUILTIN_PARAMS[k].default}
        args = ", ".join(f"{k}={v}" for k, v in shown.items())
        return f"{type(self).__name__}({args})"


def check_no_lookahead(strategy: Strategy, bars: pd.DataFrame, checks: int = 20) -> list:
    """Catch strategies that peek at future data.

    Re-runs the strategy on the data cut off at several dates. An honest
    strategy gives the same answers for the past whether or not the future
    exists. Compared on each cut-off day:
      * the position, and
      * every line the strategy draws with self.plot(), which catches cheating
        hidden in an indicator even when the position happens not to change
        (a value that exists with the future but not without it is a leak).
    Returns the dates where something changed (empty = honest).
    """
    full = strategy.run(bars)
    full_plots = {p.name: p.values for p in full.plots}
    start = min(len(bars) - 1, max(10, len(bars) // 10))
    cutoffs = np.linspace(start, len(bars) - 2, checks).astype(int)
    same = lambda a, b: bool(np.isclose(a, b, equal_nan=True))
    leaks = []
    for i in sorted(set(cutoffs)):
        part = strategy.run(bars.iloc[: i + 1])
        leaked = not same(part.positions.iloc[-1], full.positions.iloc[i])
        for p in part.plots:
            if p.name in full_plots and not same(p.values.iloc[-1], full_plots[p.name].iloc[i]):
                leaked = True
        if leaked:
            leaks.append(bars.index[i])
    return leaks
