"""Auto-discovery: every Strategy class in a .py file in strategies/ shows up in the app.

    strategies/
        donchian_breakout.py     class Breakout(Strategy): ...
        my_idea.py               class MyIdea(Strategy): ...

Files are re-scanned on every request and re-loaded when they change, so a new
or edited strategy appears on the next page load: no restart needed.

Two safety nets:
  * a file that fails to load (typo, missing import...) is reported with the
    error, and everything else keeps working
  * every discovered strategy is run through check_no_lookahead() on made-up
    prices; one that peeks at the future is listed but blocked from running
"""
import importlib.util
import inspect
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .strategy import Strategy, check_no_lookahead

STRATEGY_DIR = Path(__file__).resolve().parents[1] / "strategies"


@dataclass
class Found:
    cls: type
    file: str                      # e.g. "strategies/donchian_breakout.py"
    problem: str = ""              # set when the strategy is blocked


@dataclass
class Scan:
    strategies: dict = field(default_factory=dict)   # id -> Found
    errors: list = field(default_factory=list)       # (file, message)


_cache = {}   # path -> (mtime, [Found], error or None)


def _test_bars(n: int = 400) -> pd.DataFrame:
    """Made-up daily bars for the look-ahead check (no download needed)."""
    from .data import synthetic_prices
    close = synthetic_prices(n_days=n, seed=11)
    return pd.DataFrame({"open": close.shift(1).fillna(close.iloc[0]), "high": close * 1.01,
                         "low": close * 0.99, "close": close, "volume": 1_000_000.0})


def _check(cls) -> str:
    """'' if the strategy runs honestly on default settings, else what's wrong."""
    try:
        leaks = check_no_lookahead(cls(), _test_bars())
    except Exception as e:
        return f"crashed when tested with its default settings: {type(e).__name__}: {e}"
    if leaks:
        return ("it looks into the future: its position or one of its chart lines for a day "
                "changes once later prices are added (check for shift(-1), centred windows, "
                "or using the whole series)")
    return ""


def _load(path: Path, rel: str):
    """Import one file and return the Strategy classes defined in it."""
    name = f"user_strategies.{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module                 # dataclasses etc. need it registered
    try:
        # Compile straight from the source text, never from __pycache__: Python's
        # cached .pyc is judged by size + whole-second mtime, so two quick edits of
        # the same length could otherwise run the OLD code.
        source = path.read_text(encoding="utf-8")
        exec(compile(source, str(path), "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        last = traceback.format_exc().strip().splitlines()
        # keep it short: the line in the user's file and the error itself
        where = next((l.strip() for l in reversed(last) if str(path) in l), "")
        msg = f"{last[-1]}" + (f" ({where})" if where else "")
        raise RuntimeError(msg.replace(str(path), rel))
    found = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Strategy) and obj is not Strategy and obj.__module__ == name:
            found.append(Found(obj, rel, _check(obj)))
    return found


def scan(folder: Path = None, builtin: dict = None) -> Scan:
    """Find every Strategy in folder/*.py (files starting with _ are skipped)."""
    folder = Path(folder or STRATEGY_DIR)
    out = Scan()
    taken = set(builtin or {})
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("*.py")):
        if path.name.startswith("_"):
            continue
        rel = f"{folder.name}/{path.name}"
        mtime = path.stat().st_mtime
        cached = _cache.get(path)
        if cached and cached[0] == mtime:
            found, error = cached[1], cached[2]
        else:
            try:
                found, error = _load(path, rel), None
            except RuntimeError as e:
                found, error = [], str(e)
            _cache[path] = (mtime, found, error)
        if error:
            out.errors.append((rel, error))
        for f in found:
            sid = f.cls.__name__
            if sid in taken:
                out.errors.append((rel, f"class name {sid!r} is already used by another "
                                        "strategy; rename the class"))
                continue
            taken.add(sid)
            out.strategies[sid] = f
    return out
