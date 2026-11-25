"""MetaTrader 5: prices, real spreads and swap rates from your broker or prop firm (e.g. FTMO).

Setup (Windows, once):
    1. Install MetaTrader 5 and log in (e.g. an FTMO free trial or challenge account).
    2. In the QuantLab folder:  .venv\\Scripts\\pip install MetaTrader5
Then, with MT5 running:
    python -m quantlab.mt5 list cash            # symbols whose name contains "cash"
    python -m quantlab.mt5 specs US100.cash     # contract, spread, swap rates
    python -m quantlab.mt5 download US100.cash US500.cash XAUUSD
In the app, use the ticker  MT5:US100.cash  (any timeframe; 1 day uses MT5's daily bars).

What's saved (data/mt5/): 5-minute bars (all intraday timeframes are built from them) and
daily bars, both in New York time, with each bar's spread; plus the symbol's specifications
(swap rates...) in data/mt5/specs/. After a download, MT5 only needs to run to get newer bars.

Server time: MT5 servers keep their own clock (FTMO: New York 17:00 = midnight). QuantLab
measures the difference to New York time while connected, and converts every bar.
Nothing here places orders.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PREFIX = "MT5:"
NY = "America/New_York"
DEFAULT_NY_OFFSET_H = 7.0          # FTMO-style servers: server time = New York time + 7h
TIMEFRAMES = {"5m": "TIMEFRAME_M5", "1d": "TIMEFRAME_D1"}
FIRST_YEAR = 2005


class MT5Error(Exception):
    """A problem to show the user as it is (MT5 not running, unknown symbol...)."""


def _lib():
    try:
        import MetaTrader5 as mt5                     # Windows only
    except ImportError:
        raise MT5Error("The MetaTrader5 Python package isn't installed. On Windows, in the QuantLab "
                       r"folder run:  .venv\Scripts\pip install MetaTrader5  (and keep MT5 open and logged in).")
    return mt5


def terminal_candidates() -> list:
    """Where MT5 terminals usually live on Windows: default and broker-branded installs
    (e.g. "C:\\Program Files\\FTMO MetaTrader 5\\terminal64.exe"), newest first."""
    import os
    found = []
    saved = terminal_path_file()
    if os.environ.get("MT5_PATH"):
        found.append(Path(os.environ["MT5_PATH"]))
    if saved.exists():
        found.append(Path(saved.read_text(encoding="utf-8").strip()))
    roots = [os.environ.get(k) for k in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432")]
    roots += [str(Path.home() / "AppData" / "Roaming"), str(Path.home() / "AppData" / "Local" / "Programs")]
    for root in filter(None, dict.fromkeys(roots)):
        try:
            hits = sorted(Path(root).glob("*/terminal64.exe"), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            continue
        found += hits
    return list(dict.fromkeys(p for p in found if str(p)))


def terminal_path_file() -> Path:
    return data_dir() / "terminal_path.txt"


def connect(path: str = None):
    """Connect to the running MT5 terminal; returns (module, account info dict).

    Finds the terminal on its own (default and broker-branded installs); a path given
    here (or with --path, or in the MT5_PATH environment variable) is remembered."""
    mt5 = _lib()
    tried = []
    if path:
        ok = mt5.initialize(path=str(path))
        tried.append(str(path))
    else:
        ok = mt5.initialize()
        if not ok:
            for cand in terminal_candidates():
                if cand.exists() and mt5.initialize(path=str(cand)):
                    ok, path = True, str(cand)
                    break
                tried.append(str(cand))
    if not ok:
        where = ("Tried: " + "; ".join(tried[:6]) + ". ") if tried else ""
        raise MT5Error(f"Couldn't connect to MetaTrader 5 ({mt5.last_error()}). Is MT5 open and logged in? {where}"
                       "If it's installed somewhere else, give its terminal64.exe, e.g.\n"
                       '  .venv\\Scripts\\python -m quantlab.mt5 list --path "C:\\Program Files\\FTMO MetaTrader 5\\terminal64.exe"\n'
                       "(right-click the MT5 shortcut -> Open file location to find it).")
    if path:                                                  # remember the terminal that worked
        try:
            terminal_path_file().parent.mkdir(parents=True, exist_ok=True)
            terminal_path_file().write_text(str(path), encoding="utf-8")
        except OSError:
            pass
    acc = mt5.account_info()
    info = {"server": getattr(acc, "server", ""), "company": getattr(acc, "company", ""),
            "currency": getattr(acc, "currency", ""), "login": getattr(acc, "login", 0)}
    return mt5, info


def saved_name(symbol: str, tf: str) -> str:
    """The saved file's symbol name, matched without caring about case (US100.CASH -> US100.cash)."""
    folder = data_dir() / tf
    if (folder / f"{symbol}.csv").exists():
        return symbol
    for p in folder.glob("*.csv"):
        if p.stem.lower() == symbol.lower():
            return p.stem
    return symbol


def resolve(mt5, symbol: str) -> str:
    """MT5's own spelling of a symbol name (names typed in capitals still match)."""
    if mt5.symbol_select(symbol, True):
        return symbol
    for s in mt5.symbols_get() or []:
        if s.name.lower() == symbol.lower() and mt5.symbol_select(s.name, True):
            return s.name
    raise MT5Error(f"MT5 doesn't know {symbol!r}. List symbols with: python -m quantlab.mt5 list")


def is_mt5(ticker: str) -> bool:
    return str(ticker).upper().startswith(PREFIX)


def symbol_of(ticker: str) -> str:
    return str(ticker)[len(PREFIX):] if is_mt5(ticker) else str(ticker)


def data_dir() -> Path:
    from .download import DATA_DIR
    return DATA_DIR / "mt5"


def cache_path(symbol: str, tf: str) -> Path:
    return data_dir() / tf / f"{symbol}.csv"


def specs_path(symbol: str) -> Path:
    return data_dir() / "specs" / f"{symbol}.json"


# ---- Time: server clock -> New York ------------------------------------------------------

def ny_offset_hours(mt5, symbol: str) -> float:
    """Hours to subtract from server time to get New York time (FTMO: 7).

    Measured from the latest tick while the market is open; falls back to the
    common 'New York close = midnight' convention (7h) when ticks are stale."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return DEFAULT_NY_OFFSET_H
    server_minus_utc = (tick.time - time.time()) / 3600
    nearest = round(server_minus_utc * 2) / 2
    if abs(server_minus_utc - nearest) > 0.05:              # stale tick (market closed)
        return DEFAULT_NY_OFFSET_H
    ny_minus_utc = pd.Timestamp.now(tz=NY).utcoffset().total_seconds() / 3600
    return nearest - ny_minus_utc


def to_frame(rates, point: float, offset_h: float, daily: bool) -> pd.DataFrame:
    """MT5 rates (structured array) -> OHLCV + Spread (in price units), New York time."""
    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "Spread"])
    df = pd.DataFrame(rates)
    server = pd.to_datetime(df["time"], unit="s")             # server wall clock, no zone
    out = pd.DataFrame({"Open": df["open"].astype(float), "High": df["high"].astype(float),
                        "Low": df["low"].astype(float), "Close": df["close"].astype(float),
                        "Volume": df.get("tick_volume", df.get("real_volume", 0)).astype(float),
                        "Spread": df.get("spread", 0).astype(float) * point})
    if daily:
        out.index = pd.DatetimeIndex(server.dt.normalize(), name="Date")
    else:
        ny_wall = server - pd.to_timedelta(offset_h, unit="h")
        idx = pd.DatetimeIndex(ny_wall).tz_localize(NY, ambiguous="NaT", nonexistent="NaT")
        out.index = pd.DatetimeIndex(idx, name="Datetime")
        out = out[out.index.notna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


# ---- Specs: swaps and contract details ---------------------------------------------------

def swap_bps_per_night(info, price: float):
    """(long, short) swap COST per night in bp of the position's value (positive = you pay).

    MT5 gives swaps in several 'modes'; the common ones are converted, others give None."""
    mode = getattr(info, "swap_mode", 0)
    lng, sht = float(getattr(info, "swap_long", 0.0)), float(getattr(info, "swap_short", 0.0))
    point = float(getattr(info, "point", 0.0)) or 1e-9
    size = float(getattr(info, "trade_contract_size", 1.0)) or 1.0
    if mode == 0 or price <= 0:                              # disabled
        return 0.0, 0.0
    if mode == 1:                                            # points per lot
        f = lambda s: s * point / price
    elif mode in (2, 3, 4):                                  # money per lot (assumes account currency = quote)
        f = lambda s: s / (size * price)
    elif mode in (5, 6):                                     # interest, % per year
        f = lambda s: s / 100 / 360
    else:
        return None, None
    return -f(lng) * 10_000, -f(sht) * 10_000


def specs(symbol: str, mt5=None) -> dict:
    """Contract specification of a symbol, with swaps converted to bp per night."""
    mt5 = mt5 or connect()[0]
    symbol = resolve(mt5, symbol)
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    price = float(getattr(tick, "bid", 0) or getattr(info, "bid", 0) or 0)
    swap_l, swap_s = swap_bps_per_night(info, price)
    spread_px = float(getattr(info, "spread", 0)) * float(info.point)
    return {"symbol": symbol, "description": getattr(info, "description", ""),
            "digits": int(getattr(info, "digits", 2)), "point": float(info.point),
            "contract_size": float(getattr(info, "trade_contract_size", 1.0)),
            "currency_profit": getattr(info, "currency_profit", ""),
            "volume_min": float(getattr(info, "volume_min", 0.01)), "volume_step": float(getattr(info, "volume_step", 0.01)),
            "swap_mode": int(getattr(info, "swap_mode", 0)),
            "swap_long_raw": float(getattr(info, "swap_long", 0.0)), "swap_short_raw": float(getattr(info, "swap_short", 0.0)),
            "swap_long_bps": swap_l, "swap_short_bps": swap_s,
            "spread_now_bps": spread_px / price * 10_000 if price else None, "price": price,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def saved_specs(symbol: str) -> dict:
    try:
        return json.loads(specs_path(symbol).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# ---- Download & load --------------------------------------------------------------------

def fetch(mt5, symbol: str, tf: str, start: datetime, end: datetime, offset_h: float) -> pd.DataFrame:
    """Bars from start to end, a year at a time (MT5 limits how many come back per call)."""
    const = getattr(mt5, TIMEFRAMES[tf])
    point = float(mt5.symbol_info(symbol).point)
    parts, y = [], start.year
    while y <= end.year:
        a = max(start, datetime(y, 1, 1, tzinfo=timezone.utc))
        b = min(end, datetime(y + 1, 1, 1, tzinfo=timezone.utc))
        rates = mt5.copy_rates_range(symbol, const, a, b)
        if rates is not None and len(rates):
            parts.append(to_frame(rates, point, offset_h, daily=(tf == "1d")))
        y += 1
    if not parts:
        return to_frame(None, point, offset_h, tf == "1d")
    df = pd.concat(parts)
    return df[~df.index.duplicated(keep="last")].sort_index()


def _write(path: Path, df: pd.DataFrame, daily: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if not daily:
        out.index = out.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        out.index.name = "Datetime"
    tmp = path.with_suffix(".tmp")
    out.to_csv(tmp, float_format="%.6f")
    tmp.replace(path)


def read(path: Path, daily: bool) -> pd.DataFrame:
    if daily:
        return pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df = pd.read_csv(path, index_col="Datetime")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(NY)
    return df


def download(symbol: str, tf: str = "5m", refresh: bool = False, max_age: float = 15 * 60) -> Path:
    """Bring data/mt5/<tf>/<symbol>.csv up to date from the running MT5 (5m or 1d).

    If MT5 isn't available but a saved file exists, the saved file is used."""
    symbol = saved_name(symbol, tf)
    path = cache_path(symbol, tf)
    if path.exists() and not refresh and time.time() - path.stat().st_mtime < max_age:
        return path
    try:
        mt5, _ = connect()
    except MT5Error:
        if path.exists():
            return path
        raise
    symbol = resolve(mt5, symbol)
    path = cache_path(symbol, tf)
    offset = ny_offset_hours(mt5, symbol)
    now = datetime.now(timezone.utc)
    daily = tf == "1d"
    if path.exists():
        old = read(path, daily)
        start = (old.index[-1] - pd.Timedelta(days=3)).to_pydatetime()
        start = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start.astimezone(timezone.utc)
        new = fetch(mt5, symbol, tf, start, now, offset)
        both = pd.concat([old, new])
        df = both[~both.index.duplicated(keep="last")].sort_index()
    else:
        df = fetch(mt5, symbol, tf, datetime(FIRST_YEAR, 1, 1, tzinfo=timezone.utc), now, offset)
        if df.empty:
            raise MT5Error(f"MT5 returned no {tf} bars for {symbol!r}. Open its chart in MT5 once "
                           "(that downloads history), or raise 'Max bars in chart' in MT5 options.")
    _write(path, df, daily)
    try:                                                      # keep the swap rates current
        s = specs(symbol, mt5)
        specs_path(symbol).parent.mkdir(parents=True, exist_ok=True)
        specs_path(symbol).write_text(json.dumps(s, indent=2), encoding="utf-8")
    except Exception:
        pass
    return path


def load(ticker: str, timeframe: str, history: str = None) -> pd.DataFrame:
    """Bars for an MT5 ticker (e.g. 'MT5:US100.cash'), like download.load_bars returns."""
    from .download import HISTORY, RESAMPLE, check_history, clean_spikes, read_intraday_cached
    symbol = saved_name(symbol_of(ticker), "1d" if timeframe == "1d" else "5m")
    if timeframe == "1d":
        df = read(download(symbol, "1d"), daily=True)
    else:
        path = download(symbol, "5m")
        raw = read_intraday_cached(path)
        days = HISTORY[check_history(history)]
        if days:
            raw = raw[raw.index >= raw.index[-1] - pd.Timedelta(days=days)]
        raw = clean_spikes(raw)
        df = raw if timeframe == "5m" else resample_with_spread(raw, RESAMPLE[timeframe])
    bars = df[["Open", "High", "Low", "Close", "Volume", "Spread"]].copy()
    bars.columns = ["open", "high", "low", "close", "volume", "spread"]
    sp = saved_specs(symbol)
    bars.attrs.update(source="mt5", adjusted=False, symbol=symbol,
                      swap_long_bps=sp.get("swap_long_bps"), swap_short_bps=sp.get("swap_short_bps"))
    return bars


def resample_with_spread(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    from .download import resample_bars
    out = resample_bars(df[["Open", "High", "Low", "Close", "Volume"]], rule)
    wall = df.tz_localize(None)
    sp = wall["Spread"].resample(rule, label="left", closed="left", origin="start_day").mean()
    sp.index = sp.index.tz_localize(df.index.tz, ambiguous="NaT", nonexistent="NaT")
    out["Spread"] = sp.reindex(out.index).fillna(wall["Spread"].median())
    return out


def saved() -> list:
    """Symbols already downloaded, with their first/last bar."""
    rows = []
    for tf in ("5m", "1d"):
        for p in sorted((data_dir() / tf).glob("*.csv")):
            rows.append({"ticker": PREFIX + p.stem, "tf": tf, "size_mb": round(p.stat().st_size / 1e6, 1)})
    return rows


def main(argv=None):
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ("list", "specs", "download"):
        print(__doc__)
        return
    cmd, rest = args[0], args[1:]
    path = None
    if "--path" in rest:
        i = rest.index("--path")
        if i + 1 >= len(rest):
            print("  --path needs the full path of terminal64.exe")
            return
        path = rest[i + 1]
        rest = rest[:i] + rest[i + 2:]
    try:
        mt5, info = connect(path)
        print(f"Connected: {info['company']} · {info['server']} · account currency {info['currency']}")
        if cmd == "list":
            needle = (rest[0] if rest else "").lower()
            for s in mt5.symbols_get() or []:
                if needle in s.name.lower() or needle in getattr(s, "description", "").lower():
                    print(f"  {s.name:20s} {getattr(s, 'description', '')}")
        elif cmd == "specs":
            for sym in rest:
                s = specs(sym, mt5)
                print(f"  {sym}: {s['description']} · contract {s['contract_size']:g} · spread now "
                      f"{s['spread_now_bps'] or 0:.2f} bp · swap per night: long {s['swap_long_bps']} bp, "
                      f"short {s['swap_short_bps']} bp (mode {s['swap_mode']}, raw {s['swap_long_raw']}/{s['swap_short_raw']})")
        else:
            for sym in rest:
                for tf in ("1d", "5m"):
                    p = download(sym, tf, refresh=True)
                    df = read(p, tf == "1d")
                    print(f"  {sym} {tf}: {len(df):,} bars, {df.index[0]} -> {df.index[-1]}")
                print(f"  use it in the app as  {PREFIX}{sym}")
    except MT5Error as e:
        print(f"  {e}")


if __name__ == "__main__":
    main()
