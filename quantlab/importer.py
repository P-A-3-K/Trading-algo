"""Import candles from CSV files (Dukascopy export, MT5 export, or any OHLC CSV).

    python -m quantlab.importer GER40 GER40_2016.csv GER40_2017.csv ...   [--tz UTC]
    python -m quantlab.importer UK100 "C:\\Downloads\\uk100_*.csv" --tz UTC

Then use the ticker  CSV:GER40  anywhere in QuantLab (every timeframe; 1 day is built
from the intraday candles on the New York clock).

What it reads (detected automatically):
  * Dukascopy's historical data export: "Gmt time,Open,High,Low,Close,Volume" with times like
    "02.01.2019 00:00:00.000" (UTC), or "Local time" with "... GMT+0100" at the end.
  * MetaTrader 5 export (tab separated): <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> ...
    MT5 times are the server's clock: give --tz, e.g. --tz UTC+2 (FTMO: UTC+2 in winter).
  * Any CSV with a date/time column and open/high/low/close (volume optional).
--tz: the time zone of times WITHOUT their own offset: UTC (default), UTC+2, Europe/Berlin...

Candles shorter than 5 minutes are combined into 5-minute candles; the result is saved in
data/csv/5m/<NAME>.csv (New York time) and merged with what was imported before.
"""
import glob
import re
from pathlib import Path

import pandas as pd

PREFIX = "CSV:"
NY = "America/New_York"


def is_csv(ticker: str) -> bool:
    return str(ticker).upper().startswith(PREFIX)


def name_of(ticker: str) -> str:
    return str(ticker)[len(PREFIX):] if is_csv(ticker) else str(ticker)


def data_dir() -> Path:
    from .download import DATA_DIR
    return DATA_DIR / "csv"


def cache_path(name: str) -> Path:
    return data_dir() / "5m" / f"{name}.csv"


def _zone(tz: str):
    """'UTC', 'UTC+2', 'GMT-0500', 'Europe/Berlin' -> something tz_localize accepts."""
    m = re.fullmatch(r"(?:UTC|GMT)\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?", tz.strip(), re.I)
    if m:
        sign = 1 if m.group(1) == "+" else -1
        minutes = sign * (int(m.group(2)) * 60 + int(m.group(3) or 0))
        from datetime import timedelta, timezone
        return timezone(timedelta(minutes=minutes))
    return "UTC" if tz.strip().upper() in ("UTC", "GMT", "Z") else tz.strip()


def parse(path, tz: str = "UTC") -> pd.DataFrame:
    """One CSV -> OHLCV indexed by UTC time."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    first = text.splitlines()[0] if text else ""
    sep = "\t" if "\t" in first else ";" if first.count(";") > first.count(",") else ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    cols = {c: re.sub(r"[<>\s]", "", str(c)).lower() for c in df.columns}
    df = df.rename(columns=cols)
    if "date" in df.columns and "time" in df.columns:                  # MT5 export: separate columns
        stamp = df["date"].astype(str) + " " + df["time"].astype(str)
    else:
        tcol = next((c for c in df.columns if "time" in c or "date" in c), df.columns[0])
        stamp = df[tcol].astype(str)
    need = ["open", "high", "low", "close"]
    if not set(need) <= set(df.columns):
        raise ValueError(f"{Path(path).name}: needs open/high/low/close columns; found {list(df.columns)}")
    stamp = stamp.str.strip()
    # Dukascopy local-time exports end with "GMT+0100": use that offset
    off = stamp.str.extract(r"GMT([+-]\d{2}):?(\d{2})$")
    clean = stamp.str.replace(r"\s*GMT[+-]\d{2}:?\d{2}$", "", regex=True)
    dayfirst = bool(re.match(r"^\d{2}\.\d{2}\.\d{4}", clean.iloc[0])) if len(clean) else False
    when = pd.to_datetime(clean.str.replace(r"^(\d{4})\.(\d{2})\.(\d{2})", r"\1-\2-\3", regex=True),
                          dayfirst=dayfirst, format="mixed")
    if off[0].notna().any():
        hours = off[0].astype(float) + off[1].astype(float).where(off[0].astype(float) >= 0,
                                                                 -off[1].astype(float)) / 60
        when = when - pd.to_timedelta(hours.fillna(0), unit="h")
        idx = pd.DatetimeIndex(when).tz_localize("UTC")
    else:
        idx = pd.DatetimeIndex(when).tz_localize(_zone(tz), ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
    vol = df["volume"] if "volume" in df.columns else df.get("tickvol", pd.Series(0, index=df.index))
    out = pd.DataFrame({"Open": df["open"].astype(float).to_numpy(), "High": df["high"].astype(float).to_numpy(),
                        "Low": df["low"].astype(float).to_numpy(), "Close": df["close"].astype(float).to_numpy(),
                        "Volume": pd.to_numeric(vol, errors="coerce").fillna(0).to_numpy()}, index=idx)
    out = out[out.index.notna()]
    # Dukascopy fills closed-market minutes with flat zero-volume candles: drop those
    flat = (out["High"] == out["Low"]) & (out["Volume"] == 0)
    return out[~flat]


def to_5m(df: pd.DataFrame) -> pd.DataFrame:
    """Combine 1-minute (or any <5m) candles into 5-minute candles, in UTC."""
    step = pd.Series(df.index[1:] - df.index[:-1]).median() if len(df) > 1 else pd.Timedelta("5min")
    if step >= pd.Timedelta("5min"):
        return df
    agg = df.resample("5min", label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    return agg.dropna(subset=["Open", "High", "Low", "Close"])


def import_files(name: str, files, tz: str = "UTC") -> Path:
    """Parse, combine to 5m, merge with earlier imports and save data/csv/5m/<name>.csv."""
    from .download import clean_ticker
    name = name_of(clean_ticker(PREFIX + name))
    paths = []
    for f in files:
        hits = sorted(glob.glob(str(f)))
        paths += [Path(h) for h in hits] if hits else [Path(f)]
    if not paths:
        raise ValueError("No CSV files given.")
    parts = []
    for p in paths:
        if not p.exists():
            raise ValueError(f"File not found: {p}")
        parts.append(parse(p, tz))
    df = pd.concat(parts)
    df = to_5m(df[~df.index.duplicated(keep="last")].sort_index())
    path = cache_path(name)
    if path.exists():
        old = pd.read_csv(path, index_col="Datetime")
        old.index = pd.to_datetime(old.index, utc=True)
        both = pd.concat([old, df])
        df = both[~both.index.duplicated(keep="last")].sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = out.index.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.index.name = "Datetime"
    out.to_csv(path, float_format="%.6f")
    return path


def load(ticker: str, timeframe: str, history: str = None) -> pd.DataFrame:
    """Bars for a CSV ticker (e.g. 'CSV:GER40'), like download.load_bars returns."""
    from .download import HISTORY, RESAMPLE, check_history, clean_spikes, read_intraday_cached, resample_bars
    name = name_of(ticker)
    path = cache_path(name)
    if not path.exists():
        matches = [p for p in path.parent.glob("*.csv") if p.stem.lower() == name.lower()]
        if not matches:
            raise ValueError(f"No imported candles for {name!r}. Import them first: python -m quantlab.importer "
                             f"{name} your_file.csv")
        path = matches[0]
    raw = read_intraday_cached(path)
    if timeframe == "1d":
        df = resample_bars(raw, "1D")
        df.index = df.index.tz_localize(None).normalize()
    else:
        days = HISTORY[check_history(history)]
        if days:
            raw = raw[raw.index >= raw.index[-1] - pd.Timedelta(days=days)]
        raw = clean_spikes(raw)
        df = raw if timeframe == "5m" else resample_bars(raw, RESAMPLE[timeframe])
    bars = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    bars.columns = ["open", "high", "low", "close", "volume"]
    bars.attrs.update(source="csv", adjusted=False)
    return bars


def saved() -> list:
    return [PREFIX + p.stem for p in sorted((data_dir() / "5m").glob("*.csv"))]


def main(argv=None):
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    tz = "UTC"
    if "--tz" in args:
        i = args.index("--tz")
        tz = args[i + 1] if i + 1 < len(args) else "UTC"
        args = args[:i] + args[i + 2:]
    if len(args) < 2:
        print(__doc__)
        return
    try:
        p = import_files(args[0], args[1:], tz)
        df = pd.read_csv(p, index_col="Datetime")
        print(f"  {len(df):,} five-minute candles, {df.index[0]} -> {df.index[-1]} (UTC), saved in {p}")
        print(f"  use it in the app as  {PREFIX}{p.stem}")
    except (ValueError, OSError) as e:
        print(f"  {e}")


if __name__ == "__main__":
    main()
