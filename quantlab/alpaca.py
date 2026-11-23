"""Alpaca market data: years of 5-minute candles (free account), instead of Yahoo's 60 days.

Setup (once): make a free account at alpaca.markets, create API keys
(Paper trading -> API keys is fine: market data works with either), then either
  * paste them on QuantLab's Data page (http://127.0.0.1:8000/settings), or
  * set environment variables APCA_API_KEY_ID and APCA_API_SECRET_KEY.
The keys are saved in alpaca_keys.json in the project folder, which git ignores:
they stay on your computer and are only ever sent to Alpaca.

Download ahead of time (shows progress; the app also downloads on first use):
    python -m quantlab.alpaca SPY QQQ GLD

How it's stored: data/alpaca/5m/<TICKER>.csv, every 5-minute bar since 2016
(pre-market and after-hours included), adjusted for splits and dividends like the
daily data. 15m, 30m, 1h and 4h candles are built from these 5m bars, so every
timeframe agrees. Each later use only fetches the bars since the last download.

Free plan: full-market (SIP) history, except the latest 15 minutes. If your plan
refuses SIP, QuantLab falls back to the IEX feed (one exchange: fewer trades).
"""
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
KEYS_FILE = ROOT / "alpaca_keys.json"
URL = "https://data.alpaca.markets/v2/stocks/bars"
HISTORY_START = "2016-01-01"          # Alpaca's historical data starts around here
PAGE = 10_000                          # most bars Alpaca returns per request
RECENT_DELAY = timedelta(minutes=16)   # free plan: no SIP data from the last 15 minutes
ADJUSTMENT = "all"                     # splits and dividends, like the daily candles


class AlpacaError(Exception):
    """A problem worth showing to the user as it is (bad keys, no data...)."""


# ---- Keys -------------------------------------------------------------------------------

def load_keys():
    """(key_id, secret) from the environment or alpaca_keys.json, else None."""
    k, s = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if k and s:
        return k.strip(), s.strip()
    try:
        d = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
        if d.get("key_id") and d.get("secret"):
            return d["key_id"].strip(), d["secret"].strip()
    except (OSError, ValueError):
        pass
    return None


def keys_source() -> str:
    """Where the keys come from: 'environment', 'file' or ''."""
    if os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY"):
        return "environment"
    return "file" if load_keys() else ""


def save_keys(key_id: str, secret: str, feed: str = None):
    key_id, secret = (key_id or "").strip(), (secret or "").strip()
    if not key_id or not secret:
        raise AlpacaError("Paste both the API key ID and the secret key.")
    if any(c.isspace() for c in key_id + secret) or len(key_id) > 100 or len(secret) > 200:
        raise AlpacaError("Those don't look like Alpaca keys (no spaces; key ID ~20 characters, secret ~40).")
    data = {"key_id": key_id, "secret": secret}
    if feed:
        data["feed"] = feed
    KEYS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(KEYS_FILE, 0o600)        # only you can read it (no effect on Windows)
    except OSError:
        pass


def delete_keys():
    if KEYS_FILE.exists():
        KEYS_FILE.unlink()


def masked(key_id: str) -> str:
    return key_id[:4] + "…" + key_id[-3:] if len(key_id) > 8 else "…"


def enabled() -> bool:
    return load_keys() is not None


def preferred_feed() -> str:
    try:
        return json.loads(KEYS_FILE.read_text(encoding="utf-8")).get("feed") or "sip"
    except (OSError, ValueError):
        return "sip"


def _remember_feed(feed: str):
    try:
        d = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
        if d.get("feed") != feed:
            d["feed"] = feed
            KEYS_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except (OSError, ValueError):
        pass


# ---- Requests ---------------------------------------------------------------------------

def _get(params: dict, keys, session=None, timeout: float = 30, retries: int = 5) -> dict:
    """One request, with polite waiting when Alpaca says 'too many requests'."""
    headers = {"APCA-API-KEY-ID": keys[0], "APCA-API-SECRET-KEY": keys[1]}
    http = session or requests
    for attempt in range(retries):
        resp = http.get(URL, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 429 and attempt < retries - 1:       # rate limit: wait, retry
            time.sleep(float(resp.headers.get("Retry-After") or 3 * (attempt + 1)))
            continue
        if resp.status_code in (401, 403):
            try:
                msg = resp.json().get("message", "")
            except ValueError:
                msg = resp.text[:200]
            raise PermissionError(msg or f"HTTP {resp.status_code}")
        if resp.status_code == 422:
            try:
                msg = resp.json().get("message", "")
            except ValueError:
                msg = resp.text[:200]
            raise AlpacaError(f"Alpaca refused the request: {msg}")
        resp.raise_for_status()
        return resp.json()
    raise AlpacaError("Alpaca kept saying 'too many requests'. Wait a minute and try again.")


def parse_bars(rows: list) -> pd.DataFrame:
    """Alpaca bars [{t, o, h, l, c, v, ...}] -> OHLCV indexed by New York time (bar start)."""
    from .download import EXCHANGE_TZ
    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"],
                            index=pd.DatetimeIndex([], tz=EXCHANGE_TZ, name="Datetime"))
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["t"], utc=True, format="ISO8601").dt.tz_convert(EXCHANGE_TZ)
    out = pd.DataFrame({"Open": df["o"].astype(float), "High": df["h"].astype(float),
                        "Low": df["l"].astype(float), "Close": df["c"].astype(float),
                        "Volume": df["v"].astype(float)})
    out.index = pd.DatetimeIndex(idx, name="Datetime")
    return out[~out.index.duplicated(keep="last")].sort_index()


def fetch_bars(ticker: str, start, end=None, timeframe: str = "5Min", feed: str = None,
               keys=None, progress=None) -> pd.DataFrame:
    """Every bar of `ticker` from start to end (default: 16 minutes ago), page by page.

    progress: function(bars_so_far, last_time) called after each page.
    Falls back from the SIP feed to IEX if the account isn't allowed SIP.
    """
    from .download import clean_ticker
    keys = keys or load_keys()
    if not keys:
        raise AlpacaError("No Alpaca keys yet: add them on the Data page (/settings).")
    symbol = clean_ticker(ticker)
    if symbol.startswith("^"):
        raise AlpacaError(f"{symbol} is an index; Alpaca only has stocks and ETFs.")
    now = datetime.now(timezone.utc)
    end = pd.Timestamp(end or now - RECENT_DELAY)
    end = min(end.tz_localize("UTC") if end.tz is None else end, pd.Timestamp(now - RECENT_DELAY))
    start = pd.Timestamp(start)
    start = start.tz_localize("UTC") if start.tz is None else start
    if start >= end:
        return parse_bars([])
    feed = feed or preferred_feed()
    params = {"symbols": symbol, "timeframe": timeframe, "start": start.isoformat(),
              "end": end.isoformat(), "limit": PAGE, "adjustment": ADJUSTMENT,
              "feed": feed, "sort": "asc"}
    rows, token = [], None
    with requests.Session() as session:
        while True:
            if token:
                params["page_token"] = token
            try:
                data = _get(params, keys, session)
            except PermissionError as e:
                if feed == "sip" and not rows:                   # plan without SIP: use IEX
                    feed = params["feed"] = "iex"
                    _remember_feed("iex")
                    continue
                raise AlpacaError(f"Alpaca refused the keys ({e}). Check them on the Data page.")
            page = (data.get("bars") or {}).get(symbol) or []
            rows.extend(page)
            if progress and page:
                progress(len(rows), page[-1]["t"])
            token = data.get("next_page_token")
            if not token:
                break
    return parse_bars(rows)


def test_keys(key_id: str, secret: str):
    """Try the keys on a few days of SPY. Returns (message, feed); raises AlpacaError.

    feed: "sip" (all US exchanges) when the plan allows it, else "iex" (one exchange).
    """
    end = datetime.now(timezone.utc) - timedelta(days=1)
    params = {"symbols": "SPY", "timeframe": "1Hour", "start": (end - timedelta(days=7)).isoformat(),
              "end": end.isoformat(), "limit": 100, "adjustment": ADJUSTMENT, "sort": "asc"}
    last = None
    for feed in ("sip", "iex"):
        try:
            data = _get({**params, "feed": feed}, (key_id.strip(), secret.strip()), retries=2)
        except PermissionError as e:
            last = e
            continue
        except requests.RequestException as e:
            raise AlpacaError(f"Couldn't reach Alpaca ({type(e).__name__}). Check your internet connection.")
        n = len((data.get("bars") or {}).get("SPY") or [])
        what = "all US exchanges (SIP)" if feed == "sip" else "the IEX exchange only (your plan doesn't include SIP)"
        return f"Connected: received {n} test candles for SPY from {what}.", feed
    raise AlpacaError(f"Alpaca refused these keys ({last}). Copy both again from the Alpaca "
                      "dashboard (API Keys), and check there are no extra spaces.")


# ---- Saved history ----------------------------------------------------------------------

def cache_path(ticker: str) -> Path:
    from .download import DATA_DIR, clean_ticker
    return DATA_DIR / "alpaca" / "5m" / f"{clean_ticker(ticker)}.csv"


def saved() -> list:
    """Tickers already downloaded from Alpaca: first/last bar and file size."""
    from .download import DATA_DIR
    out = []
    for p in sorted((DATA_DIR / "alpaca" / "5m").glob("*.csv")):
        try:
            with open(p, "rb") as f:
                f.readline()
                first = f.readline().decode()[:10]
                f.seek(max(0, p.stat().st_size - 400))
                last = f.read().decode(errors="ignore").strip().splitlines()[-1][:10]
        except (OSError, IndexError):
            continue
        mb = p.stat().st_size / 1e6
        out.append({"ticker": p.stem, "first": first, "last": last,
                    "size": f"{mb:.1f} MB" if mb >= 1 else f"{mb * 1000:.0f} KB"})
    return out


def _write(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = out.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    out.index.name = "Datetime"
    tmp = path.with_suffix(".tmp")
    out.to_csv(tmp, float_format="%.6f")
    tmp.replace(path)                       # never leave a half-written file behind


def download(ticker: str, refresh: bool = False, progress=None, max_age: float = 15 * 60) -> Path:
    """Bring data/alpaca/5m/<TICKER>.csv up to date and return its path.

    First time: everything since 2016 (about 50 requests, a minute or two).
    After that: only the bars since the last download. Adjusted prices shift
    after every dividend, so if the overlap with the saved bars doesn't match,
    the whole history is downloaded again.
    """
    from .download import merge_bars, read_intraday
    path = cache_path(ticker)
    if path.exists() and not refresh and time.time() - path.stat().st_mtime < max_age:
        return path
    try:
        if not path.exists():
            df = fetch_bars(ticker, HISTORY_START, progress=progress)
            if df.empty:
                raise AlpacaError(f"Alpaca has no 5-minute data for {ticker!r}. Check the symbol "
                                  "(Alpaca has US stocks and ETFs only).")
            _write(path, df)
            return path
        old = read_intraday(path)
        new = fetch_bars(ticker, old.index[-1] - pd.Timedelta(days=3), progress=progress)
        overlap = new.index.intersection(old.index)
        if len(overlap):
            drift = (new.loc[overlap, "Close"] / old.loc[overlap, "Close"] - 1).abs().max()
            if drift > 0.001:                   # a dividend or split re-adjusted the past
                df = fetch_bars(ticker, HISTORY_START, progress=progress)
                _write(path, df)
                return path
        _write(path, merge_bars(old, new))
    except (requests.RequestException, AlpacaError) as e:
        if path.exists():                      # offline / Alpaca down: use what we have
            print(f"  note: couldn't update {ticker} from Alpaca ({e}); using the saved data")
            os.utime(path)                      # don't retry on every page load
            return path
        if isinstance(e, requests.RequestException):
            raise AlpacaError(f"Couldn't reach Alpaca ({type(e).__name__}).")
        raise
    return path


if __name__ == "__main__":
    import sys
    tickers = [a for a in sys.argv[1:] if not a.startswith("--")] or ["SPY"]
    if not enabled():
        raise SystemExit("No Alpaca keys yet. Add them on the Data page of the app, or set "
                         "APCA_API_KEY_ID and APCA_API_SECRET_KEY.")
    for t in tickers:
        last = [0.0]

        def show(n, ts, t=t):
            if time.time() - last[0] > 1:
                print(f"\r  {t}: {n:,} bars, up to {ts[:10]}", end="", flush=True)
                last[0] = time.time()
        try:
            p = download(t, refresh="--refresh" in sys.argv, progress=show)
            print(f"\r  {t}: {len(pd.read_csv(p)):,} bars saved in {p.relative_to(ROOT)}" + " " * 10)
        except AlpacaError as e:
            print(f"\r  {t}: {e}")
