"""Find ticker symbols by company or fund name (Yahoo Finance search).

    search("apple")  ->  [{"symbol": "AAPL", "name": "Apple Inc.", "type": "Stock",
                           "exchange": "NASDAQ", "us": True, "supported": True}, ...]

QuantLab handles stocks, ETFs and indices. Futures, crypto and currencies are
returned with supported=False so the app can say why they're not offered.
"""
import time

import requests

from .download import HEADERS, TICKER_RE

SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
TYPES = {"EQUITY": "Stock", "ETF": "ETF", "INDEX": "Index", "MUTUALFUND": "Fund",
         "FUTURE": "Future", "CRYPTOCURRENCY": "Crypto", "CURRENCY": "Currency", "OPTION": "Option"}
SUPPORTED = {"EQUITY", "ETF", "INDEX", "MUTUALFUND"}
US_EXCHANGES = {"NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ", "NAS": "NASDAQ",
                "NYQ": "NYSE", "NYS": "NYSE", "PCX": "NYSE Arca", "ASE": "NYSE American",
                "BTS": "Cboe BZX", "BATS": "Cboe BZX", "PNK": "OTC", "OQB": "OTC", "OQX": "OTC",
                "SNP": "S&P index", "WCB": "Cboe index", "DJI": "Dow Jones index", "NIM": "Nasdaq index"}
_cache = {}            # query -> (time, results)
CACHE_SECONDS = 600


def search(query: str, limit: int = 12, timeout: float = 8) -> list:
    """Symbols matching a name or symbol, US stocks/ETFs/indices first."""
    q = " ".join(str(query).split())[:60]
    if len(q) < 1:
        return []
    key = q.lower()
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    resp = requests.get(SEARCH_URL, headers=HEADERS, timeout=timeout,
                        params={"q": q, "quotesCount": 20, "newsCount": 0, "listsCount": 0})
    resp.raise_for_status()
    out = []
    for x in resp.json().get("quotes", []):
        symbol, qtype = str(x.get("symbol", "")).upper(), x.get("quoteType", "")
        if not symbol or not TICKER_RE.match(symbol):
            continue
        exch = x.get("exchange", "")
        out.append({
            "symbol": symbol,
            "name": (x.get("longname") or x.get("shortname") or "").strip(),
            "type": TYPES.get(qtype, qtype.title() or "Other"),
            "exchange": US_EXCHANGES.get(exch, x.get("exchDisp") or exch),
            "us": exch in US_EXCHANGES,
            "supported": qtype in SUPPORTED,
        })
    # supported first, US before foreign; keep Yahoo's relevance order otherwise
    out.sort(key=lambda r: (not r["supported"], not r["us"]))
    out = out[:limit]
    _cache[key] = (time.time(), out)
    return out
