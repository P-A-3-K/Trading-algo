"""A starter list of popular US tickers, offered in the app's ticker box.

Any other Yahoo Finance symbol still works: search by name in the ticker box,
or type the symbol, and it downloads.
Download all of these at once with:  python -m quantlab.download --popular
"""

POPULAR = {
    "Broad market ETF": {
        "SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow Jones 30",
        "IWM": "Russell 2000 small caps", "VTI": "Total US stock market",
        "EFA": "Developed markets ex-US", "EEM": "Emerging markets",
    },
    "Bonds & commodities ETF": {
        "TLT": "20+ year Treasury bonds", "IEF": "7-10 year Treasury bonds",
        "LQD": "Investment-grade corporate bonds", "HYG": "High-yield corporate bonds",
        "GLD": "Gold", "SLV": "Silver", "USO": "Crude oil", "VNQ": "US real estate (REITs)",
    },
    "Sector ETF": {
        "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health care",
        "XLY": "Consumer discretionary", "XLP": "Consumer staples", "XLI": "Industrials",
        "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real estate", "XLC": "Communication services",
    },
    "Stock": {
        "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon",
        "GOOGL": "Alphabet (Google)", "META": "Meta Platforms", "TSLA": "Tesla",
        "BRK-B": "Berkshire Hathaway", "JPM": "JPMorgan Chase", "V": "Visa",
        "JNJ": "Johnson & Johnson", "XOM": "Exxon Mobil", "WMT": "Walmart",
        "PG": "Procter & Gamble", "KO": "Coca-Cola", "DIS": "Walt Disney", "NFLX": "Netflix",
        "AMD": "Advanced Micro Devices", "INTC": "Intel", "BA": "Boeing",
    },
    "Index (no dividends)": {
        "^GSPC": "S&P 500 index", "^DJI": "Dow Jones Industrial Average",
        "^IXIC": "Nasdaq Composite",
    },
}


def popular_tickers() -> list:
    return [t for group in POPULAR.values() for t in group]


def describe(ticker: str):
    """(name, group) for a ticker in the list, or (None, None)."""
    for group, names in POPULAR.items():
        if ticker in names:
            return names[ticker], group
    return None, None


def menu(saved: list) -> list:
    """Ticker box entries: every popular ticker plus anything else saved,
    with its name, group, and whether it's already downloaded."""
    saved = set(saved)
    rows = [{"symbol": t, "name": n, "group": g, "saved": t in saved}
            for g, names in POPULAR.items() for t, n in names.items()]
    known = {r["symbol"] for r in rows}
    rows += [{"symbol": t, "name": None, "group": "Your downloads", "saved": True}
             for t in sorted(saved - known)]
    return rows


def find(query: str, saved: list, limit: int = 8) -> list:
    """Popular/saved tickers whose symbol or name matches what was typed.

    Symbol matches first (exact, then starts-with), then name matches.
    """
    q = str(query).strip().lower()
    if not q:
        return []
    rows = menu(saved)
    def score(r):
        sym, name = r["symbol"].lower(), (r["name"] or "").lower()
        if sym == q or sym == "^" + q:
            return 0
        if sym.startswith(q) or sym.lstrip("^").startswith(q):
            return 1
        if any(w.startswith(q) for w in name.replace("(", " ").split()):
            return 2
        if q in name:
            return 3
        return None
    hits = [(score(r), i, r) for i, r in enumerate(rows)]
    return [r for s, i, r in sorted((h for h in hits if h[0] is not None), key=lambda h: (h[0], h[1]))][:limit]
