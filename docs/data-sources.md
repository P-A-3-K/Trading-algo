# Data source decision

**Market:** US stocks & ETFs, daily bars.
**Chosen source:** Yahoo Finance chart endpoint (free, no API key).

## What we compared
| Source | Cost / key | History | Adjusted prices | Verdict |
|---|---|---|---|---|
| Yahoo Finance | Free, no key | Decades (SPY from 1993) | Yes: adjusted close, dividends, splits | **Use it** |
| Stooq | Free, no key | Decades | Yes | Blocked: now requires a browser check, so scripts can't download |
| Alpha Vantage | Free key, 25 calls/day | ~20 years | Adjusted series is paid | Backup if Yahoo breaks |
| Broker APIs (Alpaca, IBKR) | Free with account | Varies | Yes | Later, for paper trading (Phase 4) |
| Paid vendors (Polygon, Norgate) | Paid | Full, incl. delisted stocks | Yes | Only if we need survivorship-bias-free data |

## Why adjusted prices matter
SPY closed at $43.94 on its first day (1993), but its *adjusted* close for that day is $24.05.
The difference is 30+ years of dividends folded back into the price. Backtest on raw close and
you silently throw away the dividends, so buy-and-hold looks worse than it really was.
Splits are worse: a 2-for-1 split looks like a 50% crash in one day.
Careful: Yahoo's `Close` column is already split-adjusted, just not dividend-adjusted.
Only `Adj Close` handles both. (AAPL really traded near $499 before its 2020 4-for-1 split;
Yahoo's `Close` shows about $125.)
**Rule: always compute returns from adjusted close.**

## Terms of use (read this)
Yahoo's endpoint is unofficial: it is not a supported, documented API, it can change or rate-limit
without notice, and Yahoo's terms allow personal, non-commercial use. Fine for learning and research.
Not fine for redistributing the data or running a commercial product. So:
- download once and cache CSVs in `data/` (gitignored), don't re-download every run
- keep requests slow (a pause between tickers)
- if we ever go commercial, switch to a licensed vendor

## Known limitation: survivorship bias
Free sources only cover tickers that still exist. Testing "the S&P 500 today" over 20 years
ignores companies that went bankrupt or were dropped, which makes results look too good.
We will handle this in the Data checks / Validation lessons.

## Intraday data (added later)

| Timeframe | Source | History available for free |
|---|---|---|
| 5m | Yahoo `interval=5m` | last ~60 days |
| 15m, 30m | built from 5m | same as 5m |
| 1h | Yahoo `interval=1h` | last ~2 years |
| 4h | built from 1h (bins 0/4/8/12/16/20 New York time) | same as 1h |

- Includes pre-market and after-hours (`includePrePost=true`), 4:00-20:00 New York.
  Yahoo reports **zero volume** outside regular hours.
- **Not adjusted for dividends** (Yahoo doesn't adjust intraday). Minor over weeks/months.
- Every download is **merged** into `data/<tf>/<TICKER>.csv`, so history grows over time.
- **Bad prints:** thin after-hours trading produces wicks like a 6% drop on a bar whose
  open and close are equal. `clean_spikes()` trims wicks that stick out >2% beyond the body
  and >10x the typical range of the previous 50 bars (only past bars are used).

## Alpaca (years of intraday history, free account)

With Alpaca keys (Data page in the app, or `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`), every
intraday timeframe comes from Alpaca instead of Yahoo:

| | Yahoo (no keys) | Alpaca (free keys) |
|---|---|---|
| 5m-30m history | ~60 days | since 2016 |
| 1h-4h history | ~2 years | since 2016 |
| Exchanges | consolidated | all US exchanges (SIP); IEX only if the plan refuses SIP |
| Adjusted | no | splits and dividends (`adjustment=all`), like the daily data |
| Latest bars | live-ish | 15 minutes delayed (free plan) |

- One file per ticker: `data/alpaca/5m/<TICKER>.csv`. 15m, 30m, 1h and 4h are built from it,
  so all timeframes agree. Each use fetches only the bars since the last download; when a new
  dividend re-adjusts the past, the whole history is downloaded again (~50 requests).
- First download per ticker: about a minute or two (`python -m quantlab.alpaca SPY` shows progress).
- Indices (^GSPC) still come from Yahoo: Alpaca has stocks and ETFs only.
- **History** menu: intraday pages load only the last N months/years. The chart has to draw every
  candle (5m defaults to 6 months), sweeps only compute (5m defaults to 1 year); "All history"
  is there when you want it.
- Endpoint: `GET https://data.alpaca.markets/v2/stocks/bars` with headers `APCA-API-KEY-ID` /
  `APCA-API-SECRET-KEY`, 10,000 bars per page (`next_page_token`).
