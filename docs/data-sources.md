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
