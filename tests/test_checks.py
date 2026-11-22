import numpy as np
import pandas as pd

from quantlab.checks import (fill_gaps, find_extra, find_missing, find_stale,
                             trading_days)


def prices_on(days):
    return pd.Series(np.arange(1.0, len(days) + 1), index=days)


def test_calendar_skips_weekends_and_holidays():
    days = trading_days("2025-12-24", "2026-01-02")
    assert pd.Timestamp("2025-12-25") not in days      # Christmas
    assert pd.Timestamp("2025-12-27") not in days      # Saturday
    assert pd.Timestamp("2026-01-01") not in days      # New Year
    assert pd.Timestamp("2025-12-26") in days


def test_find_missing_groups_runs():
    days = trading_days("2024-01-02", "2024-01-31")
    kept = days.delete([3, 10, 11, 12])                # one 1-day gap, one 3-day gap
    gaps = find_missing(kept)
    assert gaps["days"].tolist() == [1, 3]
    assert gaps["start"].iloc[1] == days[10]


def test_no_gaps_on_full_calendar():
    assert find_missing(trading_days("2024-01-02", "2024-06-28")).empty


def test_find_extra_flags_weekend_row():
    days = trading_days("2024-01-02", "2024-01-12").append(
        pd.DatetimeIndex(["2024-01-06"])).sort_values()  # a Saturday
    assert list(find_extra(days)) == [pd.Timestamp("2024-01-06")]


def test_find_stale():
    idx = trading_days("2024-01-02", "2024-01-05")
    df = pd.DataFrame({"High": [2, 2, 3, 3], "Low": [1, 2, 1, 1],
                       "Close": [2, 2, 2, 3], "Volume": [10, 10, 0, 10]}, index=idx)
    assert list(find_stale(df)) == [idx[1], idx[2]]


def test_fill_gaps_forward_only_and_limited():
    days = trading_days("2024-01-02", "2024-01-31")
    s = prices_on(days)
    damaged = s.drop(days[[3]].append(days[10:15]))     # 1-day gap, 5-day gap
    filled = fill_gaps(damaged, max_fill=3)
    assert filled[days[3]] == s[days[2]]                # forward, not backward
    assert filled[days[10:13]].tolist() == [s[days[9]]] * 3
    assert filled[days[13:15]].isna().all()             # too long: left empty
    assert len(filled) == len(days)
