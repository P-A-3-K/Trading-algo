import numpy as np
import pandas as pd
import pytest

from quantlab import mtf, optimize, strategies
from quantlab.strategy import Param, Strategy, check_no_lookahead


def five_minute(days=("2024-03-08", "2024-03-11", "2024-03-12"), seed=0):
    """Three days of 5m bars, 04:00-20:00 New York, across the March DST change."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) + pd.Timedelta(minutes=240 + 5 * i)
                            for d in days for i in range(192)]).tz_localize("America/New_York")
    c = 100 + np.cumsum(np.random.default_rng(seed).normal(0, 0.1, len(idx)))
    return pd.DataFrame({"open": c, "high": c + 0.05, "low": c - 0.05, "close": c, "volume": 1.0}, index=idx)


def at(bars, s):
    return bars.index.get_loc(pd.Timestamp(s, tz="America/New_York"))


def test_closed_candles_only():
    b = five_minute()
    h4 = mtf.higher(b, "4h")
    candle = b.loc["2024-03-11 04:00":"2024-03-11 07:55"]
    i = at(b, "2024-03-11 09:00")
    assert h4.close.iloc[i] == pytest.approx(candle.close.iloc[-1])           # 04:00-08:00 candle
    assert h4.high.iloc[i] == pytest.approx(candle.high.max())
    assert h4.close.iloc[at(b, "2024-03-11 07:55")] == pytest.approx(candle.close.iloc[-1])  # closes at 08:00
    assert h4.close.iloc[at(b, "2024-03-11 07:50")] == pytest.approx(b.loc["2024-03-08 19:55", "close"])
    assert np.isnan(h4.close.iloc[0])                                          # nothing finished yet
    d = mtf.higher(b, "1d")
    assert d.close.iloc[at(b, "2024-03-11 10:00")] == pytest.approx(b.loc["2024-03-08 19:55", "close"])


def test_forming_candle():
    b = five_minute()
    f = mtf.higher(b, "4h", forming=True)
    i = at(b, "2024-03-11 09:05")
    part = b.loc["2024-03-11 08:00":"2024-03-11 09:05"]
    assert f.open.iloc[i] == pytest.approx(part.open.iloc[0])
    assert f.high.iloc[i] == pytest.approx(part.high.max()) and f.close.iloc[i] == b.close.iloc[i]


def test_never_changes_when_the_future_is_cut_off():
    b = five_minute()
    full = mtf.higher(b, "4h")
    ind = mtf.on_higher(b, "1h", lambda x: x.close.rolling(3).mean())
    forming = mtf.higher(b, "1h", forming=True)
    for cut in range(30, len(b), 41):
        part = b.iloc[:cut]
        assert np.allclose(mtf.higher(part, "4h").fillna(-1), full.iloc[:cut].fillna(-1))
        assert np.allclose(mtf.on_higher(part, "1h", lambda x: x.close.rolling(3).mean()).fillna(-1),
                           ind.iloc[:cut].fillna(-1))
        assert np.allclose(mtf.higher(part, "1h", forming=True), forming.iloc[:cut])


def test_same_or_lower_timeframe_gives_the_bars_themselves():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    daily = pd.DataFrame({k: [1.0, 2.0, 3.0] for k in mtf.COLS}, index=idx)
    assert mtf.higher(daily, "1d").close.tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="Unknown timeframe"):
        mtf.higher(daily, "2h")


def test_mtf_strategies_pass_the_lookahead_check_and_cheating_is_caught():
    b = five_minute(seed=3)
    for name, kw in [("MtfRsi", {"htf": "1h", "period": 5, "htf_period": 5}),
                     ("FairValueGap", {"trend_tf": "1h", "trend_len": 5})]:
        assert check_no_lookahead(strategies.get(name)(**kw), b, checks=30) == []

    class Naive(Strategy):
        name = "naive"
        def compute(self, bars):
            h = bars.close.resample("4h").last().reindex(bars.index, method="ffill")
            self.plot("4h close", h)
            return (h > bars.close).astype(float)
    assert check_no_lookahead(Naive(), b, checks=30)


def test_choice_settings():
    p = Param("4h", choices=("1h", "4h"), label="HTF")
    assert p.is_choice and not p.is_int and p.coerce("htf", "1h") == "1h"
    assert p.describe()["type"] == "choice" and p.describe()["choices"] == ["1h", "4h"]
    with pytest.raises(ValueError, match="one of"):
        p.coerce("htf", "2h")
    fvg = strategies.get("FairValueGap")
    g = optimize.grid(fvg, {"rr": (2, 3, 1), "trend_tf": ["off", "4h", "off"]})
    assert len(g) == 4 and {x["trend_tf"] for x in g} == {"off", "4h"}
    assert optimize.is_varied(["off", "4h"]) and not optimize.is_varied(["4h"])
    assert optimize.is_varied((1, 5, 1)) and not optimize.is_varied((5, 5, 1))
