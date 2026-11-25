"""MT5 connector, tested against a fake MetaTrader5 module (the real one is Windows-only)."""
import sys
import time
import types
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from quantlab import costs as C
from quantlab import download as dl
from quantlab import mt5 as M


class FakeMT5(types.ModuleType):
    TIMEFRAME_M5, TIMEFRAME_D1 = 5, 1440

    def __init__(self, server_minus_utc_h=2.0):
        super().__init__("MetaTrader5")
        self.offset = server_minus_utc_h
        self.calls = []

    def initialize(self):
        return True

    def last_error(self):
        return (0, "ok")

    def account_info(self):
        return types.SimpleNamespace(server="FTMO-Demo", company="FTMO", currency="USD", login=1)

    def symbol_select(self, sym, on):
        return sym in ("US100.cash", "XAUUSD")

    def symbol_info(self, sym):
        return types.SimpleNamespace(point=0.01, digits=2, trade_contract_size=1.0, description="US Tech 100",
                                     currency_profit="USD", volume_min=0.01, volume_step=0.01, spread=150,
                                     swap_mode=1, swap_long=-200.0, swap_short=50.0, bid=20000.0)

    def symbol_info_tick(self, sym):
        return types.SimpleNamespace(time=int(time.time() + self.offset * 3600), bid=20000.0)

    def copy_rates_from_pos(self, sym, tf, start, count):
        return self.copy_rates_range(sym, tf, datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 5, tzinfo=timezone.utc))

    def symbols_get(self):
        return [types.SimpleNamespace(name="US100.cash", description="US Tech 100")]

    def copy_rates_range(self, sym, tf, a, b):
        self.calls.append((tf, a, b))
        step = 300 if tf == self.TIMEFRAME_M5 else 86400
        # server-clock bars for 2 days in Jan 2024 only
        start = int(datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp())
        t = np.arange(start, start + 2 * 86400, step)
        t = t[(t >= a.timestamp()) & (t < b.timestamp())]
        if not len(t):
            return None
        px = 20000 + np.arange(len(t), dtype=float)
        rec = np.zeros(len(t), dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
                                       ("close", "f8"), ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")])
        rec["time"], rec["open"], rec["high"], rec["low"], rec["close"] = t, px, px + 2, px - 2, px + 1
        rec["tick_volume"], rec["spread"] = 10, 150
        return rec


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    f = FakeMT5()
    monkeypatch.setitem(sys.modules, "MetaTrader5", f)
    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    dl._READ_CACHE.clear()
    return f


def test_ticker_names():
    assert dl.clean_ticker("mt5:US100.cash") == "MT5:US100.cash"         # case of the symbol kept
    assert M.is_mt5("MT5:XAUUSD") and M.symbol_of("MT5:XAUUSD") == "XAUUSD"
    for bad in ("MT5:../x", "MT5:", "MT5:a b"):
        with pytest.raises(ValueError):
            dl.clean_ticker(bad)


def test_server_time_to_new_york(fake):
    # server = UTC+2 now; New York is UTC-5 in January (UTC-4 in summer)
    ny_now = pd.Timestamp.now(tz="America/New_York").utcoffset().total_seconds() / 3600
    assert M.ny_offset_hours(fake, "US100.cash") == pytest.approx(2.0 - ny_now)
    rates = fake.copy_rates_range("US100.cash", fake.TIMEFRAME_M5, datetime(2024, 1, 1, tzinfo=timezone.utc),
                                  datetime(2024, 1, 3, tzinfo=timezone.utc))
    df = M.to_frame(rates, 0.01, 7.0, daily=False)
    assert str(df.index[0]) == "2024-01-01 17:00:00-05:00"           # server midnight = NY 17:00
    assert df["Spread"].iloc[0] == pytest.approx(1.5)                 # 150 points x 0.01


def test_swap_conversion():
    info = types.SimpleNamespace(swap_mode=1, swap_long=-200.0, swap_short=50.0, point=0.01, trade_contract_size=1.0)
    lng, sht = M.swap_bps_per_night(info, 20000.0)
    assert lng == pytest.approx(1.0) and sht == pytest.approx(-0.25)  # -2.00 per night on 20,000 = 1 bp cost
    info.swap_mode = 5; info.swap_long = -7.2                          # interest: -7.2% a year
    assert M.swap_bps_per_night(info, 20000.0)[0] == pytest.approx(7.2 / 360 * 100)


def test_download_load_and_costs(fake):
    p = M.download("US100.cash", "5m")
    assert p.exists() and M.specs_path("US100.cash").exists()
    bars = dl.load_bars("MT5:US100.cash", "5m")
    assert {"spread"} <= set(bars.columns) and bars.attrs["source"] == "mt5"
    assert bars.attrs["swap_long_bps"] == pytest.approx(1.0)
    h1 = dl.load_bars("MT5:US100.cash", "1h")
    assert len(h1) < len(bars) and h1["spread"].iloc[0] == pytest.approx(1.5)
    daily = dl.load_bars("MT5:US100.cash", "1d")
    assert daily.index.tz is None and len(daily) == 2
    assert "MT5:US100.cash" in dl.cached_tickers()
    # Real spread used: 1.5 / 2 on ~20,000 = 0.375 bp half-spread instead of the default 1 bp
    prep = C.CostModel(slippage_range=0, impact=0).prepare(bars)
    assert prep.linear[0] == pytest.approx(0.75 / bars["close"].iloc[0])
    # Unknown symbol and no MT5 package: clear messages
    with pytest.raises(M.MT5Error, match="doesn't know"):
        M.download("NOPE", "5m")


def test_swaps_charged_per_night():
    idx = pd.DatetimeIndex(["2024-01-04 15:55", "2024-01-05 15:55", "2024-01-08 15:55"]).tz_localize("America/New_York")
    assert list(C.nights_between(idx)) == [0.0, 1.0, 3.0]                # Thu->Fri: 1, Fri->Mon: 3
    daily = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08"])
    assert list(C.nights_between(daily)) == [0.0, 1.0, 3.0]
    from quantlab.backtest import run_backtest
    bars = pd.DataFrame({"close": [100.0, 100.0, 100.0]}, index=daily)
    prep = C.CostModel(spread_bps=0, extended_spread_bps=0, slippage_range=0, impact=0,
                       swap_long_bps=2.0).prepare(bars)
    res = run_backtest(bars["close"], pd.Series([1.0, 1.0, 1.0], index=daily), costs=prep)
    # long from day 1's close: 1 night into day 2 (2 bp), 3 nights into day 3 (6 bp)
    assert res.returns.round(8).tolist() == [0.0, -0.0002, -0.0006]


def test_no_package_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "MetaTrader5", None)
    with pytest.raises(M.MT5Error, match="pip install MetaTrader5"):
        M.connect()


def test_finds_branded_terminal_and_remembers_it(tmp_path, monkeypatch):
    f = FakeMT5()
    term = tmp_path / "FTMO MetaTrader 5" / "terminal64.exe"
    term.parent.mkdir(); term.write_text("x", encoding="utf-8")
    attempts = []
    def init(path=None):
        attempts.append(path)
        return path == str(term)
    f.initialize = init
    monkeypatch.setitem(sys.modules, "MetaTrader5", f)
    monkeypatch.setattr(dl, "DATA_DIR", tmp_path / "data")
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    mt5, info = M.connect()
    assert attempts[0] is None and attempts[-1] == str(term) and info["company"] == "FTMO"
    assert M.terminal_path_file().read_text(encoding="utf-8") == str(term)
    f.initialize = lambda path=None: False
    with pytest.raises(M.MT5Error, match="--path"):
        M.connect()


def test_specs_price_fallback_and_typical_spread(fake, monkeypatch):
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    fake.symbol_info_tick = lambda sym: types.SimpleNamespace(time=0, bid=0.0)      # no tick yet
    s = M.specs("US100.cash", fake)
    assert s["price"] == 20000.0                                  # from symbol_info.bid
    assert s["swap_long_bps"] == pytest.approx(1.0)               # -200 points x 0.01 on 20,000
    assert s["spread_all_bps"] == pytest.approx(1.5 / 20000 * 10000, rel=0.05)


def test_full_redownload(fake):
    M.download("US100.cash", "5m")
    p = M.cache_path("US100.cash", "5m")
    p.write_text("Datetime,Open,High,Low,Close,Volume,Spread\n2024-01-02T00:00:00Z,1,1,1,1,0,0\n", encoding="utf-8")
    M.main(["download", "US100.cash", "--full"])
    assert len(M.read(p, daily=False)) > 100                          # fetched again from the start
