"""The automatic trader against a simulated broker (MT5 only exists on Windows)."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from quantlab import live as L


class FakeBroker:
    """Prices from a 1-minute path; fills stop orders and stop-losses when the price crosses them."""

    def __init__(self, path: pd.Series, mode="demo", balance=100_000.0):
        self.path, self.mode = path, mode
        self.now = path.index[0]
        self.cash, self.orders_, self.positions_, self.sent, self.closed = balance, [], [], [], []
        self.next = 1

    def set_time(self, t):
        """Advance to t, filling orders / stops on every minute passed."""
        for ts, px in self.path[(self.path.index > self.now) & (self.path.index <= t)].items():
            for o in list(self.orders_):
                if (o["side"] > 0 and px >= o["price"]) or (o["side"] < 0 and px <= o["price"]):
                    self.orders_.remove(o)
                    self.positions_.append({**o, "price": o["price"], "ticket": self.next}); self.next += 1
            for p in list(self.positions_):
                if (p["side"] > 0 and px <= p["sl"]) or (p["side"] < 0 and px >= p["sl"]):
                    self._close(p, p["sl"])
        self.now = t

    def px(self):
        return float(self.path[self.path.index <= self.now].iloc[-1])

    def _close(self, p, price):
        self.positions_.remove(p)
        self.cash += (price - p["price"]) * p["side"] * p["volume"]
        self.closed.append((p, price))
        return price

    def account(self):
        return {"server": "Fake", "mode": self.mode, "balance": self.cash, "equity": self.equity(), "currency": "USD"}

    def equity(self):
        return self.cash + sum((self.px() - p["price"]) * p["side"] * p["volume"] for p in self.positions_)

    def spec(self, symbol):
        return {"tick_value": 1.0, "tick_size": 1.0, "volume_min": 0.01, "volume_max": 1000, "volume_step": 0.01}

    def quote(self, symbol):
        return self.px(), self.px()

    def bars_m1(self, symbol, start, end):
        s = self.path[(self.path.index >= start) & (self.path.index < end) & (self.path.index < self.now)]
        return pd.DataFrame({"high": s + 0.5, "low": s - 0.5}, index=s.index) if len(s) else None

    def send(self, symbol, kind, side, volume, price, sl, tp, magic, comment):
        self.sent.append((kind, side, volume, price, sl))
        o = {"side": side, "volume": volume, "price": price, "sl": sl, "magic": magic, "ticket": self.next}
        self.next += 1
        if kind == "market":
            self.positions_.append(o)
        else:
            self.orders_.append(o)
        return o["ticket"]

    def positions(self, symbol, magic):
        return [p for p in self.positions_ if p["magic"] == magic]

    def orders(self, symbol, magic):
        return [o for o in self.orders_ if o["magic"] == magic]

    def cancel(self, ticket):
        self.orders_ = [o for o in self.orders_ if o["ticket"] != ticket]

    def close(self, pos):
        p = next(x for x in self.positions_ if x["ticket"] == pos["ticket"])
        return self._close(p, self.px())


def us_day(prices_after_range, day="2024-03-05"):
    """1-minute prices, New York 9:30-16:00: flat at 100 during the range, then the given path."""
    idx = pd.date_range(f"{day} 09:30", f"{day} 15:59", freq="1min", tz="America/New_York").tz_convert("UTC")
    px = [100.0] * 30 + list(prices_after_range)
    px += [px[-1]] * (len(idx) - len(px))
    return pd.Series(px[:len(idx)], index=idx)


def run_day(path, tmp_path, **cfg_kw):
    cfg = L.LiveConfig(legs=[L.LegConfig("US", "US100.cash", settings={"session": "US", "rr": 0}, weight=0.5)], **cfg_kw)
    broker = FakeBroker(path)
    trader = L.Trader(cfg, broker, folder=tmp_path, echo=False)
    trader.check_account()
    for t in pd.date_range(path.index[0], path.index[-1], freq="1min"):
        broker.set_time(t)
        trader.step(t.to_pydatetime())
    return trader, broker, pd.read_csv(tmp_path / "journal.csv")


def test_breakout_fills_one_side_cancels_other_and_closes_at_the_end(tmp_path):
    path = us_day([100.2, 100.8, 101.5, 102.0, 102.5])                    # breaks up from the 99.5-100.5 range
    trader, broker, j = run_day(path, tmp_path)
    kinds = [s[0] for s in broker.sent]
    assert kinds == ["stop", "stop"]                                       # buy stop and sell stop
    buy = [s for s in broker.sent if s[1] > 0][0]
    assert buy[3] == pytest.approx(100.5) and buy[4] == pytest.approx(99.5)   # at the high, stop at the low
    events = list(j["event"])
    assert "filled" in events and events.count("cancelled") >= 1 and "closed" in events
    assert not broker.orders_ and not broker.positions_                  # flat at the end
    assert broker.cash > 100_000                                          # it rose after the breakout


def test_one_trade_per_day_even_if_stopped_between_checks(tmp_path):
    # breaks up, falls back through the stop within the same minute steps, then breaks down
    path = us_day([101.0, 99.0, 98.0, 97.0, 96.0])
    trader, broker, j = run_day(path, tmp_path)
    assert len(broker.closed) == 1                                        # the sell stop was cancelled in time
    assert list(j["event"]).count("filled") + list(j["event"]).count("trade closed") >= 1


def test_sizing_and_guards(tmp_path):
    spec = {"tick_value": 1.0, "tick_size": 1.0, "volume_min": 0.01, "volume_max": 1000, "volume_step": 0.01}
    # 1% risk on a 1% stop -> 100% x weight 0.5 x multiplier 2 = 100% of 100k at price 100 -> 1000 units
    assert L.lots_for(100_000, 0.5, 2.0, 1.0, 100.0, 99.0, spec) == pytest.approx(1000)
    assert L.lots_for(100_000, 0.5, 1.0, 1.0, 100.0, 99.5, spec) == pytest.approx(500)   # capped at 100%
    assert L.lots_for(100, 0.5, 1.0, 1.0, 100.0, 99.0, {**spec, "volume_min": 5}) == 0.0
    # a real-money account is refused
    cfg = L.LiveConfig(legs=[L.LegConfig("US", "X", settings={"session": "US"})])
    with pytest.raises(SystemExit, match="REAL"):
        L.Trader(cfg, FakeBroker(us_day([100]), mode="real"), folder=tmp_path, echo=False).check_account()
    # daily guard: a 5% drop pauses and flattens
    path = us_day([101.0] + [101.0 - i * 0.5 for i in range(1, 30)])
    trader, broker, j = run_day(path, tmp_path / "g", multiplier=10.0, daily_loss_guard=0.01)
    assert "guard" in list(j["event"]) and not broker.positions_


def test_dry_run_sends_nothing(tmp_path):
    cfg = L.LiveConfig(legs=[L.LegConfig("US", "US100.cash", settings={"session": "US", "rr": 0})])
    path = us_day([101.0, 102.0])
    broker = FakeBroker(path)
    trader = L.Trader(cfg, broker, dry=True, folder=tmp_path, echo=False)
    trader.check_account()
    for t in pd.date_range(path.index[0], path.index[-1], freq="1min"):
        broker.set_time(t); trader.step(t.to_pydatetime())
    j = pd.read_csv(tmp_path / "journal.csv")
    assert broker.sent == [] and any(str(e).startswith("would") for e in j["event"])


def test_restart_does_not_trade_twice(tmp_path):
    path = us_day([101.0, 102.0])
    run_day(path, tmp_path)
    trader, broker, j = run_day(path, tmp_path)                           # same day, new process
    assert broker.sent == []


def test_init_writes_a_config(tmp_path, monkeypatch):
    from quantlab import download as dl
    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    L.main(["init"])
    cfg = L.LiveConfig.load(tmp_path / "live" / "live.json")
    assert [l.symbol for l in cfg.legs] == ["US100.cash", "GER40.cash"]
