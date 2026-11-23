import json
import threading
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import pytest

from quantlab import alpaca, app
from quantlab import download as dl


def bars_json(start="2024-03-04 09:30", n=10, price=100.0, freq="5min"):
    idx = pd.date_range(start, periods=n, freq=freq, tz="America/New_York").tz_convert("UTC")
    return [{"t": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "o": price + i, "h": price + i + 1, "l": price + i - 1,
             "c": price + i + 0.5, "v": 1000 + i, "n": 5, "vw": price + i} for i, t in enumerate(idx)]


class FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._p, self.headers, self.text = status, payload, {}, json.dumps(payload)

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    monkeypatch.setattr(alpaca, "KEYS_FILE", tmp_path / "alpaca_keys.json")
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    dl._READ_CACHE.clear()
    return tmp_path


def test_keys_file_and_environment(tmp_env, monkeypatch):
    assert alpaca.load_keys() is None and not alpaca.enabled()
    alpaca.save_keys(" PKTESTKEY123 ", "secret/abc ")
    assert alpaca.load_keys() == ("PKTESTKEY123", "secret/abc") and alpaca.keys_source() == "file"
    assert alpaca.masked("PKTESTKEY123") == "PKTE…123"
    with pytest.raises(alpaca.AlpacaError):
        alpaca.save_keys("has space", "x")
    monkeypatch.setenv("APCA_API_KEY_ID", "ENVKEY"); monkeypatch.setenv("APCA_API_SECRET_KEY", "ENVSECRET")
    assert alpaca.load_keys() == ("ENVKEY", "ENVSECRET") and alpaca.keys_source() == "environment"
    alpaca.delete_keys()
    assert not alpaca.KEYS_FILE.exists()


def test_keys_file_is_git_ignored():
    from pathlib import Path
    assert "alpaca_keys.json" in (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8").split()


def test_fetch_pages_headers_and_iex_fallback(tmp_env, monkeypatch):
    alpaca.save_keys("KEYID", "SECRET")
    calls = []

    def fake_get(self, url, params=None, headers=None, timeout=None):
        calls.append((dict(params), dict(headers)))
        if params["feed"] == "sip":
            return FakeResp(403, {"code": 40310000, "message": "subscription does not permit querying recent SIP data"})
        if not params.get("page_token"):
            return FakeResp(200, {"bars": {"SPY": bars_json(n=3)}, "next_page_token": "abc"})
        return FakeResp(200, {"bars": {"SPY": bars_json("2024-03-04 09:45", n=2)}, "next_page_token": None})
    monkeypatch.setattr("requests.Session.get", fake_get)
    df = alpaca.fetch_bars("spy", "2024-03-01", "2024-03-05")
    assert len(df) == 5
    assert str(df.index[0]) == "2024-03-04 09:30:00-05:00"
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert calls[0][1] == {"APCA-API-KEY-ID": "KEYID", "APCA-API-SECRET-KEY": "SECRET"}
    assert calls[0][0]["symbols"] == "SPY" and calls[0][0]["timeframe"] == "5Min" and calls[0][0]["adjustment"] == "all"
    assert [c[0]["feed"] for c in calls] == ["sip", "iex", "iex"] and calls[2][0]["page_token"] == "abc"
    assert alpaca.preferred_feed() == "iex"                   # remembered for next time


def test_bad_keys_message(tmp_env, monkeypatch):
    alpaca.save_keys("KEYID", "SECRET")
    monkeypatch.setattr("requests.Session.get", lambda self, *a, **k: FakeResp(401, {"message": "forbidden."}))
    with pytest.raises(alpaca.AlpacaError, match="refused the keys"):
        alpaca.fetch_bars("SPY", "2024-03-01", "2024-03-05")
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(403, {"message": "forbidden."}))
    with pytest.raises(alpaca.AlpacaError, match="refused these keys"):
        alpaca.test_keys("KEYID", "SECRET")


def test_download_first_then_incremental_then_readjusted(tmp_env, monkeypatch):
    alpaca.save_keys("K", "S")
    served = {"price": 100.0, "starts": []}

    def fake_fetch(ticker, start, end=None, **kw):
        served["starts"].append(str(pd.Timestamp(start).date()))
        if str(pd.Timestamp(start).date()) == alpaca.HISTORY_START:
            return alpaca.parse_bars(bars_json("2024-03-04 09:30", n=20, price=served["price"]))
        return alpaca.parse_bars(bars_json("2024-03-04 10:55", n=10, price=served["price"] + 17))
    monkeypatch.setattr(alpaca, "fetch_bars", fake_fetch)
    p = alpaca.download("SPY")
    assert len(dl.read_intraday(p)) == 20
    p = alpaca.download("SPY", refresh=True)                 # 3 overlapping bars match -> just append
    assert len(dl.read_intraday(p)) == 27 and served["starts"][-1] != alpaca.HISTORY_START
    served["price"] = 90.0                                    # a dividend re-adjusted the past
    p = alpaca.download("SPY", refresh=True)
    assert served["starts"][-1] == alpaca.HISTORY_START
    assert dl.read_intraday(p)["Open"].iloc[0] == 90.0


def test_intraday_uses_alpaca_and_builds_timeframes(tmp_env, monkeypatch):
    rows = []
    for day in pd.bdate_range("2024-01-02", periods=30):
        rows += bars_json(f"{day.date()} 04:00", n=192, price=100 + day.day)
    monkeypatch.setattr(alpaca, "fetch_bars", lambda *a, **k: alpaca.parse_bars(rows))
    assert dl.intraday_source("SPY") == "yahoo"
    alpaca.save_keys("K", "S")
    assert dl.intraday_source("SPY") == "alpaca" and dl.intraday_source("^GSPC") == "yahoo"
    b5 = dl.load_bars("SPY", "5m")
    assert len(b5) == 30 * 192 and b5.attrs["source"] == "alpaca" and b5.attrs["adjusted"]
    b1h = dl.load_bars("SPY", "1h")
    assert len(b1h) == 30 * 16 and b1h["volume"].sum() == pytest.approx(b5["volume"].sum())
    b4h = dl.load_bars("SPY", "4h")
    assert list(b4h.index[:4].hour) == [4, 8, 12, 16]
    short = dl.load_bars("SPY", "5m", "60d")
    assert short.index[0] >= b5.index[-1] - pd.Timedelta(days=60)
    with pytest.raises(ValueError, match="Unknown history"):
        dl.load_bars("SPY", "5m", "3w")


@pytest.fixture()
def server(tmp_env):
    srv = app.make_server(port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown(); srv.server_close()


def post(url, form):
    req = urllib.request.Request(url, data=urllib.parse.urlencode(form).encode(), method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_data_page_saves_tested_keys(server, monkeypatch):
    with urllib.request.urlopen(server + "/settings") as r:
        body = r.read().decode()
    assert '"enabled": false' in body and "Generate New Keys" in body
    monkeypatch.setattr(alpaca, "test_keys", lambda k, s: ("Connected: 7 candles.", "sip"))
    status, body = post(server + "/settings", {"action": "save", "key_id": "PKABCDEFGH123", "secret": "sss"})
    assert status == 200 and "Connected: 7 candles." in body and '"enabled": true' in body
    assert "sss" not in body and "PKABCDEFGH123" not in body          # the page never shows the keys
    assert alpaca.load_keys() == ("PKABCDEFGH123", "sss")
    def refuse(k, s):
        raise alpaca.AlpacaError("Alpaca refused these keys (forbidden).")
    monkeypatch.setattr(alpaca, "test_keys", refuse)
    status, body = post(server + "/settings", {"action": "save", "key_id": "PKOTHER", "secret": "x"})
    assert status == 400 and "refused" in body
    assert alpaca.load_keys() == ("PKABCDEFGH123", "sss")             # bad keys aren't saved
    status, body = post(server + "/settings", {"action": "delete"})
    assert status == 200 and alpaca.load_keys() is None
