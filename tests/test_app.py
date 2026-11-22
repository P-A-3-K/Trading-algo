import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

from quantlab import app
from quantlab import download as dl
from quantlab.data import synthetic_prices


@pytest.fixture()
def server(tmp_path, monkeypatch):
    """A real server on a free port, reading a synthetic TEST ticker (no network)."""
    close = synthetic_prices(n_days=400, seed=7)
    df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                       "Close": close, "Adj Close": close, "Volume": 1000},
                      index=close.index.rename("Date"))
    df.to_csv(tmp_path / "TEST.csv")
    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dl, "fetch", lambda t: (_ for _ in ()).throw(RuntimeError("offline")))
    srv = app.make_server(port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown(); srv.server_close()


def get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_page_with_strategy(server):
    status, body = get(server + "/?ticker=test&strategy=SmaCrossover&fast=10&slow=30")
    assert status == 200 and "SmaCrossover(fast=10, slow=30)" in body
    assert '"equity":{' in body and '"stats":[' in body


def test_plain_candles(server):
    status, body = get(server + "/api/backtest?ticker=TEST&strategy=none")
    data = json.loads(body)
    assert status == 200 and data["strategy"] == "" and data["equity"] is None
    assert len(data["bars"]) == 400


def test_api_lists(server):
    assert json.loads(get(server + "/api/tickers")[1]) == {"tickers": ["TEST"]}
    ids = [s["id"] for s in json.loads(get(server + "/api/strategies")[1])["strategies"]]
    assert ids[:3] == ["BuyAndHold", "SmaCrossover", "MeanReversion"]
    assert "Breakout" in ids                  # discovered from strategies/


@pytest.mark.parametrize("query,message", [
    ("ticker=../../etc/passwd", "Not a valid ticker"),
    ("ticker=TEST&strategy=Nope", "Unknown strategy"),
    ("ticker=TEST&strategy=SmaCrossover&fast=1", "at least 2"),
])
def test_bad_input_is_a_400_with_a_message(server, query, message):
    status, body = get(server + "/?" + query)
    assert status == 400 and message in body
    status, body = get(server + "/api/backtest?" + query)
    assert status == 400 and message in json.loads(body)["error"]


def test_download_failure_is_reported(server):
    status, body = get(server + "/?ticker=NEWONE")
    assert status == 500 and "offline" in body


def test_unknown_path(server):
    assert get(server + "/nope")[0] == 404


def test_ticker_validation():
    assert dl.clean_ticker(" brk-b ") == "BRK-B"
    for bad in ["../x", "a/b", "", "TOO_LONG_TICKER_NAME", "x y"]:
        with pytest.raises(ValueError):
            dl.clean_ticker(bad)


def test_page_carries_menu_data(server):
    body = get(server + "/?ticker=TEST&strategy=MeanReversion&entry=1.5")[1]
    data = json.loads(body.split("const DATA = ")[1].split(";\n")[0])
    assert data["app"]["strategy"] == "MeanReversion"
    assert data["app"]["settings"] == {"lookback": 20, "entry": 1.5}
    assert data["app"]["tickers"] == ["TEST"]
    assert [s["id"] for s in data["app"]["strategies"]][0] == "BuyAndHold"
    plain = get(server + "/?ticker=TEST&strategy=none")[1]
    assert '"strategy":"none","settings":{}' in plain


def test_unknown_ticker_message(monkeypatch):
    class Resp:
        status_code = 404
        def raise_for_status(self): raise AssertionError("should not get here")
    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: Resp())
    with pytest.raises(ValueError, match="no data for 'ZZZZQ'"):
        dl.fetch("ZZZZQ")
