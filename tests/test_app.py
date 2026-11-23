import json
import threading
import time
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
    tickers = json.loads(get(server + "/api/tickers")[1])
    assert tickers["tickers"] == ["TEST"]
    pop = {t["symbol"]: t for t in tickers["popular"]}
    assert pop["SPY"]["name"] == "S&P 500" and pop["SPY"]["saved"] is False
    assert pop["TEST"] == {"symbol": "TEST", "name": None, "group": "Your downloads", "saved": True}
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
    assert data["app"]["settings"] == {"lookback": 20, "entry": 1.5, "flat_at_close": False}
    assert {"symbol": "TEST", "name": None, "group": "Your downloads", "saved": True} in data["app"]["tickers"]
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


def test_phone_mode_address_helpers(monkeypatch):
    assert "Wi-Fi" in app.describe_address("192.168.1.23")
    assert "Wi-Fi" in app.describe_address("10.0.0.5")
    assert "Tailscale" in app.describe_address("100.101.102.103")
    import socket
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [
        (0, 0, 0, "", ("127.0.0.1", 0)), (0, 0, 0, "", ("169.254.3.4", 0)),
        (0, 0, 0, "", ("100.70.1.2", 0)), (0, 0, 0, "", ("192.168.1.23", 0))])
    addrs = app.lan_addresses()
    assert "127.0.0.1" not in addrs and "169.254.3.4" not in addrs
    assert {"100.70.1.2", "192.168.1.23"} <= set(addrs)


def test_search_route_local_and_remote(server, monkeypatch):
    from quantlab import search
    monkeypatch.setattr(search, "search", lambda q: [
        {"symbol": "GLD", "name": "SPDR Gold Shares", "type": "ETF", "exchange": "NYSE Arca", "us": True, "supported": True},
        {"symbol": "GC=F", "name": "Gold futures", "type": "Future", "exchange": "COMEX", "us": True, "supported": False}])
    r = json.loads(get(server + "/api/search?q=gold")[1])
    assert [x["symbol"] for x in r["local"]] == ["GLD"]            # from the popular list
    assert [x["symbol"] for x in r["remote"]] == ["GC=F"]          # GLD not repeated
    def offline(q):
        raise search.requests.ConnectionError("no network")
    monkeypatch.setattr(search, "search", offline)
    r = json.loads(get(server + "/api/search?q=tech")[1])
    assert r["local"][0]["symbol"] == "XLK" and "unavailable" in r["error"]


def test_index_tickers_allowed():
    assert dl.clean_ticker("^gspc") == "^GSPC"
    for bad in ["^^X", "SP^Y", "^"]:
        with pytest.raises(ValueError):
            dl.clean_ticker(bad)


def test_market_scan_runs_in_the_background(server):
    status, body = get(server + "/optimize?strategy=SmaCrossover&mode=scan&tickers=TEST,NOPE&tfs=1d")
    assert status == 200 and '"tickers":["TEST","NOPE"]' in body      # the page restores the markets
    q = "strategy=SmaCrossover&fast_from=10&fast_to=30&fast_step=10&slow_from=50&slow_to=100&slow_step=50&tickers=TEST,NOPE&tfs=1d"
    status, body = get(server + "/api/scan/start?" + q)
    assert status == 200, body
    job = json.loads(body)["id"]
    for _ in range(100):
        st = json.loads(get(server + f"/api/scan/status?id={job}")[1])
        if st["state"] != "running":
            break
        time.sleep(0.1)
    assert st["state"] == "done", st
    m = {r["ticker"]: r for r in st["result"]["markets"]}
    assert m["TEST"]["error"] is None and m["NOPE"]["error"]
    # too big is refused at once, with a message
    status, body = get(server + "/api/scan/start?strategy=SmaCrossover&fast_from=2&fast_to=400&fast_step=1"
                       "&slow_from=5&slow_to=500&slow_step=10&tickers=" + ",".join(f"T{i}" for i in range(30)) + "&tfs=1d")
    assert status == 400 and "max" in json.loads(body)["error"]
    assert get(server + "/api/scan/status?id=999")[0] == 400


def test_chart_splits_training_and_test_periods(server):
    plain = json.loads(get(server + "/api/backtest?ticker=TEST&strategy=SmaCrossover&fast=10&slow=30")[1])
    mid = plain["bars"][200]["time"]                       # a date inside the synthetic data
    status, body = get(server + f"/api/backtest?ticker=TEST&strategy=SmaCrossover&fast=10&slow=30&test_from={mid}")
    assert status == 200, body
    d = json.loads(body)
    assert d["split"]["test"][0] == mid and d["split"]["train"][1] < mid
    assert set(d["periods"]) == {"all", "train", "test"}
    sharpe = {p: next(r["strategy"] for r in d["periods"][p]["stats"] if r["key"] == "sharpe") for p in d["periods"]}
    assert sharpe["train"] != sharpe["test"]
    # without test_from: no split; a split leaving almost nothing on one side is refused
    assert plain["split"] is None
    status, body = get(server + "/api/backtest?ticker=TEST&strategy=SmaCrossover&fast=10&slow=30&test_from=2035-01-01")
    assert status == 400 and "test period" in json.loads(body)["error"]


def test_prop_check_route(server):
    status, body = get(server + "/api/prop?ticker=TEST&strategy=SmaCrossover&fast=10&slow=30")
    assert status == 200, body
    d = json.loads(body)
    assert d["rules"]["target"] == 0.10 and len(d["rows"]) >= 8 and len(d["benchmark"]) == len(d["rows"])
    status, body = get(server + "/api/prop?ticker=TEST&strategy=SmaCrossover&fast=10&slow=30&prop=custom"
                       "&p_target=8&p_target2=0&p_maxloss=6&p_daily=3&p_trailing=1")
    d = json.loads(body)
    assert d["rules"] == {**d["rules"], "target": 0.08, "target2": 0.0, "max_loss": 0.06, "trailing": True}
    assert get(server + "/api/prop?ticker=TEST&strategy=SmaCrossover&prop=nope")[0] == 400
    assert get(server + "/api/prop?ticker=TEST&strategy=none")[0] == 400


def test_sweep_over_a_choice_setting(server):
    status, body = get(server + "/optimize?ticker=TEST&strategy=FairValueGap&rr_from=2&rr_to=2&rr_step=1"
                       "&swing_from=2&swing_to=2&swing_step=1&trend_tf_values=off,1d&run=1")
    assert status == 200, body[:300]
    assert '"trend_tf":"1d"' in body and '"trend_tf":"off"' in body
    status, body = get(server + "/optimize?ticker=TEST&strategy=FairValueGap&trend_tf_values=2h&run=1")
    assert status == 400 and "one of" in body
