import numpy as np
import pandas as pd
import pytest

from quantlab import download as dl
from quantlab import importer as I


@pytest.fixture()
def files(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "DATA_DIR", tmp_path / "data")
    dl._READ_CACHE.clear()
    idx = pd.date_range("2024-03-04 07:00", "2024-03-05 21:00", freq="1min", tz="UTC")
    c = 18000 + np.arange(len(idx)) * 0.1
    vol = np.where((idx.hour >= 7) & (idx.hour < 21), 1.5, 0.0)
    duka = pd.DataFrame({"Gmt time": idx.strftime("%d.%m.%Y %H:%M:%S.000"), "Open": c, "High": c + 1,
                         "Low": c - 1, "Close": c, "Volume": vol})
    duka.loc[vol == 0, ["Open", "High", "Low", "Close"]] = 18000.0          # closed-market filler
    duka.to_csv(tmp_path / "gmt.csv", index=False)
    local = duka.rename(columns={"Gmt time": "Local time"})
    local["Local time"] = (idx + pd.Timedelta(hours=1)).strftime("%d.%m.%Y %H:%M:%S.000") + " GMT+0100"
    local.to_csv(tmp_path / "local.csv", index=False)
    s = idx + pd.Timedelta(hours=2)
    pd.DataFrame({"<DATE>": s.strftime("%Y.%m.%d"), "<TIME>": s.strftime("%H:%M:%S"), "<OPEN>": c, "<HIGH>": c + 1,
                  "<LOW>": c - 1, "<CLOSE>": c, "<TICKVOL>": 10, "<VOL>": 0, "<SPREAD>": 5}
                 ).to_csv(tmp_path / "mt5.csv", sep="\t", index=False)
    return tmp_path


def test_formats_agree(files):
    a = I.parse(files / "gmt.csv")
    b = I.parse(files / "local.csv")
    m = I.parse(files / "mt5.csv", "UTC+2")
    assert a.index[0] == b.index[0] == m.index[0] == pd.Timestamp("2024-03-04 07:00", tz="UTC")
    assert len(a) == len(b) and a["Close"].equals(b["Close"])
    assert (a["Volume"] > 0).all()                                    # filler minutes dropped
    with pytest.raises(ValueError, match="open/high/low/close"):
        (files / "bad.csv").write_text("time,price\n2024-01-01,1\n", encoding="utf-8")
        I.parse(files / "bad.csv")


def test_import_merge_and_load(files):
    I.import_files("GER40", [files / "gmt.csv"])
    I.import_files("GER40", [files / "gmt.csv"])                       # importing twice changes nothing
    b5 = dl.load_bars("CSV:GER40", "5m")
    assert len(b5) == 2 * 14 * 12 and str(b5.index[0]) == "2024-03-04 02:00:00-05:00"
    assert b5["open"].iloc[0] == 18000.0 and b5["close"].iloc[0] == pytest.approx(18000.4)
    assert len(dl.load_bars("CSV:ger40", "1h")) == 2 * 14                 # name matched regardless of case
    d = dl.load_bars("CSV:GER40", "1d")
    assert d.index.tz is None and len(d) == 2
    assert "CSV:GER40" in dl.cached_tickers()
    with pytest.raises(ValueError, match="Import them first"):
        dl.load_bars("CSV:NOPE", "5m")
