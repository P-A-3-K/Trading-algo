import pandas as pd
import pytest

from quantlab import compare
from quantlab import download as dl
from quantlab.data import synthetic_prices


def test_compare_table(tmp_path, monkeypatch):
    c = synthetic_prices(n_days=600, seed=4)
    pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Adj Close": c, "Volume": 1000},
                 index=c.index.rename("Date")).to_csv(tmp_path / "TEST.csv")
    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dl, "fetch", lambda t: (_ for _ in ()).throw(RuntimeError("offline")))
    df = compare.compare("TEST", "1d", ["SmaCrossover fast=10 slow=50", "RsiReversion"])
    assert list(df.index) == ["SmaCrossover fast=10 slow=50", "RsiReversion", "Buy & hold"]
    assert df.loc["Buy & hold", "in market"] == 1.0 and df["sharpe"].notna().all()
    assert "Buy & hold" in compare.show(df)
    with pytest.raises(ValueError):
        compare.parse_strategy("SmaCrossover fast=500 slow=10")
