import os
import time

import pytest

from quantlab import discovery

GOOD = '''
from quantlab.strategy import Param, Strategy
class Momentum(Strategy):
    name = "Momentum"
    params = {"days": Param(20, 2, 200, "Lookback (days)")}
    def compute(self, bars):
        return (bars.close.pct_change(self.days) > 0).astype(float)
'''
CHEAT = '''
from quantlab.strategy import Strategy
class Peek(Strategy):
    name = "Peek"
    def compute(self, bars):
        return (bars.close.shift(-1) > bars.close).astype(float)
'''


@pytest.fixture()
def folder(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery, "_cache", {})
    return tmp_path


def write(folder, name, text):
    p = folder / name
    p.write_text(text, encoding="utf-8")
    return p


def test_finds_strategies_and_ignores_underscore_files(folder):
    write(folder, "momentum.py", GOOD)
    write(folder, "_helpers.py", GOOD.replace("Momentum", "Hidden"))
    s = discovery.scan(folder)
    assert list(s.strategies) == ["Momentum"] and s.errors == []
    assert s.strategies["Momentum"].file == f"{folder.name}/momentum.py"
    assert s.strategies["Momentum"].problem == ""


def test_broken_file_is_reported_not_fatal(folder):
    write(folder, "momentum.py", GOOD)
    write(folder, "broken.py", "def oops(:\n")
    s = discovery.scan(folder)
    assert "Momentum" in s.strategies
    (file, err), = s.errors
    assert file.endswith("broken.py") and "SyntaxError" in err and str(folder) not in err


def test_lookahead_strategy_is_blocked(folder):
    write(folder, "peek.py", CHEAT)
    s = discovery.scan(folder)
    assert "looks into the future" in s.strategies["Peek"].problem


def test_name_clash_with_builtin(folder):
    write(folder, "dup.py", GOOD.replace("class Momentum", "class SmaCrossover"))
    s = discovery.scan(folder, builtin={"SmaCrossover"})
    assert s.strategies == {} and "already used" in s.errors[0][1]


def test_edited_file_is_reloaded(folder):
    p = write(folder, "momentum.py", GOOD)
    assert discovery.scan(folder).strategies["Momentum"].cls.params["days"].default == 20
    p.write_text(GOOD.replace("Param(20,", "Param(30,"), encoding="utf-8")
    t = time.time() + 5
    os.utime(p, (t, t))                       # make sure the change is visible
    assert discovery.scan(folder).strategies["Momentum"].cls.params["days"].default == 30


def test_repo_strategies_folder_is_clean():
    """Every file shipped in strategies/ loads and passes the look-ahead check."""
    s = discovery.scan()
    assert s.errors == []
    assert "Breakout" in s.strategies
    assert all(not f.problem for f in s.strategies.values())
