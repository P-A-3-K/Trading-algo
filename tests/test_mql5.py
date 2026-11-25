"""MQL5 Expert Advisor helpers: the holiday calendar and the installer (with a fake MT5)."""
import re
import types

from quantlab import mql5 as Q


def test_calendar_has_known_holidays_and_early_closes():
    text = Q.calendar_mqh(2026, 2026)
    arrays = {m.group(1): [int(x) for x in m.group(2).split(",")]
              for m in re.finditer(r"int (QL_\w+)\[\] = \{([^}]*)\};", text)}
    assert set(arrays) == {f"QL_{k}_{s}" for k in Q.EXCHANGES for s in ("CLOSED", "EARLY", "EARLY_MIN")}
    assert 20261126 in arrays["QL_US_CLOSED"]                    # Thanksgiving
    assert 20261225 in arrays["QL_EU_CLOSED"]                    # Christmas (Xetra)
    i = arrays["QL_US_EARLY"].index(20261127)                    # day after Thanksgiving: 13:00 close
    assert arrays["QL_US_EARLY_MIN"][i] == 13 * 60
    assert len(arrays["QL_US_EARLY"]) == len(arrays["QL_US_EARLY_MIN"])
    assert all(20260101 <= d <= 20261231 for d in arrays["QL_US_CLOSED"])


def test_repo_ea_files_exist_and_include_the_calendar():
    ea = (Q.SRC / "QuantLabORB.mq5").read_text(encoding="utf-8")
    assert '#include "QuantLabCalendar.mqh"' in ea
    for name in ("QL_US_CLOSED", "QL_EU_EARLY_MIN", "QL_UK_CLOSED", "QL_JP_EARLY"):
        assert name in ea and name in (Q.SRC / "QuantLabCalendar.mqh").read_text(encoding="utf-8")


def test_install_copies_and_compiles(tmp_path, monkeypatch, capsys):
    data, term = tmp_path / "data", tmp_path / "terminal"
    term.mkdir()
    (term / "metaeditor64.exe").write_bytes(b"")
    fake = types.SimpleNamespace(terminal_info=lambda: types.SimpleNamespace(data_path=str(data), path=str(term)))
    from quantlab import mt5 as M
    monkeypatch.setattr(M, "connect", lambda path=None: (fake, {"company": "FTMO", "server": "FTMO-Demo"}))
    ran = []

    def fake_run(cmd, check=False):
        ran.append(cmd)
        log = cmd[2].split(":", 1)[1]
        open(log, "wb").write("Result: 0 errors, 0 warnings\r\n".encode("utf-16"))

    monkeypatch.setattr(Q.subprocess, "run", fake_run)
    Q.install()
    dest = data / "MQL5" / "Experts" / "QuantLab"
    assert (dest / "QuantLabORB.mq5").exists() and (dest / "QuantLabCalendar.mqh").exists()
    assert ran and ran[0][1].startswith("/compile:")
    out = capsys.readouterr().out
    assert "0 errors" in out and "Compiled OK" in out
