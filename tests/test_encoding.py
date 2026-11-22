"""Windows reads text files as cp1252 unless told otherwise, which breaks on
characters like (c) and TM. Every text read/write must name its encoding."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALL = re.compile(r"\.(read_text|write_text)\(|(?<![\w.])open\(")


def test_text_io_names_an_encoding():
    offenders = []
    for path in list((ROOT / "quantlab").rglob("*.py")) + list((ROOT / "lessons").glob("*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if CALL.search(line) and "encoding=" not in line and "webbrowser.open" not in line:
                offenders.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_rendering_never_relies_on_default_encoding():
    """Python's own check: -X warn_default_encoding flags every text open()
    without an encoding; -W error turns those into failures."""
    import subprocess, sys
    code = ("import pandas as pd; from quantlab import chart; "
            "df = pd.DataFrame({'Open':[1.0],'High':[1.0],'Low':[1.0],'Close':[1.0],"
            "'Adj Close':[1.0],'Volume':[1]}, index=pd.to_datetime(['2024-01-02'])); "
            "assert 'TradingView' in chart.render('X', df)")
    run = subprocess.run([sys.executable, "-X", "warn_default_encoding",
                          "-W", "error::EncodingWarning", "-c", code],
                         cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-2000:]
