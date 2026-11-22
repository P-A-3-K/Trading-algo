"""QuantLab web app: a small local web server for the chart and backtests.

    python -m quantlab.app                 # starts on http://127.0.0.1:8000 and opens your browser
    python -m quantlab.app --port 8050 --no-browser

Pages and data it serves:
    /                                   chart page; choose with the address, e.g.
    /?ticker=QQQ&strategy=SmaCrossover&fast=20&slow=100
    /api/strategies                     every strategy with its settings (for the menus)
    /api/tickers                        tickers already downloaded into data/
    /api/backtest?ticker=SPY&strategy=MeanReversion&entry=1.5   the page's data as JSON

It only listens on 127.0.0.1 (your own computer), so nobody else on your
network can reach it. Built on Python's standard library: nothing to install.
"""
import argparse
import html
import json
import threading
import time
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import math

from . import chart, optimize, strategies
from . import download as dl

DEFAULT_TICKER = "SPY"
DEFAULT_STRATEGY = "SmaCrossover"


def strategy_from_query(query: dict):
    """Build the strategy the address asks for, e.g. strategy=SmaCrossover&fast=20.

    strategy=none (or empty) means plain candles. Settings the strategy
    doesn't have are ignored, so one address format works for all of them.
    """
    name = query.get("strategy", DEFAULT_STRATEGY)
    if not name or name.lower() == "none":
        return None
    cls = strategies.get(name)
    settings = {k: v for k, v in query.items() if k in cls.params and v != ""}
    return cls(**settings)


RANK_METRICS = ("sharpe", "cagr", "final", "max_dd", "trades")


def nice_step(span: float, is_int: bool, target: int = 19) -> float:
    """A round step that gives about `target` values across `span`."""
    raw = span / target
    for e in range(-3, 6):                  # smallest round step that is big enough
        for m in (1, 2, 2.5, 5):
            step = m * 10 ** e
            if step >= raw:
                return max(1, int(round(step))) if is_int else step
    return raw


def default_ranges(cls) -> dict:
    """Starting from/to/step for each setting: its allowed range in ~20 round steps."""
    out = {}
    for k, p in cls.params.items():
        lo = p.min if p.min is not None else p.default
        hi = p.max if p.max is not None else p.default
        if hi == lo:
            out[k] = [p.default, p.default, 1 if p.is_int else 0.1]
            continue
        step = nice_step(hi - lo, p.is_int)
        start = math.ceil(lo / step) * step
        stop = math.floor(hi / step) * step
        if p.is_int:
            start, stop = int(start), int(stop)
        else:
            start, stop, step = round(start, 6), round(stop, 6), round(step, 6)
        out[k] = [start, stop, step]
    return out


def clean(v):
    """JSON-safe number: NaN/inf -> None, numpy -> float."""
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else v


def optimize_payload(ticker: str, query: dict) -> dict:
    """Everything the optimizer page needs; runs the sweep when query has run=1."""
    cls = strategies.get(query.get("strategy") or DEFAULT_STRATEGY)
    if not cls.params:
        raise ValueError(f"{cls.name} has no settings to sweep. Pick another strategy.")
    ranges = default_ranges(cls)
    for k, p in cls.params.items():
        if f"{k}_from" in query:
            ranges[k] = [float(query[f"{k}_from"]), float(query[f"{k}_to"]), float(query[f"{k}_step"])]
            if p.is_int:
                ranges[k] = [int(round(x)) for x in ranges[k]]
    split = float(query.get("split", 0.7))
    rank = query.get("rank", "sharpe")
    if rank not in RANK_METRICS:
        rank = "sharpe"
    app = {"strategy": cls.__name__, "strategies": strategies_info(),
           "tickers": dl.cached_tickers(), "ranges": ranges, "split": split,
           "rank": rank, "max_combos": optimize.MAX_COMBINATIONS}
    payload = {"ticker": ticker, "app": app, "result": None}
    if query.get("run"):
        bars = dl.load_bars(ticker)
        res = optimize.sweep(bars, cls, {k: tuple(v) for k, v in ranges.items()}, split)
        if res.table.empty:
            raise ValueError("No valid combinations: " + (res.skipped[0][1] if res.skipped else "empty grid"))
        varied = [k for k, (a, b, _) in ranges.items() if b > a]
        p_int = {k: cls.params[k].is_int for k in ranges}
        rows = [{k: (clean(v) if k not in ranges else (int(v) if p_int[k] else float(v)))
                 for k, v in row.items()} for row in res.table.to_dict("records")]
        idx = bars.index
        split_pos = idx.get_loc(res.split_date)
        payload["result"] = {
            "rows": rows, "params": varied, "count": len(rows),
            "skipped": len(res.skipped),
            "skip_reason": res.skipped[0][1] if res.skipped else "",
            "seconds": res.seconds,
            "benchmark": {k: clean(v) for k, v in res.benchmark.items()},
            "report": {k: (clean(v) if not isinstance(v, str) else v)
                       for k, v in optimize.overfitting_report(res, "sharpe").items()},
            "axis_values": {k: optimize.values(*ranges[k], p_int[k]) for k in varied},
            "start": str(res.start.date()), "end": str(res.end.date()),
            "split_date": str(res.split_date.date()),
            "split_before": str(idx[split_pos - 1].date()),
        }
    return payload


def strategies_info() -> list:
    return [{"id": key, **cls.info()} for key, cls in strategies.available().items()]


class Handler(BaseHTTPRequestHandler):
    server_version = "QuantLab/1.0"

    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        routes = {"/": self.page, "/optimize": self.optimize_page,
                  "/api/strategies": self.api_strategies,
                  "/api/tickers": self.api_tickers, "/api/backtest": self.api_backtest,
                  "/favicon.ico": self.no_content}
        handler = routes.get(url.path)
        if handler is None:
            return self.send_text(HTTPStatus.NOT_FOUND, "Not found")
        try:
            handler(query)
        except ValueError as e:                    # bad ticker, strategy or setting
            self.fail(url.path, HTTPStatus.BAD_REQUEST, str(e))
        except Exception as e:                     # e.g. the download failed
            traceback.print_exc()
            self.fail(url.path, HTTPStatus.INTERNAL_SERVER_ERROR,
                      f"{type(e).__name__}: {e}")

    # ---- routes --------------------------------------------------------------
    def page(self, query):
        ticker = query.get("ticker") or DEFAULT_TICKER
        strategy = strategy_from_query(query)
        payload = chart.build_payload(ticker, strategy)
        # What the control bar needs: menus, and the current choice to pre-select
        payload["app"] = {
            "strategy": type(strategy).__name__ if strategy else "none",
            "settings": dict(strategy.settings) if strategy else {},
            "strategies": strategies_info(),
            "tickers": dl.cached_tickers(),
        }
        self.send(HTTPStatus.OK, chart.PAGE.format(body=chart.render_payload(payload)),
                  "text/html; charset=utf-8")

    def optimize_page(self, query):
        ticker = dl.clean_ticker(query.get("ticker") or DEFAULT_TICKER)
        payload = optimize_payload(ticker, query)
        html_page = (chart.WEB / "optimize.html").read_text(encoding="utf-8").replace(
            "/*__DATA__*/null", json.dumps(payload, separators=(",", ":")).replace("</", "<\\/"))
        self.send(HTTPStatus.OK, chart.PAGE.format(body=html_page), "text/html; charset=utf-8")

    def api_strategies(self, query):
        self.send_json({"strategies": strategies_info()})

    def api_tickers(self, query):
        self.send_json({"tickers": dl.cached_tickers()})

    def api_backtest(self, query):
        ticker = query.get("ticker") or DEFAULT_TICKER
        self.send_json(chart.build_payload(ticker, strategy_from_query(query)))

    def no_content(self, query):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    # ---- helpers -------------------------------------------------------------
    def fail(self, path, status, message):
        if path.startswith("/api/"):
            return self.send_json({"error": message}, status)
        self.send(status, error_page(message), "text/html; charset=utf-8")

    def send(self, status, body: str, content_type: str):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, obj, status=HTTPStatus.OK):
        self.send(status, json.dumps(obj, separators=(",", ":")), "application/json")

    def send_text(self, status, text):
        self.send(status, text, "text/plain; charset=utf-8")

    def log_message(self, fmt, *args):            # one short line per request
        print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")


def error_page(message: str) -> str:
    return chart.PAGE.format(body=f"""<title>QuantLab Error</title>
<style>
:root {{ --bg:#f6f7f9; --panel:#fff; --fg:#15191f; --muted:#5d6673; --line:#dfe3e8; --down:#c8423b; --accent:#2f5fd0; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0f1216; --panel:#151a20; --fg:#e6e9ee;
  --muted:#8b95a3; --line:#262d36; --down:#ef6a61; --accent:#7aa2ff; color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --bg:#0f1216; --panel:#151a20; --fg:#e6e9ee; --muted:#8b95a3; --line:#262d36;
  --down:#ef6a61; --accent:#7aa2ff; color-scheme: dark; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:640px; margin:10vh auto; padding-inline:max(16px, env(safe-area-inset-left, 0px)); }}
.box {{ background:var(--panel); border:1px solid var(--line); border-left:4px solid var(--down); border-radius:8px; padding:16px 20px; }}
h1 {{ font-size:20px; margin:0 0 8px; }} code {{ font-family:ui-monospace, monospace; }} a {{ color:var(--accent); }}
</style>
<main><div class="box"><h1>That didn't work</h1><p><code>{html.escape(message)}</code></p>
<p><a href="javascript:history.back()">Go back</a> &nbsp;·&nbsp; <a href="/">Start over with SPY</a></p></div></main>""")


def make_server(host="127.0.0.1", port=8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the QuantLab web app.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 = only this computer (default)")
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = ap.parse_args(argv)
    server = make_server(args.host, args.port)
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"QuantLab is running at {url}  (press Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
