"""QuantLab web app: a small local web server for the chart and backtests.

    python -m quantlab.app                 # starts on http://127.0.0.1:8000 and opens your browser
    python -m quantlab.app --port 8050 --no-browser

Pages and data it serves:
    /                                   chart page; choose with the address, e.g.
    /?ticker=QQQ&strategy=SmaCrossover&fast=20&slow=100
    /api/strategies                     every strategy with its settings (for the menus)
    /api/tickers                        downloaded tickers, plus the popular list
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

from . import chart, strategies, universe
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


def strategies_info() -> list:
    return [{"id": key, **cls.info()} for key, cls in strategies.available().items()]


class Handler(BaseHTTPRequestHandler):
    server_version = "QuantLab/1.0"

    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        routes = {"/": self.page, "/api/strategies": self.api_strategies,
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
            "tickers": universe.menu(dl.cached_tickers()),
        }
        self.send(HTTPStatus.OK, chart.PAGE.format(body=chart.render_payload(payload)),
                  "text/html; charset=utf-8")

    def api_strategies(self, query):
        self.send_json({"strategies": strategies_info()})

    def api_tickers(self, query):
        self.send_json({"tickers": dl.cached_tickers(),
                        "popular": universe.menu(dl.cached_tickers())})

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
