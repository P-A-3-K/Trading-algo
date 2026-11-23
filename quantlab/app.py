"""QuantLab web app: a small local web server for the chart and backtests.

    python -m quantlab.app                 # starts on http://127.0.0.1:8000 and opens your browser
    python -m quantlab.app --port 8050 --no-browser
    python -m quantlab.app --phone           # also reachable from your phone on the same Wi-Fi

Pages and data it serves:
    /                                   chart page; choose with the address, e.g.
    /?ticker=QQQ&strategy=SmaCrossover&fast=20&slow=100&tf=5m   (tf: 5m 15m 30m 1h 4h 1d)
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
    settings = {k: v for k, v in query.items() if k in cls.all_params() and v != ""}
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
    """Starting from/to/step for each setting: its allowed range in ~20 round steps.

    Only the first two settings vary by default (the rest stay at their default),
    so a strategy with many settings doesn't start with a huge grid.
    """
    out = {}
    for n, (k, p) in enumerate(cls.all_params().items()):
        if p.is_bool:                                   # yes/no settings: fixed at default
            out[k] = [int(p.default), int(p.default), 1]
            continue
        lo = p.min if p.min is not None else p.default
        hi = p.max if p.max is not None else p.default
        if n >= 2:
            lo = hi = p.default
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


COST_FIELDS = {"c_comm": "commission_bps", "c_spread": "spread_bps", "c_ext": "extended_spread_bps",
               "c_slip": "slippage_range", "c_impact": "impact", "c_account": "account"}


def costs_of(query: dict):
    """(preset name, CostModel) from the address: cost=realistic|flat5|none|expensive|custom."""
    from .costs import PRESETS, CostModel
    name = query.get("cost") or "realistic"
    if name == "custom":
        base = CostModel().as_dict()
        for q, field in COST_FIELDS.items():
            if query.get(q, "") != "":
                try:
                    base[field] = float(query[q])
                except ValueError:
                    raise ValueError(f"cost setting {field} must be a number, got {query[q]!r}")
                if base[field] < 0:
                    raise ValueError(f"cost setting {field} can't be negative")
        return name, CostModel(**base)
    if name not in PRESETS:
        raise ValueError(f"Unknown cost preset {name!r}. Choose from: {', '.join(PRESETS)}, custom")
    return name, PRESETS[name]


def cost_menu(name: str, model) -> dict:
    from .costs import PRESETS
    return {"preset": name, "values": {q: getattr(model, f) for q, f in COST_FIELDS.items()},
            "presets": {k: {q: getattr(m, f) for q, f in COST_FIELDS.items()} for k, m in PRESETS.items()}}


def timeframe_of(query: dict) -> str:
    return dl.check_timeframe(query.get("tf") or "1d")


TIMEFRAME_MENU = [[k, v[1]] for k, v in dl.TIMEFRAMES.items()]


def optimize_payload(ticker: str, query: dict) -> dict:
    """Everything the optimizer page needs; runs the sweep when query has run=1."""
    cls = strategies.get(query.get("strategy") or DEFAULT_STRATEGY)
    if not cls.params:
        raise ValueError(f"{cls.name} has no settings to sweep. Pick another strategy.")
    ranges = default_ranges(cls)
    for k, p in cls.all_params().items():
        if f"{k}_from" in query:
            ranges[k] = [float(query[f"{k}_from"]), float(query[f"{k}_to"]), float(query[f"{k}_step"])]
            if p.is_int or p.is_bool:
                ranges[k] = [int(round(x)) for x in ranges[k]]
    split = float(query.get("split", 0.7))
    rank = query.get("rank", "sharpe")
    if rank not in RANK_METRICS:
        rank = "sharpe"
    mode = "wf" if query.get("mode") == "wf" else "split"
    tf = timeframe_of(query)
    intraday = tf != "1d"
    # Walk-forward windows: years on daily bars, trading days on intraday bars
    wf = {"unit": "days" if intraday else "years",
          "train": float(query.get("train", 20 if intraday else 8)),
          "test": float(query.get("test", 5 if intraday else 2)),
          "anchored": query.get("window") == "anchored"}
    cost_name, cost_model = costs_of(query)
    app = {"strategy": cls.__name__, "strategies": strategies_info(), "errors": strategy_errors(),
           "costs": cost_menu(cost_name, cost_model),
           "tickers": dl.cached_tickers(), "ranges": ranges, "split": split,
           "rank": rank, "mode": mode, "wf": wf,
           "max_combos": optimize.WF_MAX_COMBINATIONS if mode == "wf" else optimize.MAX_COMBINATIONS}
    app["timeframes"], app["timeframe"] = TIMEFRAME_MENU, tf
    payload = {"ticker": ticker, "timeframe": tf, "intraday": intraday, "app": app, "result": None}
    if query.get("run") and mode == "wf":
        bars = dl.load_bars(ticker, tf)
        per_year = 252 if wf["unit"] == "days" else 1        # days -> years
        res = optimize.walk_forward(bars, cls, {k: tuple(v) for k, v in ranges.items()},
                                    wf["train"] / per_year, wf["test"] / per_year,
                                    wf["anchored"], rank, costs=cost_model)
        payload["result"] = walk_forward_payload(res, rank)
        return payload
    if query.get("run"):
        bars = dl.load_bars(ticker, tf)
        res = optimize.sweep(bars, cls, {k: tuple(v) for k, v in ranges.items()}, split,
                             costs=cost_model)
        if res.table.empty:
            raise ValueError("No valid combinations: " + (res.skipped[0][1] if res.skipped else "empty grid"))
        varied = [k for k, (a, b, _) in ranges.items() if b > a]
        p_int = {k: cls.all_params()[k].is_int or cls.all_params()[k].is_bool for k in ranges}
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


def walk_forward_payload(res, rank: str) -> dict:
    """Walk-forward result as plain data for the page."""
    from .metrics import full_stats
    def curve(r):
        eq = chart.START_CAPITAL * (1 + r).cumprod()
        return [{"time": chart.tkey(d), "value": round(float(v), 2)}
                for d, v in zip(eq.index, eq.values)]
    def stats(r, held=None):
        st = full_stats(r, held)
        return {"final": chart.START_CAPITAL * (1 + st["total_return"]), "cagr": st["cagr"],
                "sharpe": st["sharpe"], "max_dd": st["max_drawdown"],
                "exposure": st.get("time_in_market")}
    when = lambda v: v.strftime("%Y-%m-%d %H:%M") if v.tzinfo is not None else str(v.date())
    folds = [{**{k: (when(v) if hasattr(v, "date") else v) for k, v in f.items()
                 if k != "settings"}, "settings": f["settings"]} for f in res.folds]
    pos_train = [f["train_cagr"] for f in res.folds]
    eff = (sum(f["test_cagr"] for f in res.folds) / sum(pos_train)
           if sum(pos_train) > 0 else None)
    return {
        "mode": "wf", "folds": folds, "params": res.params, "rank": rank,
        "count": res.combos, "skipped": len(res.skipped),
        "skip_reason": res.skipped[0][1] if res.skipped else "",
        "seconds": res.seconds, "start": when(res.returns.index[0]),
        "end": when(res.returns.index[-1]),
        "curves": {"wf": curve(res.returns), "bh": curve(res.benchmark), "hind": curve(res.hindsight)},
        "stats": {"wf": stats(res.returns, res.held), "bh": stats(res.benchmark),
                  "hind": stats(res.hindsight)},
        "hindsight_settings": res.hindsight_settings,
        "distinct_settings": len({tuple(sorted(f["settings"].items())) for f in res.folds}),
        "efficiency": clean(eff),
        "luck": chart.luck_payload(res.returns, res.benchmark),
    }


def strategies_info() -> list:
    """Menu entries: built-ins first, then your strategies/ files (with any problem)."""
    builtin, found = strategies.catalog()
    out = [{"id": c.__name__, "source": "builtin", "file": "", "problem": "", **c.info()}
           for c in builtin]
    out += [{"id": sid, "source": "user", "file": f.file, "problem": f.problem, **f.cls.info()}
            for sid, f in found.strategies.items()]
    return out


def strategy_errors() -> list:
    """Files in strategies/ that couldn't be loaded, for a notice in the app."""
    return [{"file": f, "error": e} for f, e in strategies.catalog()[1].errors]


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
        cost_name, cost_model = costs_of(query)
        payload = chart.build_payload(ticker, strategy, timeframe=timeframe_of(query),
                                      costs=cost_model)
        # What the control bar needs: menus, and the current choice to pre-select
        payload["app"] = {
            "strategy": type(strategy).__name__ if strategy else "none",
            "settings": dict(strategy.settings) if strategy else {},
            "strategies": strategies_info(),
            "errors": strategy_errors(),
            "timeframes": TIMEFRAME_MENU,
            "costs": cost_menu(cost_name, cost_model),
            "tickers": dl.cached_tickers(),
        }
        self.send(HTTPStatus.OK, chart.PAGE.format(body=chart.render_payload(payload)),
                  "text/html; charset=utf-8")

    def optimize_page(self, query):
        ticker = dl.clean_ticker(query.get("ticker") or DEFAULT_TICKER)
        payload = optimize_payload(ticker, query)
        library = (chart.WEB / "vendor" / "lightweight-charts.standalone.production.js").read_text(encoding="utf-8")
        html_page = (chart.WEB / "optimize.html").read_text(encoding="utf-8").replace(
            "/*__DATA__*/null", json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
        ).replace("/*__LIBRARY__*/", library)
        self.send(HTTPStatus.OK, chart.PAGE.format(body=html_page), "text/html; charset=utf-8")

    def api_strategies(self, query):
        self.send_json({"strategies": strategies_info(), "errors": strategy_errors()})

    def api_tickers(self, query):
        self.send_json({"tickers": dl.cached_tickers()})

    def api_backtest(self, query):
        ticker = query.get("ticker") or DEFAULT_TICKER
        self.send_json(chart.build_payload(ticker, strategy_from_query(query),
                                           timeframe=timeframe_of(query),
                                           costs=costs_of(query)[1]))

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


def lan_addresses() -> list:
    """This computer's addresses that other devices can reach (Wi-Fi/LAN, Tailscale).

    Leaves out 127.x (this computer only) and 169.254.x (no network).
    """
    import socket
    found = []
    try:                                   # the address used for the default route
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))   # no packet is sent for UDP connect
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    except OSError:
        pass
    out = []
    for ip in found:
        if ip not in out and not ip.startswith(("127.", "169.254.", "0.")):
            out.append(ip)
    return out


def describe_address(ip: str) -> str:
    """Which kind of network an address belongs to, for the start-up message."""
    a, b = (int(x) for x in ip.split(".")[:2])
    if a == 100 and 64 <= b <= 127:
        return "Tailscale (works from anywhere on your Tailscale devices)"
    if a == 192 and b == 168 or a == 10 or a == 172 and 16 <= b <= 31:
        return "same Wi-Fi / home network"
    return "network"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the QuantLab web app.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 = only this computer (default)")
    ap.add_argument("--phone", action="store_true",
                    help="also allow other devices (your phone) on your network to connect")
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = ap.parse_args(argv)
    host = "0.0.0.0" if args.phone else args.host
    server = make_server(host, args.port)
    port = server.server_address[1]
    local = f"http://127.0.0.1:{port}/"
    print(f"QuantLab is running at {local}  (press Ctrl+C to stop)")
    if host == "0.0.0.0":
        addrs = lan_addresses()
        print()
        if addrs:
            print("  On your phone, open one of these in the browser:")
            for ip in addrs:
                print(f"      http://{ip}:{port}/      ({describe_address(ip)})")
        else:
            print("  Couldn't find this computer's network address. Run 'ipconfig' and use")
            print(f"  the IPv4 address it shows: http://<that address>:{port}/")
        print()
        print("  Your phone must be on the same Wi-Fi (or on Tailscale).")
        print("  NOTE: there is no password, so anyone on this network can open QuantLab")
        print("  while this window is open. Fine at home; don't use it on public Wi-Fi.")
        print()
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(local,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
