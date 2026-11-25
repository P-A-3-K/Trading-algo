"""Automatic trader: runs the opening range breakout (ORB) legs on an MT5 account.

    python -m quantlab.live init                 # writes data/live/live.json (edit it)
    python -m quantlab.live run --dry-run        # logs what it WOULD do, sends nothing
    python -m quantlab.live run                  # trades the (demo) account MT5 is logged into
    python -m quantlab.live status               # today's journal

Every trading day, for every leg (e.g. US100 at the US open, GER40 at the Frankfurt open):
  1. After the opening range (e.g. 9:30-10:00 New York) it measures the range's high and low
     from MT5's 1-minute candles.
  2. It places a BUY STOP at the high and a SELL STOP at the low (if shorts are on), each with
     its stop-loss at the other side of the range (or the middle), and a take-profit at
     rr x the risk (none when rr = 0). If the price already broke out, it enters at market.
  3. When one fills, the other is cancelled: one trade a day, like the backtest.
  4. A couple of minutes before the session closes, it closes the trade and cancels leftovers.
Size: the backtest's rule (risk_pct of the account on the stop distance, at most 100%)
x the leg's weight x the plan's multiplier, converted to lots from the symbol's contract.

Safety: demo accounts only unless "allow_real" is set; a daily loss guard (default 3.5%,
FTMO's limit is 5%) closes everything and pauses until the next day; a total loss guard
(default 8% below the starting balance) stops for good; only orders with its own magic
number are touched; a file data/live/STOP stops it; each day's progress is saved, so a
restart never trades a leg twice. Everything is written to data/live/journal.csv.

In MT5, the "Algo Trading" button in the toolbar must be ON (green), or orders are refused.
"""
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

MAGIC_BASE = 2611000
LATE_START_MINUTES = 5          # started later than this after the range: skip the day
EXCHANGE_CALENDARS = {"US": "XNYS", "Europe": "XETR", "London": "XLON", "Tokyo": "XJPX"}


# ---- Configuration -------------------------------------------------------------------------

@dataclass
class LegConfig:
    name: str                       # e.g. "US100 ORB"
    symbol: str                     # the broker's symbol, e.g. "US100.cash"
    strategy: str = "OpeningRangeBreakout"
    settings: dict = field(default_factory=dict)   # e.g. {"session": "US", "rr": 0}
    weight: float = 0.5             # share of the account, like on the Portfolio page


@dataclass
class LiveConfig:
    legs: list
    multiplier: float = 1.0         # the plan's size (2 = challenge at 2x)
    cushion_multiplier: float = None  # optional: use this size instead ...
    cushion_drawdown: float = 0.04    # ... while equity is this far below the starting balance
    daily_loss_guard: float = 0.035 # pause for the day after this loss (fraction of day-start equity)
    total_loss_guard: float = 0.08  # stop for good this far below the starting balance
    start_balance: float = None     # set on the first run if empty
    close_minutes_before: int = 2   # close open trades this long before the session ends
    allow_real: bool = False        # refuse real-money accounts unless True
    poll_seconds: int = 5
    terminal_path: str = None       # MT5 terminal64.exe to use (None: the remembered one)

    @staticmethod
    def load(path: Path) -> "LiveConfig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["legs"] = [LegConfig(**x) for x in d["legs"]]
        return LiveConfig(**d)

    def save(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        Path(path).write_text(json.dumps(d, indent=2), encoding="utf-8")


def live_dir() -> Path:
    from .download import DATA_DIR
    return DATA_DIR / "live"


def example_config() -> LiveConfig:
    return LiveConfig(legs=[
        LegConfig("US100 ORB", "US100.cash", settings={"session": "US", "range_minutes": 30, "rr": 0,
                                                       "stop_at": "other side", "allow_short": True, "risk_pct": 1.0},
                  weight=0.5),
        LegConfig("GER40 ORB", "GER40.cash", settings={"session": "Europe", "range_minutes": 30, "rr": 0,
                                                       "stop_at": "other side", "allow_short": True, "risk_pct": 1.0},
                  weight=0.5),
    ], multiplier=1.0)


# ---- Journal -------------------------------------------------------------------------------

JOURNAL_FIELDS = ["time_utc", "leg", "event", "side", "price", "planned", "sl", "tp", "volume", "ticket",
                  "equity", "note"]


class Journal:
    def __init__(self, path: Path, echo=True):
        self.path, self.echo = Path(path), echo
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(JOURNAL_FIELDS)

    def write(self, now: datetime, leg: str, event: str, **kw):
        row = {"time_utc": now.strftime("%Y-%m-%d %H:%M:%S"), "leg": leg, "event": event, **kw}
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=JOURNAL_FIELDS, extrasaction="ignore").writerow(row)
        if self.echo:
            extra = " ".join(f"{k}={v}" for k, v in kw.items() if v not in (None, ""))
            print(f"{row['time_utc']} UTC  {leg:12s} {event:14s} {extra}", flush=True)


# ---- Sizing --------------------------------------------------------------------------------

def lots_for(equity: float, weight: float, multiplier: float, risk_pct: float, entry: float, stop: float,
             spec: dict) -> float:
    """Lots so the position's value = equity x weight x multiplier x min(1, risk_pct / stop distance %).

    spec: tick_value (account currency per tick per lot), tick_size, volume_min/max/step."""
    risk = abs(entry - stop)
    if risk <= 0 or entry <= 0:
        return 0.0
    frac = min(1.0, risk_pct / 100 / (risk / entry))
    notional = equity * weight * multiplier * frac                       # account currency
    per_lot = entry * spec["tick_value"] / spec["tick_size"]             # one lot's value, account currency
    if per_lot <= 0:
        return 0.0
    step = spec.get("volume_step") or 0.01
    lots = math.floor(notional / per_lot / step + 1e-9) * step
    lots = min(lots, spec.get("volume_max") or lots)
    return round(lots, 8) if lots >= (spec.get("volume_min") or step) else 0.0


# ---- The leg's daily routine (works with any broker object, real or simulated) -------------

def session_of(settings: dict):
    """(time zone, open minute, close minute) of an ORB session."""
    from . import strategies
    cls = strategies.get("OpeningRangeBreakout")
    sessions = sys.modules[cls.__module__].SESSIONS
    return sessions[settings.get("session", "US")]


class LegRunner:
    """One leg's state machine for one day: wait -> placed -> in_trade -> done."""

    def __init__(self, idx: int, cfg: LegConfig, live: LiveConfig, journal: Journal, state: dict, dry=False):
        from . import strategies
        self.idx, self.cfg, self.live, self.journal, self.dry = idx, cfg, live, journal, dry
        self.params = strategies.get(cfg.strategy)(**cfg.settings).settings       # validated, with defaults
        if cfg.strategy != "OpeningRangeBreakout":
            raise ValueError(f"{cfg.name}: the live trader runs Opening range breakout legs only")
        self.tz, self.open_min, self.close_min = session_of(self.params)
        self.magic = MAGIC_BASE + idx
        self.state = state                                                         # persisted per day
        self._days = {}

    def trading_day(self, local):
        """(open today?, close minute today) from the exchange calendar: holidays are skipped and
        early-close days (e.g. the day after Thanksgiving) close early."""
        d = local.date()
        if d not in self._days:
            code = EXCHANGE_CALENDARS.get(self.params.get("session", "US"))
            try:
                import pandas_market_calendars as pmc
                sched = pmc.get_calendar(code).schedule(str(d), str(d))
            except Exception:                                    # no calendar: assume a normal weekday
                self._days[d] = (local.weekday() < 5, self.close_min)
                return self._days[d]
            if sched.empty:
                self._days[d] = (False, self.close_min)
            else:
                close = sched["market_close"].iloc[0].tz_convert(self.tz)
                self._days[d] = (True, min(self.close_min, close.hour * 60 + close.minute))
        return self._days[d]

    def local_now(self, now_utc: datetime):
        return pd.Timestamp(now_utc).tz_convert(self.tz)

    def day_key(self, now_utc):
        return f"{self.local_now(now_utc).date()}|{self.cfg.name}"

    def step(self, now_utc: datetime, broker):
        local = self.local_now(now_utc)
        if local.weekday() >= 5:
            return
        minute = local.hour * 60 + local.minute
        key = self.day_key(now_utc)
        range_end = self.open_min + int(self.params["range_minutes"])
        is_open, close_min = self.trading_day(local)
        if not is_open:
            if key not in self.state:
                self.state[key] = {"phase": "done"}
                self.journal.write(now_utc, self.cfg.name, "holiday", note=f"the {self.tz} exchange is closed today")
            return
        close_at = close_min - self.live.close_minutes_before
        if key not in self.state and range_end + LATE_START_MINUTES < minute < close_at:
            # started after today's range: the breakout may be over, so sit this day out (like the backtest would)
            self.state[key] = {"phase": "done"}
            self.journal.write(now_utc, self.cfg.name, "too late today",
                               note=f"started after the opening range ({self.tz}); trading starts next session")
            return
        st = self.state.setdefault(key, {"phase": "wait"})
        if st["phase"] == "done":
            return
        if minute >= close_at:                                                     # end of the session
            self.flatten(now_utc, broker, st, "session close")
            return
        if st["phase"] == "wait" and minute >= range_end:
            self.place(now_utc, broker, st, local)
        elif st["phase"] == "placed":
            self.check_fill(now_utc, broker, st)
        elif st["phase"] == "in_trade":
            if not broker.positions(self.cfg.symbol, self.magic):                    # stopped out / target
                self.journal.write(now_utc, self.cfg.name, "trade closed", note="stop or target hit on the server",
                                   equity=round(broker.equity(), 2))
                self.cancel_all(now_utc, broker)
                st["phase"] = "done"

    def place(self, now_utc, broker, st, local):
        start = local.normalize() + pd.Timedelta(minutes=self.open_min)
        end = start + pd.Timedelta(minutes=int(self.params["range_minutes"]))
        bars = broker.bars_m1(self.cfg.symbol, start.tz_convert("UTC"), end.tz_convert("UTC"))
        need = int(self.params["range_minutes"]) * 0.8
        if bars is None or len(bars) < need:
            self.journal.write(now_utc, self.cfg.name, "no range", note=f"{0 if bars is None else len(bars)} one-minute "
                               "candles in the range (market closed or holiday?)")
            st["phase"] = "done"
            return
        hi, lo = float(bars["high"].max()), float(bars["low"].min())
        mid = (hi + lo) / 2
        st.update(hi=hi, lo=lo)
        self.journal.write(now_utc, self.cfg.name, "range", price=f"{lo:g}-{hi:g}",
                           note=f"{len(bars)} candles {start.strftime('%H:%M')}-{end.strftime('%H:%M')} {self.tz}")
        bid, ask = broker.quote(self.cfg.symbol)
        equity = broker.equity()
        spec = broker.spec(self.cfg.symbol)
        rr = float(self.params["rr"])
        sides = [1] + ([-1] if self.params["allow_short"] else [])
        placed = []
        for side in sides:
            level = hi if side > 0 else lo
            stop = mid if self.params["stop_at"] == "middle" else (lo if side > 0 else hi)
            now_px = ask if side > 0 else bid
            beyond = (now_px >= level) if side > 0 else (now_px <= level)
            if beyond and (now_px - stop) * side <= 0:
                self.journal.write(now_utc, self.cfg.name, "skipped", side=side, note="price already past the stop")
                continue
            entry = now_px if beyond else level
            tp = entry + side * rr * abs(entry - stop) if rr > 0 else None
            vol = lots_for(equity, self.cfg.weight, self.effective_multiplier(equity), float(self.params["risk_pct"]),
                           entry, stop, spec)
            if vol <= 0:
                self.journal.write(now_utc, self.cfg.name, "skipped", side=side, note="size below the minimum lot")
                continue
            kind = "market" if beyond else "stop"
            ticket = None if self.dry else broker.send(self.cfg.symbol, kind, side, vol, entry, stop, tp, self.magic,
                                                       f"QL {self.cfg.name}")
            self.journal.write(now_utc, self.cfg.name, ("would " if self.dry else "") + f"{kind} order",
                               side="buy" if side > 0 else "sell", planned=round(entry, 6), sl=round(stop, 6),
                               tp="" if tp is None else round(tp, 6), volume=vol, ticket=ticket, equity=round(equity, 2))
            placed.append(side)
            if beyond and not self.dry:                                            # already in a trade
                st["phase"] = "in_trade"
                self.cancel_all(now_utc, broker)
                return
        st["phase"] = "placed" if placed and not self.dry else "done"
        st["n_orders"] = len(placed)

    def effective_multiplier(self, equity):
        """The plan's size, or the cushion size while the account is down by cushion_drawdown."""
        lv = self.live
        if lv.cushion_multiplier is not None and lv.start_balance and \
                equity <= lv.start_balance * (1 - lv.cushion_drawdown):
            return lv.cushion_multiplier
        return lv.multiplier

    def check_fill(self, now_utc, broker, st):
        pos = broker.positions(self.cfg.symbol, self.magic)
        if not pos and len(broker.orders(self.cfg.symbol, self.magic)) < st.get("n_orders", 0):
            # an order filled AND closed between two checks: still one trade today
            self.journal.write(now_utc, self.cfg.name, "trade closed", note="filled and closed between checks",
                               equity=round(broker.equity(), 2))
            self.cancel_all(now_utc, broker)
            st["phase"] = "done"
            return
        if pos:
            p = pos[0]
            planned = st.get("hi") if p["side"] > 0 else st.get("lo")
            self.journal.write(now_utc, self.cfg.name, "filled", side="buy" if p["side"] > 0 else "sell",
                               price=p["price"], planned=planned, volume=p["volume"], ticket=p["ticket"],
                               note=f"slippage {(p['price'] - planned) * p['side']:+g}" if planned else "")
            self.cancel_all(now_utc, broker)                                       # one trade a day (OCO)
            st["phase"] = "in_trade"

    def cancel_all(self, now_utc, broker):
        for o in broker.orders(self.cfg.symbol, self.magic):
            broker.cancel(o["ticket"])
            self.journal.write(now_utc, self.cfg.name, "cancelled", ticket=o["ticket"])

    def flatten(self, now_utc, broker, st, why):
        if not self.dry:
            self.cancel_all(now_utc, broker)
            for p in broker.positions(self.cfg.symbol, self.magic):
                px = broker.close(p)
                self.journal.write(now_utc, self.cfg.name, "closed", price=px, ticket=p["ticket"], note=why,
                                   equity=round(broker.equity(), 2))
        st["phase"] = "done"


# ---- The trader ----------------------------------------------------------------------------

class Trader:
    def __init__(self, cfg: LiveConfig, broker, dry=False, folder: Path = None, echo=True):
        self.cfg, self.broker, self.dry = cfg, broker, dry
        self.dir = Path(folder or live_dir())
        self.journal = Journal(self.dir / "journal.csv", echo=echo)
        self.state_path = self.dir / "state.json"
        try:
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.state = {}
        self.legs = [LegRunner(i, leg, cfg, self.journal, self.state.setdefault("legs", {}), dry)
                     for i, leg in enumerate(cfg.legs)]

    def save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        legs = self.state.get("legs", {})
        recent = sorted(legs)[-200:]                                               # keep the file small
        self.state["legs"] = {k: legs[k] for k in recent}
        for leg in self.legs:
            leg.state = self.state["legs"]
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def check_account(self):
        acc = self.broker.account()
        if acc["mode"] == "real" and not self.cfg.allow_real:
            raise SystemExit("This MT5 account is a REAL-money account. The trader refuses it unless "
                             '"allow_real": true is set in live.json.')
        if self.cfg.start_balance is None:
            self.cfg.start_balance = acc["balance"]
        return acc

    def guards(self, now_utc) -> bool:
        """True = keep trading. Daily guard pauses until tomorrow; total guard stops for good."""
        equity = self.broker.equity()
        day = pd.Timestamp(now_utc).tz_convert("Europe/Prague").strftime("%Y-%m-%d")   # FTMO's day (CE(S)T)
        starts = self.state.setdefault("day_start_equity", {})
        if day not in starts:
            starts[day] = equity
        if equity <= self.cfg.start_balance * (1 - self.cfg.total_loss_guard):
            self.halt(now_utc, f"total loss guard: equity {equity:.2f} is {self.cfg.total_loss_guard:.1%} "
                               f"below the starting balance {self.cfg.start_balance:.2f}")
            self.state["halted"] = True
            return False
        if equity <= starts[day] * (1 - self.cfg.daily_loss_guard):
            if not self.state.get("paused") == day:
                self.halt(now_utc, f"daily loss guard: equity {equity:.2f} down {self.cfg.daily_loss_guard:.1%} today")
                self.state["paused"] = day
            return False
        return not self.state.get("halted")

    def halt(self, now_utc, why):
        self.journal.write(now_utc, "ALL", "guard", note=why, equity=round(self.broker.equity(), 2))
        for leg in self.legs:
            st = leg.state.setdefault(leg.day_key(now_utc), {"phase": "wait"})
            leg.flatten(now_utc, self.broker, st, "guard")
            # mark every leg done for today, whatever its session
            st["phase"] = "done"

    def step(self, now_utc: datetime = None):
        now_utc = now_utc or datetime.now(timezone.utc)
        if self.guards(now_utc):
            for leg in self.legs:
                try:
                    leg.step(now_utc, self.broker)
                except Exception as e:                                             # one leg's problem stops only it
                    self.journal.write(now_utc, leg.cfg.name, "error", note=f"{type(e).__name__}: {e}")
                    leg.state.setdefault(leg.day_key(now_utc), {})["phase"] = "done"
        self.save()

    def run(self):
        acc = self.check_account()
        self.journal.write(datetime.now(timezone.utc), "ALL", "start",
                           note=f"{acc['server']} ({acc['mode']}), {'DRY RUN' if self.dry else 'trading'}, "
                                f"{len(self.legs)} legs, x{self.cfg.multiplier}", equity=round(acc["equity"], 2))
        stop_file = self.dir / "STOP"
        try:
            while not stop_file.exists():
                self.step()
                if self.state.get("halted"):
                    break
                time.sleep(self.cfg.poll_seconds)
        except KeyboardInterrupt:
            pass
        self.journal.write(datetime.now(timezone.utc), "ALL", "stopped",
                           note="open trades keep their stop-loss on the server")


# ---- MT5 broker ----------------------------------------------------------------------------

class MT5Broker:
    """The real thing: talks to the MT5 terminal through the MetaTrader5 package."""

    def __init__(self, path=None):
        from . import mt5 as M
        self.M = M
        self.mt5, self.info = M.connect(path)
        self.offsets = {}

    def account(self):
        a = self.mt5.account_info()
        mode = {0: "demo", 1: "contest", 2: "real"}.get(getattr(a, "trade_mode", 2), "real")
        return {"server": a.server, "mode": mode, "balance": float(a.balance), "equity": float(a.equity),
                "currency": a.currency}

    def equity(self):
        return float(self.mt5.account_info().equity)

    def spec(self, symbol):
        s = self.mt5.symbol_info(symbol)
        return {"tick_value": float(s.trade_tick_value), "tick_size": float(s.trade_tick_size),
                "volume_min": float(s.volume_min), "volume_max": float(s.volume_max),
                "volume_step": float(s.volume_step), "digits": int(s.digits), "point": float(s.point),
                "stops_level": int(getattr(s, "trade_stops_level", 0)), "filling": int(getattr(s, "filling_mode", 0))}

    def quote(self, symbol):
        self.mt5.symbol_select(symbol, True)
        t = self.mt5.symbol_info_tick(symbol)
        return float(t.bid), float(t.ask)

    def bars_m1(self, symbol, start_utc, end_utc):
        """1-minute candles whose New York time falls in [start, end)."""
        if symbol not in self.offsets:
            self.offsets[symbol] = self.M.ny_offset_hours(self.mt5, symbol)
        rates = self.mt5.copy_rates_from_pos(symbol, self.mt5.TIMEFRAME_M1, 0, 1500)
        df = self.M.to_frame(rates, float(self.mt5.symbol_info(symbol).point), self.offsets[symbol], daily=False)
        if df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        idx = df.index.tz_convert("UTC")
        return df[(idx >= start_utc) & (idx < end_utc)]

    def _filling(self, symbol):
        f = int(getattr(self.mt5.symbol_info(symbol), "filling_mode", 0))
        if f & 2:
            return self.mt5.ORDER_FILLING_IOC
        if f & 1:
            return self.mt5.ORDER_FILLING_FOK
        return self.mt5.ORDER_FILLING_RETURN

    def send(self, symbol, kind, side, volume, price, sl, tp, magic, comment):
        m, spec = self.mt5, self.spec(symbol)
        rnd = lambda x: None if x is None else round(x, spec["digits"])
        req = {"symbol": symbol, "volume": float(volume), "sl": rnd(sl), "magic": magic, "comment": comment[:30],
               "deviation": 20}
        if tp is not None:
            req["tp"] = rnd(tp)
        if kind == "market":
            bid, ask = self.quote(symbol)
            req.update(action=m.TRADE_ACTION_DEAL, type=m.ORDER_TYPE_BUY if side > 0 else m.ORDER_TYPE_SELL,
                       price=ask if side > 0 else bid, type_filling=self._filling(symbol))
        else:
            req.update(action=m.TRADE_ACTION_PENDING, price=rnd(price),
                       type=m.ORDER_TYPE_BUY_STOP if side > 0 else m.ORDER_TYPE_SELL_STOP,
                       type_time=m.ORDER_TIME_GTC, type_filling=m.ORDER_FILLING_RETURN)
        r = m.order_send(req)
        if r is None or r.retcode not in (m.TRADE_RETCODE_DONE, m.TRADE_RETCODE_PLACED):
            code = None if r is None else r.retcode
            hint = " (is the Algo Trading button in MT5 ON?)" if code == 10027 else ""
            raise RuntimeError(f"order refused: retcode {code} {getattr(r, 'comment', m.last_error())}{hint}")
        return int(r.order or r.deal)

    def positions(self, symbol, magic):
        ps = self.mt5.positions_get(symbol=symbol) or []
        return [{"ticket": p.ticket, "side": 1 if p.type == 0 else -1, "volume": p.volume, "price": p.price_open}
                for p in ps if p.magic == magic]

    def orders(self, symbol, magic):
        os_ = self.mt5.orders_get(symbol=symbol) or []
        return [{"ticket": o.ticket} for o in os_ if o.magic == magic]

    def cancel(self, ticket):
        self.mt5.order_send({"action": self.mt5.TRADE_ACTION_REMOVE, "order": ticket})

    def close(self, pos):
        p = next(x for x in (self.mt5.positions_get(ticket=pos["ticket"]) or []))
        bid, ask = self.quote(p.symbol)
        m = self.mt5
        r = m.order_send({"action": m.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume, "position": p.ticket,
                          "type": m.ORDER_TYPE_SELL if p.type == 0 else m.ORDER_TYPE_BUY,
                          "price": bid if p.type == 0 else ask, "deviation": 20, "magic": p.magic,
                          "type_filling": self._filling(p.symbol)})
        return getattr(r, "price", None)


# ---- Command line --------------------------------------------------------------------------

def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else ""
    cfg_path = live_dir() / "live.json"
    if "--config" in args:
        cfg_path = Path(args[args.index("--config") + 1])
    if cmd == "init":
        if cfg_path.exists() and "--force" not in args:
            print(f"  {cfg_path} already exists (add --force to overwrite).")
            return
        example_config().save(cfg_path)
        print(f"  Wrote {cfg_path}. Check the symbol names (python -m quantlab.mt5 list cash), the weights and\n"
              f"  'multiplier' (your plan's size), then: python -m quantlab.live run --dry-run")
    elif cmd == "run":
        if not cfg_path.exists():
            print("  No live.json yet: run  python -m quantlab.live init  first.")
            return
        cfg = LiveConfig.load(cfg_path)
        path = args[args.index("--path") + 1] if "--path" in args else cfg.terminal_path
        (live_dir() / "STOP").unlink(missing_ok=True)
        trader = Trader(cfg, MT5Broker(path), dry="--dry-run" in args)
        trader.run()
        cfg.save(cfg_path)                                                         # keeps start_balance
    elif cmd == "status":
        j = live_dir() / "journal.csv"
        if not j.exists():
            print("  No journal yet.")
            return
        df = pd.read_csv(j)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        print(df[df["time_utc"].str.startswith(today)].to_string(index=False) or "  Nothing today yet.")
    elif cmd == "stop":
        live_dir().mkdir(parents=True, exist_ok=True)
        (live_dir() / "STOP").write_text("stop", encoding="utf-8")
        print("  The running trader will stop within a few seconds (open trades keep their stop-loss).")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
