"""Fill report: what do the robot's trades REALLY cost on the broker?

    python -m quantlab.fills                    # robot trades of the last 90 days
    python -m quantlab.fills --days 30
    python -m quantlab.fills --magic 7700123    # only this magic number

Reads the trade history from the open MT5 terminal (orders and deals of QuantLab's robots:
the MQL5 Expert Advisor and the Python live trader) and, for every fill, compares the price
the robot ASKED for with the price it GOT:
  * entry by stop order : the stop's trigger price;
  * exit at stop loss   : the stop-loss level;
  * entry / exit at market (e.g. the end-of-day close): the price sent with the order.
Slippage is in bp (1/100 of a percent) of the price; positive = worse for us.
Cost per round trip = spread (median half-spread of the symbol, paid on entry and exit)
+ entry slippage + exit slippage + commission. Compared with the backtest's assumption
from  python -m quantlab.ftmo_orb  (data/edges/ftmo_orb.csv) when available.
Results are saved in data/fills/fills.csv.
"""
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import download as dl

EA_MAGICS = range(7_700_000, 7_704_000)        # mql5/QuantLabORB.mq5 (automatic magic numbers)
LIVE_MAGICS = range(2_611_000, 2_612_000)      # quantlab/live.py
# MT5 constants (numbers, so the report works without the MetaTrader5 package in tests)
DEAL_BUY, DEAL_SELL = 0, 1
ENTRY_IN, ENTRY_OUT = 0, 1
REASONS = {0: "manual", 1: "mobile", 2: "web", 3: "robot", 4: "stop loss", 5: "take profit", 6: "stop out"}
ORDER_TYPES = {0: "market", 1: "market", 2: "limit", 3: "limit", 4: "stop", 5: "stop"}


def is_ours(magic: int, magics=None) -> bool:
    if magics:
        return magic in magics
    return magic in EA_MAGICS or magic in LIVE_MAGICS


def round_trips(orders, deals, magics=None) -> pd.DataFrame:
    """One row per closed position of our robots, with asked vs got prices."""
    orders = {o.ticket: o for o in orders}
    by_pos = {}
    for d in deals:
        if d.position_id:
            by_pos.setdefault(d.position_id, []).append(d)
    rows = []
    for pid, ds in by_pos.items():
        ins = [d for d in ds if d.entry == ENTRY_IN]
        outs = [d for d in ds if d.entry == ENTRY_OUT]
        if not ins or not outs or not is_ours(ins[0].magic, magics):
            continue
        d_in, d_out = ins[0], outs[-1]
        side = 1 if d_in.type == DEAL_BUY else -1
        o_in, o_out = orders.get(d_in.order), orders.get(d_out.order)
        entry_kind = ORDER_TYPES.get(getattr(o_in, "type", 0), "market")
        entry_asked = getattr(o_in, "price_open", 0.0) or np.nan
        exit_reason = REASONS.get(d_out.reason, str(d_out.reason))
        if exit_reason == "stop loss":
            exit_asked = getattr(o_in, "sl", 0.0) or getattr(o_out, "price_open", 0.0) or np.nan
        elif exit_reason == "take profit":
            exit_asked = getattr(o_in, "tp", 0.0) or getattr(o_out, "price_open", 0.0) or np.nan
        else:
            exit_asked = getattr(o_out, "price_open", 0.0) or np.nan
        # slippage: positive = worse. Buying higher / selling lower than asked is worse.
        slip_in = side * (d_in.price - entry_asked) / entry_asked * 1e4
        slip_out = -side * (d_out.price - exit_asked) / exit_asked * 1e4
        volume = sum(d.volume for d in ins)
        rows.append({
            "symbol": d_in.symbol, "magic": d_in.magic, "side": "long" if side > 0 else "short",
            "opened": datetime.fromtimestamp(d_in.time, timezone.utc), "closed": datetime.fromtimestamp(d_out.time, timezone.utc),
            "volume": volume, "entry by": entry_kind, "entry asked": entry_asked, "entry got": d_in.price,
            "exit by": exit_reason, "exit asked": exit_asked, "exit got": d_out.price,
            "entry slip bp": slip_in, "exit slip bp": slip_out,
            "result bp": side * (d_out.price / d_in.price - 1) * 1e4,
            "profit": sum(d.profit for d in ds), "commission": sum(d.commission + getattr(d, "fee", 0.0) for d in ds),
            "swap": sum(d.swap for d in ds),
        })
    return pd.DataFrame(rows)


def notional(mt5, symbol: str, volume: float, price: float) -> float:
    """Position value in account currency (same formula as the robot's sizing)."""
    info = mt5.symbol_info(symbol) if mt5 else None
    tv, ts = getattr(info, "trade_tick_value", 0.0), getattr(info, "trade_tick_size", 0.0)
    return volume * price * tv / ts if tv and ts else np.nan


def summary(trips: pd.DataFrame, half_spreads: dict, assumed: dict = None) -> pd.DataFrame:
    """Per symbol: trades, average slippage, commission and the full cost per round trip, in bp."""
    rows = []
    for sym, t in trips.groupby("symbol"):
        half = half_spreads.get(sym, np.nan)
        comm = (-t["commission"] / t["notional"] * 1e4).mean() if "notional" in t else np.nan
        stops = t[t["exit by"] == "stop loss"]
        others = t[t["exit by"] != "stop loss"]
        slip_in, slip_out = t["entry slip bp"].mean(), t["exit slip bp"].mean()
        total = 2 * half + slip_in + slip_out + (comm if comm == comm else 0.0)
        rows.append({"symbol": sym, "trades": len(t), "half spread bp": half,
                     "entry slip bp": slip_in, "worst entry slip bp": t["entry slip bp"].max(),
                     "exit slip at stop bp": stops["exit slip bp"].mean() if len(stops) else np.nan,
                     "exit slip other bp": others["exit slip bp"].mean() if len(others) else np.nan,
                     "commission bp": comm, "cost per trip bp": total,
                     "backtest assumed bp": (assumed or {}).get(sym, np.nan)})
    return pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()


def assumed_costs() -> dict:
    """The '2% slippage' cost per trade from python -m quantlab.ftmo_orb, if it was run."""
    path = dl.DATA_DIR / "edges" / "ftmo_orb.csv"
    try:
        t = pd.read_csv(path, index_col="market")
        return t["cost bp 2%"].dropna().to_dict()
    except (OSError, KeyError, ValueError):
        return {}


def half_spread_bps(sym: str, mt5=None) -> float:
    from . import mt5 as M
    s = M.saved_specs(sym)
    if s.get("spread_rth_bps"):
        return s["spread_rth_bps"] / 2
    if mt5 is not None:
        try:
            return (M.specs(sym, mt5).get("spread_rth_bps") or np.nan) / 2
        except Exception:
            pass
    return np.nan


def report(days: int = 90, magics=None):
    from . import mt5 as M
    mt5, info = M.connect()
    end = datetime.now(timezone.utc) + timedelta(days=1)
    start = end - timedelta(days=days + 1)
    orders = mt5.history_orders_get(start, end) or ()
    deals = mt5.history_deals_get(start, end) or ()
    trips = round_trips(orders, deals, magics)
    print(f"  {info['company']} · {info['server']} · last {days} days: {len(deals)} deals, "
          f"{len(trips)} closed robot trades")
    if trips.empty:
        print("  No closed trades from QuantLab's robots yet (magic numbers 7700000-7703999 and 2611000-2611999).")
        return trips, pd.DataFrame()
    trips["notional"] = [notional(mt5, s, v, p) for s, v, p in zip(trips["symbol"], trips["volume"], trips["entry got"])]
    halves = {s: half_spread_bps(s, mt5) for s in trips["symbol"].unique()}
    return trips, summary(trips, halves, assumed_costs())


def show(trips: pd.DataFrame, summ: pd.DataFrame) -> str:
    num = lambda v: "" if v != v else f"{v:.2f}"
    out = ["Cost per round trip, bp of the position (positive slippage = worse than asked):"]
    t = summ.copy().astype(object)
    for c in summ.columns:
        t[c] = [f"{int(v)}" if c == "trades" else num(v) for v in summ[c]]
    out.append(t.to_string())
    if len(trips) and len(trips) < 30:
        out.append("  (fewer than 30 trades: averages are still rough)")
    last = trips.sort_values("closed").tail(10)[["symbol", "side", "entry by", "entry slip bp", "exit by",
                                                 "exit slip bp", "result bp"]]
    out += ["", "Last 10 trades:", last.to_string(index=False, float_format=lambda v: f"{v:.2f}")]
    return "\n".join(out)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return
    days = int(args[args.index("--days") + 1]) if "--days" in args else 90
    magics = [int(args[args.index("--magic") + 1])] if "--magic" in args else None
    try:
        trips, summ = report(days, magics)
    except Exception as e:
        print(f"  {e}")
        return
    if trips.empty:
        return
    print()
    print(show(trips, summ))
    path = dl.DATA_DIR / "fills" / "fills.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    trips.to_csv(path, index=False)
    print(f"\n  Every trade saved in {path}")


if __name__ == "__main__":
    main()
