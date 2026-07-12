"""
Turns target weights into Alpaca orders.

Logic mirrors the backtest's accounting: compare target dollar allocations
against ACTUAL account positions, skip the rebalance entirely when total
drift is inside the no-trade band, otherwise trade the differences —
sells first (frees cash), then buys. Never shorts: sells are capped at the
existing position; anything held that isn't in the target gets fully exited.
"""
import time

from live.broker import Broker, TradeRecord
from live.config import TRADE_BAND, MIN_ORDER_NOTIONAL


def plan_orders(targets: dict, positions: dict, equity: float) -> dict:
    """Compute per-ticker dollar deltas. Returns dict with the plan and the
    band decision. Positive delta = buy, negative = sell."""
    current = {s: p["market_value"] for s, p in positions.items()}
    symbols = sorted(set(targets) | set(current))
    deltas = {}
    for s in symbols:
        target_dollars = targets.get(s, 0.0) * equity
        deltas[s] = round(target_dollars - current.get(s, 0.0), 2)

    drift = sum(abs(d) for d in deltas.values()) / equity if equity > 0 else 0.0
    trade = drift > TRADE_BAND
    orders = []
    if trade:
        sells = [(s, d) for s, d in deltas.items() if d < -MIN_ORDER_NOTIONAL]
        buys = [(s, d) for s, d in deltas.items() if d > MIN_ORDER_NOTIONAL]
        # sells first: the cash they free up funds the buys
        orders = [(s, d, "sell") for s, d in sorted(sells, key=lambda x: x[1])] + \
                 [(s, d, "buy") for s, d in sorted(buys, key=lambda x: -x[1])]
    return {"deltas": deltas, "drift": drift, "trade": trade, "orders": orders}


def execute_plan(broker: Broker, plan: dict, positions: dict) -> list:
    """Run the planned orders through Alpaca. Returns a list of TradeRecord."""
    records = []
    for symbol, delta, side in plan["orders"]:
        notional = abs(delta)
        if side == "sell":
            held = positions.get(symbol, {}).get("market_value", 0.0)
            qty_held = positions.get(symbol, {}).get("qty", 0.0)
            # cap the sell at the position (never short); if we're exiting
            # ~the whole position, close it by quantity to avoid leaving a
            # fractional dust position behind
            if notional >= held * 0.98:
                rec = _close_position(broker, symbol, qty_held, held)
                records.append(rec)
                continue
            notional = min(notional, held)
        last_price = None
        if positions.get(symbol, {}).get("qty"):
            last_price = positions[symbol]["market_value"] / positions[symbol]["qty"]
        records.append(broker.execute_notional(symbol, notional, side, last_price=last_price))
        time.sleep(0.3)   # stay well under API rate limits
    return records


def _close_position(broker: Broker, symbol: str, qty: float, held_value: float) -> TradeRecord:
    rec = TradeRecord(symbol=symbol, side="sell", requested_notional=round(held_value, 2))
    bidask = broker.quotes([symbol]).get(symbol)
    if bidask:
        rec.bid_at_submit, rec.ask_at_submit = bidask
    try:
        order = broker.trading.close_position(symbol)
        rec.order_id = str(order.id)
        rec.submitted_at = str(order.submitted_at)
        import time as _t
        from live.config import ORDER_FILL_TIMEOUT
        from live.broker import TERMINAL
        deadline = _t.time() + ORDER_FILL_TIMEOUT
        while _t.time() < deadline:
            order = broker.trading.get_order_by_id(order.id)
            if str(order.status).split(".")[-1].lower() in TERMINAL:
                break
            _t.sleep(2)
        rec.status = str(order.status).split(".")[-1].lower()
        rec.filled_qty = float(order.filled_qty or 0)
        if order.filled_avg_price is not None:
            rec.filled_avg_price = float(order.filled_avg_price)
            rec.filled_notional = round(rec.filled_qty * rec.filled_avg_price, 2)
        if order.filled_at is not None and order.submitted_at is not None:
            rec.filled_at = str(order.filled_at)
            rec.latency_sec = round((order.filled_at - order.submitted_at).total_seconds(), 3)
        rec.extra["closed_full_position"] = True
    except Exception as e:
        rec.status, rec.error = "error", str(e)
    return rec
