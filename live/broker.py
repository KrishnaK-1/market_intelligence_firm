"""
Thin wrapper around the alpaca-py SDK. Everything the executor needs and
nothing else: account, positions, quotes, market orders, fill polling.
"""
import time
from dataclasses import dataclass, field

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

from live.config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER, ORDER_FILL_TIMEOUT,
)

TERMINAL = {"filled", "canceled", "expired", "rejected"}


@dataclass
class TradeRecord:
    symbol: str
    side: str
    requested_notional: float
    bid_at_submit: float = None
    ask_at_submit: float = None
    status: str = "not_submitted"
    filled_qty: float = 0.0
    filled_avg_price: float = None
    filled_notional: float = None
    submitted_at: str = None
    filled_at: str = None
    latency_sec: float = None
    error: str = None
    order_id: str = None
    extra: dict = field(default_factory=dict)


class Broker:
    def __init__(self):
        self.trading = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
        self.data = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        self._fractionable = {}

    # ── account state ────────────────────────────────────────────────
    def account(self) -> dict:
        a = self.trading.get_account()
        return {"equity": float(a.equity), "cash": float(a.cash),
                "buying_power": float(a.buying_power),
                "account_number": a.account_number, "status": str(a.status)}

    def positions(self) -> dict:
        """{symbol: {qty, market_value}} for all open positions."""
        return {p.symbol: {"qty": float(p.qty), "market_value": float(p.market_value)}
                for p in self.trading.get_all_positions()}

    def market_tradable_today(self) -> bool:
        """True if the market is open now or opens later today — i.e. a DAY
        order submitted now will execute today rather than queue overnight."""
        clock = self.trading.get_clock()
        return bool(clock.is_open) or clock.next_open.date() == clock.timestamp.date()

    def is_fractionable(self, symbol: str) -> bool:
        if symbol not in self._fractionable:
            self._fractionable[symbol] = bool(self.trading.get_asset(symbol).fractionable)
        return self._fractionable[symbol]

    def quotes(self, symbols: list) -> dict:
        """{symbol: (bid, ask)} — free plan serves IEX quotes, which is fine
        for a daily log; treat spreads as indicative, not consolidated NBBO."""
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            q = self.data.get_stock_latest_quote(req)
            return {s: (float(q[s].bid_price), float(q[s].ask_price)) for s in q}
        except Exception:
            return {}

    # ── orders ───────────────────────────────────────────────────────
    def execute_notional(self, symbol: str, notional: float, side: str,
                         last_price: float = None) -> TradeRecord:
        """Submit a market order for ~|notional| dollars and wait for the fill.
        Uses a notional (dollar) order when the asset supports fractional
        shares; otherwise falls back to whole shares. Never shorts: a sell is
        capped at the current position by the caller."""
        rec = TradeRecord(symbol=symbol, side=side, requested_notional=round(abs(notional), 2))
        bidask = self.quotes([symbol]).get(symbol)
        if bidask:
            rec.bid_at_submit, rec.ask_at_submit = bidask

        try:
            kwargs = dict(symbol=symbol,
                          side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                          time_in_force=TimeInForce.DAY)
            if self.is_fractionable(symbol):
                kwargs["notional"] = round(abs(notional), 2)
            else:
                ref = last_price or (bidask and (bidask[0] + bidask[1]) / 2)
                if not ref:
                    rec.status, rec.error = "skipped", "no price for whole-share sizing"
                    return rec
                qty = int(abs(notional) / ref)
                if qty == 0:
                    rec.status, rec.error = "skipped", "notional below one whole share"
                    return rec
                kwargs["qty"] = qty
            order = self.trading.submit_order(MarketOrderRequest(**kwargs))
            rec.order_id = str(order.id)
            rec.submitted_at = str(order.submitted_at)

            deadline = time.time() + ORDER_FILL_TIMEOUT
            while time.time() < deadline:
                order = self.trading.get_order_by_id(order.id)
                if str(order.status).split(".")[-1].lower() in TERMINAL:
                    break
                time.sleep(2)

            rec.status = str(order.status).split(".")[-1].lower()
            rec.filled_qty = float(order.filled_qty or 0)
            if order.filled_avg_price is not None:
                rec.filled_avg_price = float(order.filled_avg_price)
                rec.filled_notional = round(rec.filled_qty * rec.filled_avg_price, 2)
            if order.filled_at is not None and order.submitted_at is not None:
                rec.filled_at = str(order.filled_at)
                rec.latency_sec = round(
                    (order.filled_at - order.submitted_at).total_seconds(), 3)
        except Exception as e:
            rec.status, rec.error = "error", str(e)
        return rec
