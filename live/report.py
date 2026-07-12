"""
Daily trading report: one Excel workbook per run in live/reports/, plus a
one-line append to master_log.csv so the whole history is greppable.

Sheets:
  Trades    — every order: side, requested $, fill qty/price/$, bid/ask at
              submit, spread, submit->fill latency, status, errors
  Positions — end-of-run holdings vs targets
  Targets   — today's full target vector with per-desk gross exposure
  Account   — equity, cash, mode, band decision, data as-of date
"""
import csv
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from live.config import REPORTS_DIR, ALPACA_PAPER, TRADE_BAND


def write_report(run: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    path = REPORTS_DIR / f"trading_report_{stamp}.xlsx"

    trades = pd.DataFrame([asdict(r) for r in run.get("records", [])])
    if len(trades):
        trades["spread"] = trades["ask_at_submit"] - trades["bid_at_submit"]
        trades = trades[[
            "symbol", "side", "status", "requested_notional", "filled_qty",
            "filled_avg_price", "filled_notional", "bid_at_submit",
            "ask_at_submit", "spread", "submitted_at", "filled_at",
            "latency_sec", "order_id", "error",
        ]]
    else:
        trades = pd.DataFrame([{"note": run.get("no_trade_reason", "no orders today")}])

    positions = pd.DataFrame([
        {"symbol": s, "qty": p["qty"], "market_value": p["market_value"],
         "weight": p["market_value"] / run["account"]["equity"] if run["account"]["equity"] else None,
         "target_weight": run["targets"]["weights"].get(s, 0.0)}
        for s, p in sorted(run.get("positions_after", {}).items())
    ]) if run.get("positions_after") else pd.DataFrame([{"note": "no open positions"}])

    targets = pd.DataFrame(
        sorted(run["targets"]["weights"].items(), key=lambda x: -x[1]),
        columns=["symbol", "target_weight"],
    )
    targets.loc[len(targets)] = ["CASH (T-bill leg)", run["targets"]["cash_weight"]]

    account = pd.DataFrame([{
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "PAPER" if ALPACA_PAPER else "LIVE",
        "signals_as_of_close": run["targets"]["asof"],
        "equity": run["account"]["equity"],
        "cash": run["account"]["cash"],
        "drift_vs_target": run["plan"]["drift"],
        "trade_band": TRADE_BAND,
        "rebalanced": run["plan"]["trade"],
        "orders_placed": len(run.get("records", [])),
        "orders_filled": sum(1 for r in run.get("records", []) if r.status == "filled"),
        **{f"desk_gross_{k}": v for k, v in run["targets"]["desk_gross_exposure"].items()},
    }]).T.reset_index()
    account.columns = ["field", "value"]

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        trades.to_excel(xl, sheet_name="Trades", index=False)
        positions.to_excel(xl, sheet_name="Positions", index=False)
        targets.to_excel(xl, sheet_name="Targets", index=False)
        account.to_excel(xl, sheet_name="Account", index=False)

    _append_master(run)
    return path


def _append_master(run: dict):
    master = REPORTS_DIR / "master_log.csv"
    row = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "mode": "PAPER" if ALPACA_PAPER else "LIVE",
        "equity": run["account"]["equity"],
        "drift": round(run["plan"]["drift"], 4) if run["plan"]["drift"] is not None else "",
        "rebalanced": run["plan"]["trade"],
        "orders": len(run.get("records", [])),
        "filled": sum(1 for r in run.get("records", []) if r.status == "filled"),
        "errors": sum(1 for r in run.get("records", []) if r.status == "error"),
    }
    new = not master.exists()
    with open(master, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
