"""
Daily entrypoint — schedule this once per trading day (morning, e.g. 9:35 ET).

    python live/run_daily.py               normal run: refresh -> targets ->
                                           trade via Alpaca -> Excel report
    python live/run_daily.py --dry-run     compute targets and write the
                                           report, but place NO orders and
                                           never touch Alpaca (no keys needed)

Pipeline: refresh price caches (data through yesterday's close, a partial
bar for today can never leak in) -> run the SAME engine as the backtest ->
compare targets against actual account positions -> trade only if drift
exceeds the no-trade band -> write live/reports/trading_report_<date>.xlsx.
"""
import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live.config import require_keys, ALPACA_PAPER, TRADE_BAND
from live.targets import refresh_caches, compute_targets
from live.report import write_report


def main(dry_run: bool = False):
    print(f"=== Daily rebalance {'(DRY RUN)' if dry_run else ''} "
          f"[{'PAPER' if ALPACA_PAPER else 'LIVE'}] ===")

    print("1/4 Refreshing price data...")
    refresh_caches()

    print("2/4 Computing target weights (full engine run, ~2-4 min)...")
    targets = compute_targets()
    gross = sum(targets["weights"].values())
    print(f"    signals as of close {targets['asof']} | gross exposure "
          f"{gross:.1%} | cash {targets['cash_weight']:.1%} | "
          f"{len(targets['weights'])} tickers")

    if dry_run:
        run = {"targets": targets, "account": {"equity": 100_000.0, "cash": 100_000.0},
               "plan": {"drift": None, "trade": False, "orders": [], "deltas": {}},
               "records": [], "positions_after": {},
               "no_trade_reason": "dry run — no orders placed"}
        path = write_report(run)
        print(f"4/4 DRY RUN report -> {path}")
        return

    require_keys()
    from live.broker import Broker
    from live.execute import plan_orders, execute_plan

    broker = Broker()
    if not broker.market_tradable_today():
        print("Market is closed today (holiday/weekend) — nothing to do.")
        return

    account = broker.account()
    positions = broker.positions()
    print(f"    account equity ${account['equity']:,.2f} | "
          f"{len(positions)} open positions")

    print("3/4 Planning and executing orders...")
    plan = plan_orders(targets["weights"], positions, account["equity"])
    records = []
    if plan["trade"]:
        records = execute_plan(broker, plan, positions)
        filled = sum(1 for r in records if r.status == "filled")
        errors = [r for r in records if r.status == "error"]
        print(f"    {len(records)} orders: {filled} filled, {len(errors)} errors")
        for r in errors:
            print(f"      ERROR {r.symbol}: {r.error}")
    else:
        print(f"    drift {plan['drift']:.2%} is inside the {TRADE_BAND:.0%} band — no trades")

    run = {
        "targets": targets, "account": broker.account(), "plan": plan,
        "records": records, "positions_after": broker.positions(),
        "no_trade_reason": (None if plan["trade"] else
                            f"drift {plan['drift']:.2%} inside no-trade band"),
    }
    path = write_report(run)
    print(f"4/4 Report -> {path}")


if __name__ == "__main__":
    try:
        main(dry_run="--dry-run" in sys.argv)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
