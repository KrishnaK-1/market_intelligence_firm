"""
Daily entrypoint — schedule this once per trading day (morning, e.g. 9:35 ET).

    python live/run_daily.py               normal run: refresh -> targets ->
                                           trade via Alpaca -> Excel report
    python live/run_daily.py --dry-run     compute targets and write the
                                           report, but place NO orders and
                                           never touch Alpaca (no keys needed).
                                           Reports are labeled DRY-RUN and use
                                           a placeholder $100k — they are NOT
                                           real account data.

Every run appends to live/reports/run.log, so a scheduled run that fails or
does nothing still leaves a trace you can read afterwards.

Pipeline: refresh price caches (data through yesterday's close, a partial
bar for today can never leak in) -> run the SAME engine as the backtest ->
compare targets against actual account positions -> trade only if drift
exceeds the no-trade band AND the market is open -> write the Excel report.
"""
import sys
import logging
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live.config import (
    require_keys, ALPACA_PAPER, TRADE_BAND, REPORTS_DIR, MAX_DAILY_TURNOVER,
)
from live.targets import refresh_caches, compute_targets
from live.report import write_report


def _log():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("live")
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for h in (logging.FileHandler(REPORTS_DIR / "run.log", encoding="utf-8"),
                  logging.StreamHandler()):
            h.setFormatter(fmt)
            log.addHandler(h)
    return log


log = _log()


def main(dry_run: bool = False):
    mode_label = "DRY-RUN" if dry_run else ("PAPER" if ALPACA_PAPER else "LIVE")
    log.info(f"===== Daily rebalance [{mode_label}] =====")

    log.info("1/4 Refreshing price data...")
    refresh_caches(verbose=False)

    log.info("2/4 Computing target weights (full engine run, ~2-4 min)...")
    targets = compute_targets()
    gross = sum(targets["weights"].values())
    log.info(f"    signals as of close {targets['asof']} | gross exposure {gross:.1%} | "
             f"cash {targets['cash_weight']:.1%} | {len(targets['weights'])} tickers")

    if dry_run:
        run = {"mode_label": mode_label, "targets": targets,
               "account": {"equity": 100_000.0, "cash": 100_000.0},
               "plan": {"drift": None, "trade": False, "orders": [], "deltas": {}},
               "records": [], "positions_after": {},
               "no_trade_reason": "DRY RUN — placeholder account, no orders placed"}
        log.info(f"4/4 DRY RUN report -> {write_report(run)}")
        return

    require_keys()
    from live.broker import Broker
    from live.execute import plan_orders, execute_plan

    broker = Broker()
    if not broker.market_tradable_today():
        log.info("Market closed today (weekend/holiday) — nothing to do.")
        return
    market_open = bool(broker.trading.get_clock().is_open)

    account = broker.account()
    positions = broker.positions()
    log.info(f"    account equity ${account['equity']:,.2f} | "
             f"{len(positions)} open positions | market_open={market_open}")

    log.info("3/4 Planning orders...")
    plan = plan_orders(targets["weights"], positions, account["equity"])
    records, deferred = [], False

    if plan.get("blocked"):
        log.error(f"    CIRCUIT BREAKER: plan wants to move {plan['drift']:.1%} of the book "
                  f"(limit {MAX_DAILY_TURNOVER:.0%}) — NO trades placed. This usually means "
                  f"bad price data or a mispriced account. Inspect today's report, then "
                  f"re-run once the cause is understood.")
    elif plan["trade"] and market_open:
        records = execute_plan(broker, plan, positions)
        filled = sum(1 for r in records if r.status == "filled")
        errors = [r for r in records if r.status == "error"]
        log.info(f"    {len(records)} orders: {filled} filled, {len(errors)} errors")
        for r in errors:
            log.warning(f"    ORDER ERROR {r.symbol}: {r.error}")
    elif plan["trade"] and not market_open:
        deferred = True
        log.info(f"    Rebalance needed (drift {plan['drift']:.1%}) but market is NOT open yet "
                 f"— orders DEFERRED. Re-run 9:30am–4:00pm ET to execute.")
    else:
        log.info(f"    drift {plan['drift']:.2%} inside the {TRADE_BAND:.0%} band — no trades")

    reason = None
    if plan.get("blocked"):
        reason = (f"CIRCUIT BREAKER — plan wanted {plan['drift']:.1%} turnover "
                  f"(limit {MAX_DAILY_TURNOVER:.0%}); no trades placed")
    elif deferred:
        reason = f"rebalance deferred — market not open (drift {plan['drift']:.1%})"
    elif not plan["trade"]:
        reason = f"drift {plan['drift']:.2%} inside no-trade band"

    run = {"mode_label": mode_label, "targets": targets, "account": broker.account(),
           "plan": plan, "records": records, "positions_after": broker.positions(),
           "no_trade_reason": reason}
    log.info(f"4/4 Report -> {write_report(run)}")


if __name__ == "__main__":
    try:
        main(dry_run="--dry-run" in sys.argv)
    except Exception:
        log.error("RUN FAILED:\n" + traceback.format_exc())
        sys.exit(1)
