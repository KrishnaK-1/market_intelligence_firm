# Live Trading via Alpaca

Runs the 3-desk firm (commodities + equities + rates, long/flat, unleveraged
— the boss-report configuration) against an Alpaca account. One command per
trading day; everything else is automatic.

## Setup (one time, ~5 minutes)

1. Clone/pull this repo, branch `alpaca-live`.
2. `pip install -r requirements.txt`
3. Create an Alpaca account (free) and generate **paper trading** API keys.
4. Copy `.env.example` to `.env` in the repo root; paste in the two keys.
5. Sanity check without touching Alpaca:
   `python live/run_daily.py --dry-run`
   (downloads data, computes targets, writes a report — places no orders)

## Daily run

    python live/run_daily.py

What it does, in order:
1. Re-downloads all price history through **yesterday's close** (a partial
   bar for today can never leak into signals, whatever time you run it).
2. Runs the exact same engine as the backtest and takes today's target
   weights from its final row.
3. Reads the account's actual equity and positions from Alpaca; if total
   drift vs target is inside the 5% no-trade band, does nothing (most days).
4. Otherwise trades the differences — sells first, then buys, as market
   orders (fractional/notional where supported).
5. Writes `live/reports/trading_report_<date>.xlsx` (trades with bid/ask and
   submit-to-fill latency, positions vs targets, account state) and appends
   a summary line to `live/reports/master_log.csv`.

Schedule it once per trading day while the market is open — e.g. 9:35 am ET
via Windows Task Scheduler ("Start a program": `python`, arguments:
`live/run_daily.py`, start in: repo folder) or cron on Mac/Linux:
`35 9 * * 1-5 cd /path/to/repo && python live/run_daily.py >> live/reports/cron.log 2>&1`

## Safety properties

- **Paper by default.** Live requires deliberately setting `ALPACA_PAPER=false`.
- **Never shorts, never levers**: aborts if any target weight is negative or
  gross exposure exceeds 100%; sells are capped at the existing position.
- **Refuses to trade on bad data**: aborts if any price download comes back
  empty; skips holidays/weekends automatically.
- **No state to corrupt**: targets are recomputed from scratch daily and
  reconciled against the broker's actual positions, so a missed day
  self-heals on the next run.

## Honest caveats

- Backtest fidelity is close but not exact: the backtest trades at the
  close, this trades near the next open (slippage from that gap is small for
  50–200-day signals, but it is not zero, and live results will drift from
  backtest over time — that comparison is exactly what paper trading is for).
- Free-plan quotes are IEX, not consolidated NBBO — treat logged spreads as
  indicative.
- Several commodity ETFs here (USO, USL, UNG, UGA, DBA, CORN, WEAT, SOYB,
  CANE) are limited partnerships that issue K-1 tax forms and can generate
  UBTI in IRAs — flag to the tax side of the product before using those
  account types.
