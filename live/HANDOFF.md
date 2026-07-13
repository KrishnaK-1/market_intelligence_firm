# Alpaca Live Trading — Setup & Operations Guide

This guide takes you from a fresh computer to a running (paper-money) trading
program in about 15 minutes, and explains how to operate and maintain it.

**What this is:** a systematic trading program that manages a portfolio of 34
US-listed ETFs across three "desks" — commodities, equities, and rates/credit.
Once per trading day it recomputes its ideal portfolio from fresh market data
and places orders through your Alpaca brokerage account to close the gap. It
is **long-only, never uses leverage, and starts in paper-trading mode** (fake
money, real prices) — it cannot touch real dollars unless you deliberately
flip one setting.

---

## 1. One-time setup

**Prerequisites:** [Python 3.11+](https://www.python.org/downloads/) and
[Git](https://git-scm.com/downloads) installed. On the Python installer,
check "Add Python to PATH".

1. **Get the code** (in a terminal / PowerShell):

   ```
   git clone <repo-url>
   cd market_intelligence_firm
   git checkout alpaca-live
   ```

   (If you already have the repo: `git pull`, then `git checkout alpaca-live`.)

2. **Install the dependencies:**

   ```
   pip install -r requirements.txt
   ```

3. **Create a free Alpaca account** at [alpaca.markets](https://alpaca.markets).
   In the dashboard, switch to **Paper Trading** (toggle in the top-left) and
   generate an **API Key** and **Secret Key**.

4. **Give the program your keys.** In the repo folder, copy the file
   `.env.example` to a new file named exactly `.env`, open it in any text
   editor, and paste your two keys:

   ```
   ALPACA_API_KEY=PK...your key...
   ALPACA_SECRET_KEY=...your secret...
   ALPACA_PAPER=true
   ```

   Leave `ALPACA_PAPER=true`. The `.env` file stays on your machine — it is
   never uploaded or committed.

5. **Test without trading** (places zero orders, needs no keys):

   ```
   python live/run_daily.py --dry-run
   ```

   Takes ~3–5 minutes (it downloads 20 years of price history and runs the
   full model). Success looks like:

   ```
   1/4 Refreshing price data...
   2/4 Computing target weights (full engine run, ~2-4 min)...
       signals as of close 2026-07-10 | gross exposure 85.0% | cash 15.0% | 34 tickers
   4/4 DRY RUN report -> live/reports/trading_report_2026-07-13.xlsx
   ```

---

## 2. Running it for real (paper money)

One command, once per trading day, while the market is open:

```
python live/run_daily.py
```

What it does each run, in order:

1. **Refreshes data** — re-downloads price history through *yesterday's
   close* (it is built so a half-finished bar from today can never
   contaminate the signals, whatever time you run it).
2. **Computes targets** — runs the exact same engine as the research
   backtest and takes today's ideal portfolio weights.
3. **Checks your account** — reads real equity and positions from Alpaca.
   If the portfolio is within 5% of target (most days), it does nothing.
4. **Trades the difference** — sells first, then buys, as market orders.
5. **Writes the daily report** (see §3).

It skips weekends and market holidays by itself, and it never needs
yesterday's run to have happened — every day it re-derives everything from
scratch and reconciles against what the account actually holds, so missed
days heal themselves.

### Scheduling it (recommended)

Windows Task Scheduler → Create Basic Task:

- **Trigger:** Daily, 9:35 AM (ET — adjust if your machine is in another timezone)
- **Action:** Start a program
  - Program: `python`
  - Arguments: `live\run_daily.py`
  - Start in: the full path to the repo folder
- On the task's settings, enable **"Wake the computer to run this task."**

Mac/Linux cron equivalent:

```
35 9 * * 1-5 cd /path/to/market_intelligence_firm && python live/run_daily.py >> live/reports/cron.log 2>&1
```

---

## 3. The daily report

Every run writes an Excel file to the **`live/reports/`** folder inside the
repo — no downloading needed; just open it:

```
live/reports/trading_report_2026-07-13.xlsx
```

| Sheet | What's in it |
|---|---|
| **Trades** | every order placed today: buy/sell, dollar amount requested, shares and price filled, **bid/ask at the moment of submission**, spread, and **submit→fill latency in seconds** |
| **Positions** | what the account holds after trading, each holding's weight vs its target weight |
| **Targets** | today's full ideal portfolio, plus the cash remainder |
| **Account** | equity, cash, paper/live mode, whether it rebalanced today, per-desk exposure |

There is also **`live/reports/master_log.csv`** — one summary line per day
(equity, drift, orders placed/filled/errors) so you can see the whole history
at a glance or chart it in Excel.

On days inside the no-trade band, the report still gets written — the Trades
sheet just notes that no orders were needed. That's normal and by design;
this strategy trades meaningfully only a few times a month.

---

## 4. Care & maintenance — read this part

- **The computer must be on (and awake) at the scheduled time.** This
  program runs on your machine, not in the cloud. A closed laptop lid =
  sleep = missed run. Missed runs are harmless (it self-heals next day),
  but the strategy assumes daily checks — don't let it miss weeks.
- **VSCode does NOT need to be open.** Nothing needs to be open. The
  scheduled task runs Python invisibly in the background. If you run it
  manually in a terminal instead, keep that terminal open until it prints
  the report path (~5 minutes), then you may close it.
- **Leave it in paper mode for at least 3–6 months.** The point of paper
  trading is to compare live behavior against the research backtest before
  any real dollar moves. Only after that comparison looks right should
  `ALPACA_PAPER=false` even be discussed.
- **Don't edit code on this branch** while it's running day to day. Config
  knobs live in `live/config.py`; secrets in `.env`; everything else should
  be treated as sealed.
- **Occasionally `git pull`** to pick up fixes (your engineer will tell you
  when).
- **Internet is required** at run time (price data + Alpaca). If a run fails
  mid-way it places at most the orders it already reported — run it again;
  reconciliation makes a second run safe.
- Disk usage is trivial (one small Excel file per day).

---

## 5. Safety properties (what it will never do)

- Never trades real money unless `ALPACA_PAPER` is deliberately set to `false`.
- Never shorts and never uses leverage — it aborts if the model ever asks.
- Never trades on missing/empty data — it aborts instead.
- Sells are capped at what the account actually holds.
- Every run is stateless: it can be re-run safely at any time.

## 6. Troubleshooting

| Symptom | Meaning / fix |
|---|---|
| `Missing Alpaca credentials` | The `.env` file is missing or the key names are misspelled. |
| `Download returned no data` | yfinance/internet hiccup — just run it again. |
| `drift ... inside the no-trade band — no trades` | Normal. The portfolio is already close enough to target. |
| `Market is closed today` | Weekend or holiday; nothing to do. |
| Order shows `skipped: notional below one whole share` | That ETF doesn't support fractional shares and the trade was smaller than one share — harmless dust, it'll catch up on a bigger rebalance. |

Questions → the engineer who set this up (Krish).
