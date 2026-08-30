"""
Live-trading configuration. All secrets come from a .env file in the repo
root (see .env.example) — nothing sensitive is ever committed.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
# Paper trading by default. Set ALPACA_PAPER=false in .env ONLY when the
# strategy has proven itself on paper and live trading is intended.
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").strip().lower() != "false"

# Portfolio: the 3-desk firm, equal-weight across desks (the boss-report
# configuration). Long/flat, unleveraged — mandate switches stay off live.
INCLUDE_DESKS = ["commodities", "equities", "rates"]
FIRM_VARIANT = "equal_weight"

# Rebalance only when actual positions have drifted from target by more than
# this L1 distance (fraction of equity) — mirrors the backtest's no-trade band.
TRADE_BAND = 0.05

# Circuit breaker on RESHUFFLE turnover — trading between assets while total
# invested exposure stays put. Observed legitimate rebalances run 5-11% of the
# book; the 2026-08-20 data glitch tried 80%. Changes in exposure itself
# (crisis liquidation to cash, initial deployment) are excluded from this
# measure, so genuine de-risking is never blocked.
MAX_DAILY_TURNOVER = 0.25

MIN_ORDER_NOTIONAL = 5.0      # skip dust orders below this many dollars
ORDER_FILL_TIMEOUT = 180      # seconds to wait for a market order to fill
                              # (raised from 120 after EMB took 116s at the
                              # volatile open on the first live rebalance)
REPORTS_DIR = ROOT / "live" / "reports"


def require_keys():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise SystemExit(
            "Missing Alpaca credentials. Copy .env.example to .env in the repo "
            "root and fill in ALPACA_API_KEY and ALPACA_SECRET_KEY."
        )
