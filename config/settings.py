"""
Market Intelligence Firm — Global Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "cache"
MODEL_DIR = BASE_DIR / "models" / "saved"
LOG_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "YOUR_FRED_KEY_HERE")
BLS_API_KEY = os.getenv("BLS_API_KEY", "YOUR_BLS_KEY_HERE")

# ── Regime Model ───────────────────────────────────────────────────────
N_REGIMES = 5  # expansion, slowdown, contraction, recovery, crisis
REGIME_LABELS = {
    0: "Expansion",
    1: "Slowdown",
    2: "Contraction",
    3: "Recovery",
    4: "Crisis",
}
REGIME_COLORS = {
    0: "#22c55e",  # green
    1: "#f59e0b",  # amber
    2: "#ef4444",  # red
    3: "#3b82f6",  # blue
    4: "#7c3aed",  # purple
}

FORECAST_HORIZONS = [1, 3, 6]  # months

# ── FRED Series ────────────────────────────────────────────────────────
FRED_SERIES = {
    # Real activity
    "GDP":        "GDPC1",         # Real GDP (quarterly)
    "INDPRO":     "INDPRO",        # Industrial Production
    "PAYEMS":     "PAYEMS",        # Nonfarm Payrolls
    "UNRATE":     "UNRATE",        # Unemployment Rate
    "ICSA":       "ICSA",          # Initial Claims (weekly)
    "HOUST":      "HOUST",         # Housing Starts
    # Prices
    "CPIAUCSL":   "CPIAUCSL",      # CPI
    "PCEPI":      "PCEPI",         # PCE Price Index
    # Rates & Spreads
    "GS10":       "GS10",          # 10Y Treasury
    "GS2":        "GS2",           # 2Y Treasury
    "TB3MS":      "TB3MS",         # 3M T-Bill
    "FEDFUNDS":   "FEDFUNDS",      # Fed Funds
    "BAA10Y":     "BAA10Y",        # Baa-10Y spread (credit)
    # Surveys & Conditions
    "UMCSENT":    "UMCSENT",       # Michigan Consumer Sentiment

   # "NAPM":       "NAPM",          # ISM PMI (discontinued, use MANEMP proxy)
    "ISM_PMI":    "MANEMP",         # ISM Manufacturing Employment (proxy for PMI)
    # Financial
    "SP500":      "SP500",         # S&P 500
    "VIXCLS":     "VIXCLS",        # VIX
    # Money
    "M2SL":       "M2SL",          # M2 Money Supply
    # NBER recession indicator (target)
    "USREC":      "USREC",         # NBER Recession Indicator
}

# ── Investment Universe (ETF tickers) ─────────────────────────────────
EQUITY_SECTORS = {
    "SPY": "S&P 500",
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
}

FIXED_INCOME = {
    "TLT": "20+ Year Treasury",
    "IEF": "7-10 Year Treasury",
    "SHY": "1-3 Year Treasury",
    "LQD": "Investment Grade Corporate",
    "HYG": "High Yield Corporate",
    "TIP": "TIPS",
}

COMMODITIES = {
    "GLD": "Gold",
    "USO": "Crude Oil",
    "DBC": "Broad Commodities",
}

INTERNATIONAL = {
    "EFA": "Developed Markets ex-US",
    "EEM": "Emerging Markets",
    "VGK": "Europe",
    "EWJ": "Japan",
    "FXI": "China",
}

ALL_TICKERS = {**EQUITY_SECTORS, **FIXED_INCOME, **COMMODITIES, **INTERNATIONAL}

# ── Allocation Templates (probability-weighted by regime) ──────────────
# Each regime maps to a target allocation across asset classes (sums to 1.0)
REGIME_ALLOCATIONS = {
    0: {  # Expansion
        "US Equities": 0.45,
        "Fixed Income": 0.15,
        "Commodities": 0.15,
        "International/EM": 0.20,
        "Cash": 0.05,
    },
    1: {  # Slowdown
        "US Equities": 0.25,
        "Fixed Income": 0.30,
        "Commodities": 0.10,
        "International/EM": 0.15,
        "Cash": 0.20,
    },
    2: {  # Contraction
        "US Equities": 0.10,
        "Fixed Income": 0.40,
        "Commodities": 0.05,
        "International/EM": 0.05,
        "Cash": 0.40,
    },
    3: {  # Recovery
        "US Equities": 0.40,
        "Fixed Income": 0.20,
        "Commodities": 0.15,
        "International/EM": 0.20,
        "Cash": 0.05,
    },
    4: {  # Crisis
        "US Equities": 0.05,
        "Fixed Income": 0.30,
        "Commodities": 0.15,  # gold hedge
        "International/EM": 0.00,
        "Cash": 0.50,
    },
}

# Sub-allocations within each asset class per regime
REGIME_SUB_ALLOCATIONS = {
    0: {  # Expansion — pro-growth, cyclical tilt
        "SPY": 0.20, "XLK": 0.08, "XLF": 0.05, "XLI": 0.05, "XLY": 0.04, "XLE": 0.03,
        "IEF": 0.05, "LQD": 0.05, "HYG": 0.05,
        "GLD": 0.05, "USO": 0.05, "DBC": 0.05,
        "EFA": 0.08, "EEM": 0.08, "FXI": 0.04,
    },
    1: {  # Slowdown — defensive rotation
        "SPY": 0.10, "XLV": 0.05, "XLP": 0.05, "XLU": 0.05,
        "TLT": 0.10, "IEF": 0.10, "TIP": 0.05, "LQD": 0.05,
        "GLD": 0.07, "DBC": 0.03,
        "EFA": 0.10, "EEM": 0.05,
    },
    2: {  # Contraction — max defensive
        "SPY": 0.05, "XLP": 0.03, "XLV": 0.02,
        "TLT": 0.15, "IEF": 0.10, "SHY": 0.10, "TIP": 0.05,
        "GLD": 0.05,
        "EFA": 0.05,
    },
    3: {  # Recovery — early cycle, value + EM
        "SPY": 0.15, "XLF": 0.07, "XLI": 0.07, "XLE": 0.05, "XLB": 0.03, "XLY": 0.03,
        "IEF": 0.08, "LQD": 0.07, "HYG": 0.05,
        "GLD": 0.05, "USO": 0.05, "DBC": 0.05,
        "EFA": 0.07, "EEM": 0.08, "FXI": 0.05,
    },
    4: {  # Crisis — capital preservation, gold hedge
        "SPY": 0.03, "XLP": 0.02,
        "TLT": 0.10, "SHY": 0.15, "TIP": 0.05,
        "GLD": 0.15,
    },
}

# ── Dashboard / Server ─────────────────────────────────────────────────
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
REFRESH_HOUR = 7  # 7 AM ET daily auto-refresh
TIMEZONE = "US/Eastern"
