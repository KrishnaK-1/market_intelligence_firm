"""
Commodity price data fetcher using yfinance.
All tickers are ETFs/futures-based instruments tradeable via standard brokers.
"""
import pandas as pd
import yfinance as yf
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"

COMMODITY_GROUPS = {
    "energy": {
        "WTI":  "USO",   # WTI Crude Oil ETF
        "BRENT": "BNO",  # Brent Crude ETF
        "NG":   "UNG",   # Natural Gas ETF
        # Heating oil (formerly UHN) has no currently-tradeable ETF — UHN was delisted Sep 2018.
    },
    "agriculture": {
        "CORN":    "CORN",  # Teucrium Corn ETF
        "WHEAT":   "WEAT",  # Teucrium Wheat ETF
        "SOY":     "SOYB",  # Teucrium Soybean ETF
        # Live cattle (formerly COW) has no currently-tradeable ETN — COW was delisted Jul 2023.
    },
    "metals": {
        "GOLD":     "GLD",   # SPDR Gold Shares
        "SILVER":   "SLV",   # iShares Silver Trust
        "COPPER":   "CPER",  # United States Copper Index Fund
        "PLATINUM": "PPLT",  # Aberdeen Physical Platinum Shares ETF
    },
}

# Futures-curve proxy pairs for the rules-based desk. The trailing return
# spread between a front-month ETF and its 12-month sibling is an observable
# proxy for curve slope: front outperforming = backwardation (positive roll
# yield), front underperforming = contango (roll drag). Signal-only tickers —
# except USL, which the rules desk trades in place of USO (USO's April 2020
# forced restructurings make it a different instrument pre/post crisis).
CURVE_TICKERS = {
    "WTI_FRONT": "USO",   # front-month WTI (signal only for rules desk)
    "WTI_12M":   "USL",   # 12-month WTI ladder — cleaner roll, tradeable
    "NG_12M":    "UNL",   # 12-month natural gas ladder (curve signal for UNG)
    "WTI_OPT":   "DBO",   # optimized-roll WTI (alternative curve reference)
}

# Geopolitical / macro state variables fed to the meta-agent
MACRO_TICKERS = {
    "DXY":    "UUP",    # US Dollar Index ETF
    "OVX":    "^OVX",   # Crude Oil Volatility Index
    "GVZ":    "^GVZ",   # Gold Volatility Index
    "TIPS":   "TIP",    # Inflation expectations proxy
    "VIX":    "^VIX",   # Broad market fear — primary crisis trigger signal
    "10Y":    "^TNX",   # 10-Year Treasury yield
    "CREDIT": "HYG",    # High-yield corporate bonds — credit stress proxy (cracks before equities in many crises)
    "BDRY":   "BDRY",   # Baltic Dry shipping ETF — global freight/export disruption proxy (agriculture)
}


def fetch_group(group: str, start: str = "2010-01-01", end: str = None) -> pd.DataFrame:
    """
    Download OHLCV for one commodity group. Returns MultiIndex df sliced to [start:end].
    Cache always stores the full available history; date filtering happens on read
    so disjoint train/test windows don't require separate downloads or cache files.
    """
    tickers = list(COMMODITY_GROUPS[group].values())
    cache_path = CACHE_DIR / f"commodities_{group}.pkl"

    if cache_path.exists():
        df = pd.read_pickle(cache_path)
    else:
        df = yf.download(tickers, start="2010-01-01", end=None, auto_adjust=True, progress=False)
        df.to_pickle(cache_path)

    return df.loc[start:end]


def fetch_macro(start: str = "2010-01-01", end: str = None) -> pd.DataFrame:
    """Download macro state variables for the meta-agent, sliced to [start:end]."""
    tickers = list(MACRO_TICKERS.values())
    cache_path = CACHE_DIR / "commodities_macro.pkl"

    if cache_path.exists():
        df = pd.read_pickle(cache_path)
    else:
        df = yf.download(tickers, start="2010-01-01", end=None, auto_adjust=True, progress=False)
        df.to_pickle(cache_path)

    return df.loc[start:end]


def fetch_tickers(tickers: list, cache_name: str, start: str = "2010-01-01", end: str = None) -> pd.DataFrame:
    """Generic cached download of adjusted closes for an arbitrary ticker list.
    Returns close prices only, columns = tickers."""
    cache_path = CACHE_DIR / f"{cache_name}.pkl"

    if cache_path.exists():
        df = pd.read_pickle(cache_path)
    else:
        df = yf.download(tickers, start="2010-01-01", end=None, auto_adjust=True, progress=False)
        df.to_pickle(cache_path)

    closes = df["Close"] if "Close" in df else df.xs("Close", axis=1, level=0)
    return closes.loc[start:end]


def fetch_curve_pairs(start: str = "2010-01-01", end: str = None) -> pd.DataFrame:
    """Close prices for the futures-curve proxy ETFs, columns = logical names."""
    closes = fetch_tickers(list(CURVE_TICKERS.values()), "commodities_curve", start, end)
    name_map = {v: k for k, v in CURVE_TICKERS.items()}
    return closes.rename(columns=name_map).dropna(how="all")


def get_close_prices(group: str, start: str = "2010-01-01", end: str = None) -> pd.DataFrame:
    """Return only adjusted close prices for a group, columns = ticker names."""
    raw = fetch_group(group, start, end)
    closes = raw["Close"] if "Close" in raw else raw.xs("Close", axis=1, level=0)
    name_map = {v: k for k, v in COMMODITY_GROUPS[group].items()}
    return closes.rename(columns=name_map).dropna(how="all")
