"""
Data Ingestion Layer
Pulls from FRED API, Yahoo Finance, and Shiller CAPE (Yale).
Caches locally to avoid redundant API calls.
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config.settings import (
    ALL_TICKERS,
    DATA_DIR,
    FRED_API_KEY,
    FRED_SERIES,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  FRED API
# ═══════════════════════════════════════════════════════════════════════

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_fred_series(
    series_id: str,
    start: str = "1920-01-01",
    end: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.Series:
    """Fetch a single FRED series and return as a pandas Series."""
    key = api_key or FRED_API_KEY
    if key == "YOUR_FRED_KEY_HERE":
        raise ValueError("Set FRED_API_KEY in your .env file")

    end = end or datetime.now().strftime("%Y-%m-%d")
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
        "sort_order": "asc",
    }
    resp = requests.get(FRED_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])

    dates, values = [], []
    for o in obs:
        if o["value"] != ".":
            dates.append(pd.Timestamp(o["date"]))
            values.append(float(o["value"]))

    s = pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id)
    s = s[~s.index.duplicated(keep="last")]
    return s


def fetch_all_fred(start: str = "1920-01-01") -> pd.DataFrame:
    """Fetch all configured FRED series into a single DataFrame (monthly freq)."""
    cache_path = DATA_DIR / "fred_raw.pkl"
    if cache_path.exists():
        age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if age < timedelta(hours=20):
            logger.info("Loading FRED data from cache")
            return pd.read_pickle(cache_path)

    frames = {}
    for name, sid in FRED_SERIES.items():
        try:
            logger.info(f"Fetching FRED: {name} ({sid})")
            frames[name] = fetch_fred_series(sid, start=start)
        except Exception as e:
            logger.warning(f"Failed to fetch {name}: {e}")

    df = pd.DataFrame(frames)
    # Resample everything to monthly (end of month) to align frequencies
    df.index = pd.DatetimeIndex(df.index)
    df = df.resample("ME").last()
    df = df.ffill(limit=3)

    df.to_pickle(cache_path)
    logger.info(f"FRED data cached: {df.shape}")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  Shiller CAPE Data (from Yale)
# ═══════════════════════════════════════════════════════════════════════



def fetch_shiller_cape() -> pd.DataFrame:
    """Read Shiller's monthly CAPE data (1871-present) from local file in data/ folder."""
    cache_path = DATA_DIR / "shiller_cape.pkl"
    if cache_path.exists():
        age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if age < timedelta(days=7):
            return pd.read_pickle(cache_path)

    # Look for local file in the data/ folder
    local_path = Path(__file__).parent / "ie_data.xls"
    if not local_path.exists():
        local_path = Path(__file__).parent / "ie_data.xlsx"
    if not local_path.exists():
        local_path = Path(__file__).parent / "ie_data.csv"

    if not local_path.exists():
        raise FileNotFoundError(
            "Shiller ie_data.xls not found in the data/ folder. "
            "Download it from http://www.econ.yale.edu/~shiller/data.htm "
            "and place it in the data/ folder as ie_data.xls"
        )

    # Check file age and warn if older than current month
    file_mtime = datetime.fromtimestamp(local_path.stat().st_mtime)
    now = datetime.now()
    if (now.year, now.month) > (file_mtime.year, file_mtime.month):
        months_old = (now.year - file_mtime.year) * 12 + (now.month - file_mtime.month)
        logger.warning(
            "=" * 70 + "\n"
            f"  WARNING: Shiller ie_data file is {months_old} month(s) old!\n"
            f"  File date: {file_mtime.strftime('%Y-%m-%d')}  |  Current date: {now.strftime('%Y-%m-%d')}\n"
            f"  Location: {local_path}\n"
            f"  Please download the latest version and replace the file in data/\n"
            + "=" * 70
        )

    logger.info(f"Reading Shiller CAPE data from: {local_path}")

    # Read based on file extension
    if local_path.suffix == ".csv":
        xls = pd.read_csv(local_path, header=None, skiprows=8)
    else:
        xls = pd.read_excel(local_path, sheet_name="Data", header=None, skiprows=8)

    df = xls.iloc[:, :13].copy()
    df.columns = [
        "date_frac", "sp_price", "dividend", "earnings", "cpi", "date_frac2",
        "long_rate", "real_price", "real_dividend", "real_tr_price",
        "real_earnings", "real_tr_earnings", "CAPE",
    ]
    df = df.dropna(subset=["date_frac"])
    df = df[df["date_frac"].apply(lambda x: isinstance(x, (int, float)))]
    years = df["date_frac"].astype(float).values
    dates = []
    for y in years:
        yr = int(y)
        mo = round((y - yr) * 100)
        mo = max(1, min(12, mo))
        dates.append(pd.Timestamp(year=yr, month=mo, day=1))
    df.index = pd.DatetimeIndex(dates)
    df = df[~df.index.duplicated(keep="last")]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_pickle(cache_path)
    logger.info(f"Shiller data cached: {df.shape}, range {df.index[0]} to {df.index[-1]}")
    return df
# ═══════════════════════════════════════════════════════════════════════
#  Yahoo Finance — Asset Prices
# ═══════════════════════════════════════════════════════════════════════


def fetch_yahoo_prices(
    tickers: Optional[dict] = None,
    start: str = "1990-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch adjusted close prices for all tickers via Yahoo Finance."""
    tickers = tickers or ALL_TICKERS
    cache_path = DATA_DIR / "yahoo_prices.pkl"
    if cache_path.exists():
        age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if age < timedelta(hours=20):
            return pd.read_pickle(cache_path)

    end = end or datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Fetching Yahoo Finance prices for {len(tickers)} tickers")

    ticker_list = list(tickers.keys())
    data = yf.download(ticker_list, start=start, end=end, auto_adjust=True, threads=True)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"] if "Close" in data.columns.get_level_values(0) else data.iloc[:, :len(ticker_list)]
    else:
        prices = data

    prices = prices.resample("ME").last()
    prices.to_pickle(cache_path)
    logger.info(f"Yahoo prices cached: {prices.shape}")
    return prices


# ═══════════════════════════════════════════════════════════════════════
#  Feature Engineering for Regime Model
# ═══════════════════════════════════════════════════════════════════════


def build_feature_matrix(fred_df: pd.DataFrame, shiller_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix for regime classification.
    All features are transformed to be stationary and comparable.
    """
    feat = pd.DataFrame(index=fred_df.index)

    # ── Yield curve spreads ──
    if "GS10" in fred_df and "GS2" in fred_df:
        feat["spread_10y2y"] = fred_df["GS10"] - fred_df["GS2"]
    if "GS10" in fred_df and "TB3MS" in fred_df:
        feat["spread_10y3m"] = fred_df["GS10"] - fred_df["TB3MS"]

    # ── Credit spread ──
    if "BAA10Y" in fred_df:
        feat["credit_spread"] = fred_df["BAA10Y"]
        feat["credit_spread_chg_3m"] = fred_df["BAA10Y"].diff(3)

    # ── Unemployment rate & change ──
    if "UNRATE" in fred_df:
        feat["unrate"] = fred_df["UNRATE"]
        feat["unrate_chg_3m"] = fred_df["UNRATE"].diff(3)
        feat["unrate_chg_12m"] = fred_df["UNRATE"].diff(12)

    # ── Industrial production growth ──
    if "INDPRO" in fred_df:
        feat["indpro_yoy"] = fred_df["INDPRO"].pct_change(12) * 100
        feat["indpro_mom"] = fred_df["INDPRO"].pct_change(1) * 100

    # ── CPI inflation ──
    if "CPIAUCSL" in fred_df:
        feat["cpi_yoy"] = fred_df["CPIAUCSL"].pct_change(12) * 100

    # ── Housing starts momentum ──
    if "HOUST" in fred_df:
        feat["houst_yoy"] = fred_df["HOUST"].pct_change(12) * 100

    # ── Payrolls growth ──
    if "PAYEMS" in fred_df:
        feat["payems_yoy"] = fred_df["PAYEMS"].pct_change(12) * 100
        feat["payems_mom"] = fred_df["PAYEMS"].pct_change(1) * 100

    # ── Consumer sentiment ──
    if "UMCSENT" in fred_df:
        feat["umcsent"] = fred_df["UMCSENT"]
        feat["umcsent_chg"] = fred_df["UMCSENT"].diff(3)

    # ── Initial claims (inverse — higher is worse) ──
    if "ICSA" in fred_df:
        feat["icsa_4wk_chg"] = fred_df["ICSA"].rolling(4).mean().pct_change(12) * 100

    # ── Fed Funds ──
    if "FEDFUNDS" in fred_df:
        feat["fedfunds"] = fred_df["FEDFUNDS"]
        feat["fedfunds_chg_12m"] = fred_df["FEDFUNDS"].diff(12)

    # ── M2 growth ──
    if "M2SL" in fred_df:
        feat["m2_yoy"] = fred_df["M2SL"].pct_change(12) * 100

    # ── S&P 500 momentum ──
    if "SP500" in fred_df:
        feat["sp500_yoy"] = fred_df["SP500"].pct_change(12) * 100
        feat["sp500_mom_3m"] = fred_df["SP500"].pct_change(3) * 100

    # ── VIX ──
    if "VIXCLS" in fred_df:
        feat["vix"] = fred_df["VIXCLS"]

    # ── Shiller CAPE ──
    if shiller_df is not None and "CAPE" in shiller_df:
        cape = shiller_df["CAPE"].resample("ME").last()
        cape = cape.reindex(feat.index, method="ffill")
        feat["cape"] = cape
        feat["cape_zscore"] = (cape - cape.rolling(120).mean()) / cape.rolling(120).std()

    # ── NBER recession label (target) ──
    if "USREC" in fred_df:
        feat["recession"] = fred_df["USREC"]

    # Drop rows with insufficient data
    feat = feat.dropna(thresh=int(len(feat.columns) * 0.5))

    return feat


def clear_cache():
    """Remove all cached data files to force a fresh pull."""
    for f in DATA_DIR.glob("*.pkl"):
        f.unlink()
        logger.info(f"Removed cache: {f}")
