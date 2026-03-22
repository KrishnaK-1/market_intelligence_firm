"""
Nowcast Module - Real-Time Regime Classification
=================================================
Fetches the latest available data from BLS API, Yahoo Finance, and FRED
daily series, constructs a synthetic current-month feature row matching
the training schema, and runs XGBoost-only prediction for immediate
regime classification.

Data sources (checked in priority order):
  BLS API v2:    Payrolls, unemployment, CPI (fastest for employment data)
  Yahoo Finance: S&P 500, VIX (real-time market data)
  FRED daily:    Treasury yields, credit spreads, fed funds rate
  FRED monthly:  Fallback for all series if BLS/Yahoo unavailable

Data tiers:
  Tier 1 (daily):   S&P 500, VIX, Treasury yields, credit spreads, fed funds
  Tier 1.5 (BLS):   Payrolls, unemployment, CPI (updated day after BLS release)
  Tier 2 (weekly):  Initial claims, consumer sentiment
  Tier 3 (monthly): Industrial production, housing starts, M2

For unavailable features, carries forward the last official FRED monthly value.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config.settings import DATA_DIR, FRED_API_KEY, BLS_API_KEY, MODEL_DIR

logger = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Feature columns the model expects (24 features, no 'recession')
FEATURE_COLS = [
    "spread_10y2y", "spread_10y3m", "credit_spread", "credit_spread_chg_3m",
    "unrate", "unrate_chg_3m", "unrate_chg_12m",
    "indpro_yoy", "indpro_mom", "cpi_yoy", "houst_yoy",
    "payems_yoy", "payems_mom", "umcsent", "umcsent_chg",
    "icsa_4wk_chg", "fedfunds", "fedfunds_chg_12m",
    "m2_yoy", "sp500_yoy", "sp500_mom_3m", "vix",
    "cape", "cape_zscore",
]

# FRED series that have daily observations (Tier 1)
FRED_DAILY_SERIES = {
    "GS10": "GS10",        # 10Y Treasury yield
    "GS2": "DGS2",         # 2Y Treasury yield (daily version)
    "TB3MS": "DTB3",        # 3M T-Bill (daily version)
    "BAA10Y": "BAA10Y",    # Baa - 10Y spread
    "FEDFUNDS": "DFF",      # Daily effective fed funds rate
}

# BLS API series (faster than FRED for employment/prices data)
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
BLS_SERIES = {
    "PAYEMS": "CES0000000001",   # Total nonfarm payrolls (thousands)
    "UNRATE": "LNS14000000",     # Unemployment rate
    "CPI": "CUSR0000SA0",        # CPI All Items (index level)
}

# Treasury Fiscal Data API (tax withholding — display signal only, no key required)
TREASURY_API_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
TREASURY_DTS_ENDPOINT = "/v1/accounting/dts/deposits_withdrawals_operating_cash"


def _fetch_bls_latest(series_id: str, years_back: int = 2) -> Optional[Tuple[float, str]]:
    """
    Fetch the most recent observation from BLS API v2.
    Returns (value, date_string) or None on failure.
    BLS updates faster than FRED for employment and price data.
    """
    if not BLS_API_KEY:
        logger.debug("No BLS_API_KEY configured, skipping BLS fetch")
        return None

    try:
        end_year = datetime.now().year
        start_year = end_year - years_back
        payload = {
            "seriesid": [series_id],
            "startyear": str(start_year),
            "endyear": str(end_year),
            "registrationkey": BLS_API_KEY,
        }
        resp = requests.post(BLS_API_URL, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "REQUEST_SUCCEEDED":
            logger.warning(f"BLS API error: {data.get('message', 'unknown')}")
            return None

        series_data = data.get("Results", {}).get("series", [])
        if not series_data:
            return None

        observations = series_data[0].get("data", [])
        if not observations:
            return None

        # BLS returns most recent first
        for obs in observations:
            val_str = obs.get("value", "")
            if val_str and val_str != "-":
                year = obs["year"]
                period = obs["period"]  # e.g. "M02" for February
                if period.startswith("M"):
                    month = int(period[1:])
                    date_str = f"{year}-{month:02d}-01"
                    return float(val_str), date_str

        return None
    except Exception as e:
        logger.warning(f"BLS fetch failed for {series_id}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Real-Time Data Fetchers
# ═══════════════════════════════════════════════════════════════════════


def _fetch_fred_latest(series_id: str, lookback_days: int = 90) -> Optional[Tuple[float, str]]:
    """
    Fetch the most recent observation from a FRED series.
    Returns (value, date_string) or None on failure.
    """
    try:
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": start,
            "sort_order": "desc",
            "limit": 5,
        }
        resp = requests.get(FRED_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        for o in obs:
            if o["value"] != ".":
                return float(o["value"]), o["date"]
        return None
    except Exception as e:
        logger.warning(f"FRED fetch failed for {series_id}: {e}")
        return None


def _fetch_yahoo_latest(ticker: str, period: str = "5d") -> Optional[Tuple[float, str]]:
    """
    Fetch the most recent closing price from Yahoo Finance.
    Returns (price, date_string) or None on failure.
    """
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data.empty:
            return None
        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"].iloc[:, 0]
        else:
            close = data["Close"]
        last_val = float(close.dropna().iloc[-1])
        last_date = close.dropna().index[-1].strftime("%Y-%m-%d")
        return last_val, last_date
    except Exception as e:
        logger.warning(f"Yahoo fetch failed for {ticker}: {e}")
        return None


def _fetch_yahoo_history(ticker: str, period: str = "15mo") -> Optional[pd.Series]:
    """
    Fetch historical closing prices for computing momentum/YoY changes.
    Returns a daily Series or None on failure.
    """
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            return data["Close"].iloc[:, 0].dropna()
        return data["Close"].dropna()
    except Exception as e:
        logger.warning(f"Yahoo history fetch failed for {ticker}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Feature Row Builder
# ═══════════════════════════════════════════════════════════════════════


def fetch_realtime_features(
    fred_monthly: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build a single-row DataFrame with the same 24 feature columns as the
    training matrix, using the latest available real-time data.

    Args:
        fred_monthly: The official FRED monthly DataFrame (for carry-forward
                      imputation of unavailable features). If None, attempts
                      to load from cache.

    Returns:
        (feature_row, metadata) where:
          - feature_row: DataFrame with shape (1, 24) matching model input
          - metadata: dict with freshness info per feature
    """
    now = datetime.now()
    features = {}
    metadata = {}

    def record(name: str, value: float, source: str, as_of: str, live: bool):
        features[name] = value
        metadata[name] = {"status": "live" if live else "imputed", "value": round(value, 4), "as_of": as_of, "source": source}

    # Load cached FRED monthly for fallback values
    if fred_monthly is None:
        cache_path = DATA_DIR / "fred_raw.pkl"
        if cache_path.exists():
            fred_monthly = pd.read_pickle(cache_path)
            logger.info("Nowcast: Loaded FRED monthly cache for fallback")
        else:
            logger.warning("Nowcast: No FRED cache available for fallback")

    def get_fallback(col: str, periods_back: int = 1) -> Optional[Tuple[float, str]]:
        """Get a value from the official FRED monthly data."""
        if fred_monthly is not None and col in fred_monthly.columns:
            series = fred_monthly[col].dropna()
            if len(series) >= abs(periods_back):
                val = float(series.iloc[-periods_back])
                dt = series.index[-periods_back].strftime("%Y-%m-%d")
                return val, dt
        return None

    # ─────────────────────────────────────────────────────────────────
    # TIER 1: Daily data (always fresh)
    # ─────────────────────────────────────────────────────────────────

    # Treasury yields
    gs10_result = _fetch_fred_latest("DGS10")
    gs2_result = _fetch_fred_latest("DGS2")
    tb3m_result = _fetch_fred_latest("DTB3")

    gs10_val = gs10_result[0] if gs10_result else None
    gs2_val = gs2_result[0] if gs2_result else None
    tb3m_val = tb3m_result[0] if tb3m_result else None

    # Yield curve spreads
    if gs10_val is not None and gs2_val is not None:
        record("spread_10y2y", gs10_val - gs2_val, "FRED_daily", gs10_result[1], True)
    else:
        fb = get_fallback("GS10"), get_fallback("GS2")
        if fb[0] and fb[1]:
            record("spread_10y2y", fb[0][0] - fb[1][0], "FRED_monthly", fb[0][1], False)

    if gs10_val is not None and tb3m_val is not None:
        record("spread_10y3m", gs10_val - tb3m_val, "FRED_daily", gs10_result[1], True)
    else:
        fb = get_fallback("GS10"), get_fallback("TB3MS")
        if fb[0] and fb[1]:
            record("spread_10y3m", fb[0][0] - fb[1][0], "FRED_monthly", fb[0][1], False)

    # Credit spread (BAA - 10Y)
    baa_result = _fetch_fred_latest("BAA10Y")
    if baa_result:
        record("credit_spread", baa_result[0], "FRED_daily", baa_result[1], True)
        # 3-month change: need value from ~3 months ago
        baa_3m_ago = _fetch_fred_latest("BAA10Y", lookback_days=120)
        if baa_3m_ago:
            # Get a value that's roughly 3 months old
            baa_hist = get_fallback("BAA10Y", 4)  # ~3 months back in monthly data
            if baa_hist:
                record("credit_spread_chg_3m", baa_result[0] - baa_hist[0], "FRED_mixed", baa_result[1], True)
            else:
                fb = get_fallback("BAA10Y", 1)
                fb3 = get_fallback("BAA10Y", 4)
                if fb and fb3:
                    record("credit_spread_chg_3m", fb[0] - fb3[0], "FRED_monthly", fb[1], False)
    else:
        fb = get_fallback("BAA10Y")
        if fb:
            record("credit_spread", fb[0], "FRED_monthly", fb[1], False)
        fb3 = get_fallback("BAA10Y", 4)
        if fb and fb3:
            record("credit_spread_chg_3m", fb[0] - fb3[0], "FRED_monthly", fb[1], False)

    # S&P 500 with history for YoY and 3M momentum
    sp_history = _fetch_yahoo_history("^GSPC", period="15mo")
    if sp_history is not None and len(sp_history) > 60:
        sp_now = float(sp_history.iloc[-1])
        sp_date = sp_history.index[-1].strftime("%Y-%m-%d")

        # YoY: compare to ~252 trading days ago
        sp_1y_idx = max(0, len(sp_history) - 252)
        sp_1y = float(sp_history.iloc[sp_1y_idx])
        record("sp500_yoy", ((sp_now / sp_1y) - 1) * 100, "Yahoo", sp_date, True)

        # 3M momentum: compare to ~63 trading days ago
        sp_3m_idx = max(0, len(sp_history) - 63)
        sp_3m = float(sp_history.iloc[sp_3m_idx])
        record("sp500_mom_3m", ((sp_now / sp_3m) - 1) * 100, "Yahoo", sp_date, True)
    else:
        fb = get_fallback("SP500")
        if fb:
            # Use monthly fallback for both
            fb_prev = get_fallback("SP500", 13)  # ~12 months back
            fb_3m = get_fallback("SP500", 4)  # ~3 months back
            if fb_prev:
                record("sp500_yoy", ((fb[0] / fb_prev[0]) - 1) * 100, "FRED_monthly", fb[1], False)
            if fb_3m:
                record("sp500_mom_3m", ((fb[0] / fb_3m[0]) - 1) * 100, "FRED_monthly", fb[1], False)

    # VIX
    vix_result = _fetch_yahoo_latest("^VIX")
    if vix_result:
        record("vix", vix_result[0], "Yahoo", vix_result[1], True)
    else:
        fb = get_fallback("VIXCLS")
        if fb:
            record("vix", fb[0], "FRED_monthly", fb[1], False)

    # Fed Funds Rate (changes only at FOMC meetings, so latest FRED daily is fine)
    ff_result = _fetch_fred_latest("DFF")
    if ff_result:
        record("fedfunds", ff_result[0], "FRED_daily", ff_result[1], True)
        # 12-month change
        ff_12m = get_fallback("FEDFUNDS", 13)
        if ff_12m:
            record("fedfunds_chg_12m", ff_result[0] - ff_12m[0], "FRED_mixed", ff_result[1], True)
    else:
        fb = get_fallback("FEDFUNDS")
        if fb:
            record("fedfunds", fb[0], "FRED_monthly", fb[1], False)
        fb12 = get_fallback("FEDFUNDS", 13)
        if fb and fb12:
            record("fedfunds_chg_12m", fb[0] - fb12[0], "FRED_monthly", fb[1], False)

    # ─────────────────────────────────────────────────────────────────
    # TIER 2: Weekly/monthly data (partially available)
    # BLS API is checked FIRST (faster updates than FRED)
    # ─────────────────────────────────────────────────────────────────

    # Unemployment rate — try BLS first, fall back to FRED
    unrate_result = _fetch_bls_latest(BLS_SERIES["UNRATE"])
    unrate_source = "BLS"
    if not unrate_result:
        unrate_result = _fetch_fred_latest("UNRATE")
        unrate_source = "FRED"
    if unrate_result:
        record("unrate", unrate_result[0], unrate_source, unrate_result[1], True)
        ur_3m = get_fallback("UNRATE", 4)
        ur_12m = get_fallback("UNRATE", 13)
        if ur_3m:
            record("unrate_chg_3m", unrate_result[0] - ur_3m[0], f"{unrate_source}_mixed", unrate_result[1], True)
        if ur_12m:
            record("unrate_chg_12m", unrate_result[0] - ur_12m[0], f"{unrate_source}_mixed", unrate_result[1], True)
    else:
        for col, periods in [("unrate", 1), ("unrate_chg_3m", None), ("unrate_chg_12m", None)]:
            fb = get_fallback("UNRATE")
            fb3 = get_fallback("UNRATE", 4)
            fb12 = get_fallback("UNRATE", 13)
            if col == "unrate" and fb:
                record("unrate", fb[0], "FRED_monthly", fb[1], False)
            if col == "unrate_chg_3m" and fb and fb3:
                record("unrate_chg_3m", fb[0] - fb3[0], "FRED_monthly", fb[1], False)
            if col == "unrate_chg_12m" and fb and fb12:
                record("unrate_chg_12m", fb[0] - fb12[0], "FRED_monthly", fb[1], False)

    # Payrolls (PAYEMS) — try BLS first, fall back to FRED
    # BLS series CES0000000001 returns total nonfarm in THOUSANDS
    payems_result = _fetch_bls_latest(BLS_SERIES["PAYEMS"])
    payems_source = "BLS"
    if not payems_result:
        payems_result = _fetch_fred_latest("PAYEMS")
        payems_source = "FRED"
    if payems_result:
        pay_val = payems_result[0]
        pay_date = payems_result[1]
        pay_12m = get_fallback("PAYEMS", 13)
        pay_1m = get_fallback("PAYEMS", 2)
        if pay_12m:
            record("payems_yoy", ((pay_val / pay_12m[0]) - 1) * 100, payems_source, pay_date, True)
        if pay_1m:
            record("payems_mom", ((pay_val / pay_1m[0]) - 1) * 100, payems_source, pay_date, True)
        logger.info(f"Nowcast: Payrolls from {payems_source}: {pay_val:.0f}K as of {pay_date}")
    else:
        fb = get_fallback("PAYEMS")
        fb12 = get_fallback("PAYEMS", 13)
        fb2 = get_fallback("PAYEMS", 2)
        if fb and fb12:
            record("payems_yoy", ((fb[0] / fb12[0]) - 1) * 100, "FRED_monthly", fb[1], False)
        if fb and fb2:
            record("payems_mom", ((fb[0] / fb2[0]) - 1) * 100, "FRED_monthly", fb[1], False)

    # Consumer sentiment
    umcsent_result = _fetch_fred_latest("UMCSENT")
    if umcsent_result:
        record("umcsent", umcsent_result[0], "FRED", umcsent_result[1], True)
        um_3m = get_fallback("UMCSENT", 4)
        if um_3m:
            record("umcsent_chg", umcsent_result[0] - um_3m[0], "FRED_mixed", umcsent_result[1], True)
    else:
        fb = get_fallback("UMCSENT")
        fb3 = get_fallback("UMCSENT", 4)
        if fb:
            record("umcsent", fb[0], "FRED_monthly", fb[1], False)
        if fb and fb3:
            record("umcsent_chg", fb[0] - fb3[0], "FRED_monthly", fb[1], False)

    # Initial claims (weekly)
    icsa_result = _fetch_fred_latest("ICSA")
    if icsa_result:
        icsa_12m = get_fallback("ICSA", 13)
        if icsa_12m and icsa_12m[0] != 0:
            record("icsa_4wk_chg", ((icsa_result[0] / icsa_12m[0]) - 1) * 100, "FRED_weekly", icsa_result[1], True)
    else:
        fb = get_fallback("ICSA")
        fb12 = get_fallback("ICSA", 13)
        if fb and fb12 and fb12[0] != 0:
            record("icsa_4wk_chg", ((fb[0] / fb12[0]) - 1) * 100, "FRED_monthly", fb[1], False)

    # ─────────────────────────────────────────────────────────────────
    # TIER 3: Monthly data (usually lagged, carry-forward)
    # ─────────────────────────────────────────────────────────────────

    # Industrial production
    fb_indpro = get_fallback("INDPRO")
    fb_indpro_12 = get_fallback("INDPRO", 13)
    fb_indpro_1 = get_fallback("INDPRO", 2)
    if fb_indpro and fb_indpro_12:
        record("indpro_yoy", ((fb_indpro[0] / fb_indpro_12[0]) - 1) * 100, "FRED_monthly", fb_indpro[1], False)
    if fb_indpro and fb_indpro_1:
        record("indpro_mom", ((fb_indpro[0] / fb_indpro_1[0]) - 1) * 100, "FRED_monthly", fb_indpro[1], False)

    # CPI — try BLS first for faster updates
    cpi_result = _fetch_bls_latest(BLS_SERIES["CPI"])
    if cpi_result:
        # BLS CPI is an index level, need 12-month % change
        fb_cpi_12 = get_fallback("CPIAUCSL", 13)
        if fb_cpi_12:
            record("cpi_yoy", ((cpi_result[0] / fb_cpi_12[0]) - 1) * 100, "BLS", cpi_result[1], True)
        else:
            # Can't compute YoY without historical, use FRED fallback
            fb_cpi = get_fallback("CPIAUCSL")
            fb_cpi_12 = get_fallback("CPIAUCSL", 13)
            if fb_cpi and fb_cpi_12:
                record("cpi_yoy", ((fb_cpi[0] / fb_cpi_12[0]) - 1) * 100, "FRED_monthly", fb_cpi[1], False)
    else:
        fb_cpi = get_fallback("CPIAUCSL")
        fb_cpi_12 = get_fallback("CPIAUCSL", 13)
        if fb_cpi and fb_cpi_12:
            record("cpi_yoy", ((fb_cpi[0] / fb_cpi_12[0]) - 1) * 100, "FRED_monthly", fb_cpi[1], False)

    # Housing starts
    fb_houst = get_fallback("HOUST")
    fb_houst_12 = get_fallback("HOUST", 13)
    if fb_houst and fb_houst_12 and fb_houst_12[0] != 0:
        record("houst_yoy", ((fb_houst[0] / fb_houst_12[0]) - 1) * 100, "FRED_monthly", fb_houst[1], False)

    # M2 money supply
    fb_m2 = get_fallback("M2SL")
    fb_m2_12 = get_fallback("M2SL", 13)
    if fb_m2 and fb_m2_12:
        record("m2_yoy", ((fb_m2[0] / fb_m2_12[0]) - 1) * 100, "FRED_monthly", fb_m2[1], False)

    # CAPE (Shiller)
    cape_cache = DATA_DIR / "shiller_cape.pkl"
    if cape_cache.exists():
        shiller = pd.read_pickle(cape_cache)
        if "CAPE" in shiller.columns:
            cape_series = shiller["CAPE"].dropna()
            if len(cape_series) > 120:
                cape_val = float(cape_series.iloc[-1])
                cape_date = cape_series.index[-1].strftime("%Y-%m-%d")
                cape_mean = float(cape_series.rolling(120).mean().iloc[-1])
                cape_std = float(cape_series.rolling(120).std().iloc[-1])
                record("cape", cape_val, "Shiller_cache", cape_date, False)
                if cape_std > 0:
                    record("cape_zscore", (cape_val - cape_mean) / cape_std, "Shiller_cache", cape_date, False)

    # ─────────────────────────────────────────────────────────────────
    # Fill any missing features with 0 (shouldn't happen but safety net)
    # ─────────────────────────────────────────────────────────────────
    for col in FEATURE_COLS:
        if col not in features:
            logger.warning(f"Nowcast: Feature '{col}' not available, using 0.0")
            features[col] = 0.0
            metadata[col] = {"status": "missing", "value": 0.0, "as_of": "N/A", "source": "default"}

    # Build the single-row DataFrame in the correct column order
    feature_row = pd.DataFrame([features], columns=FEATURE_COLS)

    # Summary metadata
    live_count = sum(1 for v in metadata.values() if v["status"] == "live")
    imputed_count = sum(1 for v in metadata.values() if v["status"] == "imputed")
    missing_count = sum(1 for v in metadata.values() if v["status"] == "missing")

    summary = {
        "live_features": live_count,
        "imputed_features": imputed_count,
        "missing_features": missing_count,
        "total_features": len(FEATURE_COLS),
        "freshness_pct": round(live_count / len(FEATURE_COLS) * 100, 1),
        "last_updated": now.isoformat(),
        "feature_status": metadata,
    }

    logger.info(f"Nowcast: Built feature row — {live_count} live, {imputed_count} imputed, {missing_count} missing")

    return feature_row, summary


# ═══════════════════════════════════════════════════════════════════════
#  Nowcast Prediction
# ═══════════════════════════════════════════════════════════════════════


REGIME_LABELS = {0: "Expansion", 1: "Slowdown", 2: "Contraction", 3: "Recovery", 4: "Crisis"}


def run_nowcast(
    fred_monthly: Optional[pd.DataFrame] = None,
    classifier=None,
    model: str = "xgboost",
    neural_classifier=None,
) -> Dict[str, Any]:
    """
    Run the full nowcast pipeline:
    1. Fetch real-time features
    2. Scale using the trained model's scaler
    3. Predict using selected model (XGBoost or Neural Net)
    4. Return classification with metadata

    Args:
        fred_monthly: Official FRED monthly data for fallback values
        classifier: Trained EnsembleRegimeClassifier (contains .xgb, .scaler, .feature_cols)
        model: "xgboost" (default) or "neural_net"
        neural_classifier: Trained NeuralRegimeClassifier (for model="neural_net")

    Returns:
        Dict with regime, probabilities, confidence, freshness, signals
    """
    # Load classifier if not provided
    if classifier is None:
        model_path = MODEL_DIR / "ensemble_regime_classifier.pkl"
        if model_path.exists():
            import pickle
            with open(model_path, "rb") as f:
                classifier = pickle.load(f)
        else:
            return {"error": "No trained model found", "regime": "Unknown"}

    # Fetch real-time features
    feature_row, freshness = fetch_realtime_features(fred_monthly)

    # --- Neural Net prediction path ---
    if model == "neural_net":
        return _run_nowcast_neural(
            feature_row, freshness, fred_monthly, neural_classifier
        )

    # --- XGBoost prediction path (default) ---
    # Ensure columns match the model's expected order
    model_cols = classifier.feature_cols
    for col in model_cols:
        if col not in feature_row.columns:
            feature_row[col] = 0.0
            logger.warning(f"Nowcast: Model expects '{col}' but not in feature row, using 0.0")
    feature_row = feature_row[model_cols]

    # Scale using the trained scaler
    try:
        X_scaled = classifier.scaler.transform(feature_row.values)
    except Exception as e:
        logger.error(f"Nowcast scaling failed: {e}")
        return {"error": f"Scaling failed: {e}", "regime": "Unknown"}

    # Predict using XGBoost ONLY
    try:
        xgb_proba = classifier.xgb.predict_proba(X_scaled)[0]
        regime_idx = int(np.argmax(xgb_proba))
        regime_name = REGIME_LABELS.get(regime_idx, "Unknown")
        confidence = float(xgb_proba[regime_idx])
    except Exception as e:
        logger.error(f"Nowcast prediction failed: {e}")
        return {"error": f"Prediction failed: {e}", "regime": "Unknown"}

    # Build probability dict
    probabilities = {}
    for i, label in REGIME_LABELS.items():
        if i < len(xgb_proba):
            probabilities[label] = round(float(xgb_proba[i]), 4)
        else:
            probabilities[label] = 0.0

    # Identify key signals firing
    signals = []
    vals = feature_row.iloc[0]
    if "sp500_mom_3m" in vals and vals["sp500_mom_3m"] < 0:
        signals.append(f"S&P 500 3M momentum: {vals['sp500_mom_3m']:.1f}%")
    if "sp500_yoy" in vals and vals["sp500_yoy"] < 0:
        signals.append(f"S&P 500 YoY: {vals['sp500_yoy']:.1f}%")
    if "vix" in vals and vals["vix"] > 25:
        signals.append(f"VIX elevated: {vals['vix']:.1f}")
    if "credit_spread" in vals and vals["credit_spread"] > 3.0:
        signals.append(f"Credit spread wide: {vals['credit_spread']:.2f}%")
    if "credit_spread_chg_3m" in vals and vals["credit_spread_chg_3m"] > 0.3:
        signals.append(f"Credit spread widening: +{vals['credit_spread_chg_3m']:.2f}")
    if "spread_10y2y" in vals and vals["spread_10y2y"] < 0:
        signals.append(f"Yield curve inverted (10Y-2Y): {vals['spread_10y2y']:.2f}%")
    if "unrate_chg_3m" in vals and vals["unrate_chg_3m"] > 0.3:
        signals.append(f"Unemployment rising: +{vals['unrate_chg_3m']:.1f} (3M)")
    if "umcsent" in vals and vals["umcsent"] < 65:
        signals.append(f"Consumer sentiment low: {vals['umcsent']:.1f}")

    # ── Tax Withholding Signal (display-only, not a model feature) ──
    # Fetches 3-month avg YoY growth from Treasury Daily Statement.
    # This is independent of XGBoost — purely informational for the PM.
    withholding_data = {}
    try:
        def _fetch_wh_period(start_date, end_date):
            """Fetch withholding total for a date range from Treasury API."""
            for catg in ["Taxes - Withheld Individual/FICA",
                         "Withheld Income and Employment Taxes"]:
                params = {
                    "fields": "record_date,transaction_today_amt",
                    "filter": f"transaction_catg:eq:{catg},"
                              f"transaction_type:eq:Deposits,"
                              f"record_date:gte:{start_date},record_date:lt:{end_date}",
                    "page[size]": 500,
                }
                resp = requests.get(
                    f"{TREASURY_API_BASE}{TREASURY_DTS_ENDPOINT}",
                    params=params, timeout=15,
                )
                resp.raise_for_status()
                records = resp.json().get("data", [])
                total = sum(float(r["transaction_today_amt"]) for r in records
                            if r.get("transaction_today_amt"))
                if total > 0:
                    return total, len(records)
            return 0, 0

        now_dt = datetime.now()
        months_yoy = []

        # Fetch current month + prior 2 months (for 3-month average)
        for months_back in range(1,4):  # Skip partal month and go to full 3 months prior
            m = now_dt.month - months_back
            y = now_dt.year
            if m <= 0:
                m += 12
                y -= 1

            # Current/recent month range
            pm_start = f"{y}-{m:02d}-01"
            if m == 12:
                pm_end = f"{y + 1}-01-01"
            else:
                pm_end = f"{y}-{m + 1:02d}-01"

            # For current month, use today as end date
            if months_back == 0:
                pm_end = now_dt.strftime("%Y-%m-%d")

            # Same month last year
            pm_ly_start = f"{y - 1}-{m:02d}-01"
            if m == 12:
                pm_ly_end = f"{y}-01-01"
            else:
                pm_ly_end = f"{y - 1}-{m + 1:02d}-01"

            wh_cur, cur_days = _fetch_wh_period(pm_start, pm_end)
            wh_ly, ly_days = _fetch_wh_period(pm_ly_start, pm_ly_end)

            if wh_cur > 0 and wh_ly > 0:
                if months_back == 0 and cur_days >= 5 and ly_days > 0:
                    # Current month MTD: normalize by business days
                    daily_avg_ly = wh_ly / ly_days
                    comparable_ly = daily_avg_ly * cur_days
                    yoy_pct = ((wh_cur / comparable_ly) - 1) * 100
                    months_yoy.append(yoy_pct)
                elif months_back > 0:
                    # Completed months: direct comparison
                    yoy_pct = ((wh_cur / wh_ly) - 1) * 100
                    months_yoy.append(yoy_pct)

        if months_yoy:
            wh_3m_avg = sum(months_yoy) / len(months_yoy)
            withholding_data = {
                "withholding_yoy_3m": round(wh_3m_avg, 1),
                "months_used": len(months_yoy),
                "monthly_values": [round(v, 1) for v in months_yoy],
            }

            # Fire signal based on thresholds
            if wh_3m_avg < 0:
                signals.append(f"Tax withholding DECLINING: {wh_3m_avg:.1f}% (3M avg YoY)")
            elif wh_3m_avg < 3.0:
                signals.append(f"Tax withholding growth weak: {wh_3m_avg:.1f}% (3M avg YoY)")
            elif wh_3m_avg < 6.0:
                signals.append(f"Tax withholding growth slowing: {wh_3m_avg:.1f}% (3M avg YoY)")

            logger.info(f"Nowcast: Tax withholding 3M avg YoY: {wh_3m_avg:.1f}% "
                        f"(months: {[round(v, 1) for v in months_yoy]})")
    except Exception as e:
        logger.warning(f"Nowcast: Tax withholding signal fetch failed: {e}")

    result = {
        "regime": regime_name,
        "probabilities": probabilities,
        "confidence": round(confidence, 4),
        "data_freshness": freshness,
        "signals_firing": signals,
        "feature_values": {col: round(float(vals[col]), 4) for col in model_cols},
        "withholding_signal": withholding_data,
        "model": "XGBoost-only (no HMM/GMM)",
        "model_used": "xgboost",
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(f"Nowcast result: {regime_name} ({confidence:.1%} confidence), {len(signals)} signals firing")

    return result


def _run_nowcast_neural(
    feature_row: pd.DataFrame,
    freshness: Dict[str, Any],
    fred_monthly: Optional[pd.DataFrame],
    neural_classifier,
) -> Dict[str, Any]:
    """
    Run nowcast using the Neural Net (LSTM) classifier.
    Needs 12 months of history for the LSTM sequence.
    """
    if neural_classifier is None:
        # Try to load from disk
        nn_path = MODEL_DIR / "neural_regime_classifier.pt"
        if nn_path.exists():
            try:
                from models.neural_regime_classifier import NeuralRegimeClassifier
                neural_classifier = NeuralRegimeClassifier.load(nn_path)
            except Exception as e:
                return {"error": f"Failed to load neural classifier: {e}", "regime": "Unknown"}
        else:
            return {"error": "Neural net model not trained yet", "regime": "Unknown"}

    # Build historical context from FRED monthly data
    # The LSTM needs seq_len (12) months of history
    history = None
    if fred_monthly is not None:
        try:
            from data.ingestion import build_feature_matrix, fetch_shiller_cape
            shiller = fetch_shiller_cape()
            hist_features = build_feature_matrix(fred_monthly, shiller)
            hist_features = hist_features.drop(columns=["recession"], errors="ignore")

            # Add experimental features (oil_yoy) to history
            from models.feature_registry import FeatureRegistry
            registry = FeatureRegistry()
            extra = registry.compute_experimental_features(fred_monthly)
            if not extra.empty:
                common_idx = hist_features.index.intersection(extra.index)
                extra_aligned = extra.reindex(hist_features.index)
                hist_features = pd.concat([hist_features, extra_aligned], axis=1)

            # Add experimental feature values to the current feature_row
            # so the LSTM sees all 28 features for the latest month too
            if not extra.empty:
                last_extra = extra.iloc[-1:]
                for col in last_extra.columns:
                    if col not in feature_row.columns:
                        feature_row[col] = last_extra[col].values[0]

            # Take last (seq_len - 1) months as history
            seq_len = neural_classifier.seq_len
            if len(hist_features) >= seq_len - 1:
                history = hist_features.iloc[-(seq_len - 1):]
        except Exception as e:
            logger.warning(f"Nowcast neural: Failed to build history: {e}")

    # Predict using neural net
    try:
        nn_result = neural_classifier.predict_single(feature_row, history)
        regime_name = nn_result["regime"]
        probabilities = nn_result["probabilities"]
        confidence = nn_result["confidence"]
    except Exception as e:
        logger.error(f"Neural net nowcast prediction failed: {e}")
        return {"error": f"Neural net prediction failed: {e}", "regime": "Unknown"}

    # Signals (same logic as XGBoost path)
    signals = []
    vals = feature_row.iloc[0]
    if "sp500_mom_3m" in vals and vals["sp500_mom_3m"] < 0:
        signals.append(f"S&P 500 3M momentum: {vals['sp500_mom_3m']:.1f}%")
    if "sp500_yoy" in vals and vals["sp500_yoy"] < 0:
        signals.append(f"S&P 500 YoY: {vals['sp500_yoy']:.1f}%")
    if "vix" in vals and vals["vix"] > 25:
        signals.append(f"VIX elevated: {vals['vix']:.1f}")
    if "credit_spread" in vals and vals["credit_spread"] > 3.0:
        signals.append(f"Credit spread wide: {vals['credit_spread']:.2f}%")
    if "spread_10y2y" in vals and vals["spread_10y2y"] < 0:
        signals.append(f"Yield curve inverted (10Y-2Y): {vals['spread_10y2y']:.2f}%")
    if "unrate_chg_3m" in vals and vals["unrate_chg_3m"] > 0.3:
        signals.append(f"Unemployment rising: +{vals['unrate_chg_3m']:.1f} (3M)")

    result = {
        "regime": regime_name,
        "probabilities": probabilities,
        "confidence": round(confidence, 4),
        "data_freshness": freshness,
        "signals_firing": signals,
        "model": f"LSTM Neural Net ({len(feature_row.columns)} features)",
        "model_used": "neural_net",
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(
        f"Nowcast (neural_net): {regime_name} ({confidence:.1%} confidence), "
        f"{len(signals)} signals"
    )
    return result
