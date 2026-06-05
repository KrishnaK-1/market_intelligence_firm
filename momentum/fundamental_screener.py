"""FMP revenue growth screener for momentum universe stocks."""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

_CACHE_PATH = DATA_DIR / "fundamental_cache.pkl"
_CACHE_TTL_DAYS = 90      # Revenue data changes quarterly
_DAILY_API_LIMIT = 240    # Stay safely under FMP free-tier 250/day limit


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return pd.read_pickle(_CACHE_PATH)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    pd.to_pickle(cache, _CACHE_PATH)


def _fetch_growth(ticker: str, api_key: str) -> Optional[float]:
    """
    Fetch YoY TTM revenue growth from FMP.
    Computes (TTM_recent - TTM_prior) / |TTM_prior| using last 8 quarters.
    Returns None on any failure — caller treats None as passing the filter.
    """
    url = (
        f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}"
        f"?period=quarter&limit=8&apikey={api_key}"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or len(data) < 8:
            return None
        ttm_recent = sum((q.get("revenue") or 0) for q in data[:4])
        ttm_prior  = sum((q.get("revenue") or 0) for q in data[4:8])
        if ttm_prior == 0:
            return None
        return (ttm_recent - ttm_prior) / abs(ttm_prior)
    except Exception as e:
        logger.debug(f"FMP fetch failed for {ticker}: {e}")
        return None


def build_growth_map(tickers: List[str], api_key: str) -> Dict[str, Optional[float]]:
    """
    Returns {ticker: yoy_revenue_growth} for all tickers.

    - Uses a local 90-day cache so data is only re-fetched quarterly.
    - Fetches at most DAILY_API_LIMIT uncached tickers per run to stay within
      the FMP free-tier limit. Remaining uncached tickers return None (pass filter).
    """
    if not api_key:
        logger.warning("No FMP API key configured — revenue screener disabled, all stocks pass")
        return {}

    cache = _load_cache()
    now = datetime.now()
    cutoff = now - timedelta(days=_CACHE_TTL_DAYS)
    calls_made = 0

    for ticker in tickers:
        entry = cache.get(ticker)
        if entry and entry.get("fetched_at", datetime.min) > cutoff:
            continue  # Fresh cache entry, skip
        if calls_made >= _DAILY_API_LIMIT:
            logger.info(f"FMP daily limit reached ({_DAILY_API_LIMIT}). Remaining tickers pass by default.")
            break
        cache[ticker] = {"growth": _fetch_growth(ticker, api_key), "fetched_at": now}
        calls_made += 1

    if calls_made > 0:
        _save_cache(cache)
        logger.info(f"FMP: fetched {calls_made} tickers ({len(cache)} total cached)")

    return {t: (cache[t]["growth"] if t in cache else None) for t in tickers}


def apply_revenue_filter(
    scores: dict,
    growth_map: dict,
    threshold: float = 0.10,
    soft_multiplier: float = 1.2,
) -> dict:
    """
    Filter momentum scores by revenue growth threshold.

    - Hard filter (remove failing stocks) if 50+ stocks pass the threshold.
    - Soft weight (1.2x multiplier on passing stocks) if fewer than 50 pass.
    - Stocks with no growth data (None) are treated as passing.
    """
    passing = {
        t for t in scores
        if growth_map.get(t) is None or (growth_map.get(t) or 0) >= threshold
    }

    if len(passing) >= 50:
        return {t: s for t, s in scores.items() if t in passing}

    # Soft weight fallback
    return {
        t: s * (soft_multiplier if t in passing else 1.0)
        for t, s in scores.items()
    }
