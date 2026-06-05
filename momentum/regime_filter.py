"""SPY 200-day moving average regime filter."""
import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

_SPY_CACHE_TTL_DAYS = 1  # Refresh daily for live use


def fetch_spy_daily(start: str, end: str) -> pd.Series:
    """Download daily SPY close prices, cached for 1 day."""
    start_key = start[:7].replace("-", "")
    end_key = end[:7].replace("-", "")
    cache_path = DATA_DIR / f"spy_daily_{start_key}_{end_key}.pkl"

    if cache_path.exists():
        age = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days
        if age < _SPY_CACHE_TTL_DAYS:
            cached = pd.read_pickle(cache_path)
            if not cached.empty:
                logger.info(f"Using cached SPY daily prices ({len(cached)} days)")
                return cached

    logger.info(f"Downloading daily SPY prices ({start} to {end})...")
    raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        logger.warning("Failed to download SPY daily prices")
        return pd.Series(dtype=float)

    prices = raw["Close"].squeeze()
    prices.index = pd.DatetimeIndex(prices.index)
    prices.to_pickle(cache_path)
    logger.info(f"SPY daily prices cached: {len(prices)} days")
    return prices


def is_above_200d_ma(rebal_date: pd.Timestamp, spy_daily: pd.Series) -> bool:
    """
    Returns True if SPY close is above its 200-day simple MA as of rebal_date.
    Defaults to True (stay invested in stocks) if insufficient data.
    """
    data = spy_daily[spy_daily.index <= rebal_date]
    if len(data) < 200:
        return True  # Not enough history — default to invested

    ma_200 = float(data.iloc[-200:].mean())
    current = float(data.iloc[-1])
    above = current > ma_200
    logger.debug(
        f"  {rebal_date.date()} SPY={current:.2f} 200d-MA={ma_200:.2f} "
        f"→ {'STOCKS' if above else 'IEF'}"
    )
    return above
