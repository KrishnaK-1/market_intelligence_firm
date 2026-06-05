"""
Cross-Sectional Momentum Backtest
===================================
Monthly rebalancing on 12-1 momentum signal across S&P 500 constituents.
Universe is point-in-time accurate using sp500_ticker_start_end.csv.
Equal-weight top-50. Gross returns only.

v2 additions:
  - 200-day MA regime filter: moves to IEF when SPY is below 200-day MA
  - FMP fundamental screener: filters by YoY revenue growth before ranking
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import DATA_DIR, FMP_API_KEY, REVENUE_GROWTH_THRESHOLD
from momentum.fundamental_screener import apply_revenue_filter, build_growth_map
from momentum.regime_filter import fetch_spy_daily, is_above_200d_ma
from momentum.universe import get_constituents, get_universe_for_window, load_constituent_data

logger = logging.getLogger(__name__)

CACHE_TTL_DAYS = 7
BATCH_SIZE = 200


def _get_cache_path(start: str, end: str):
    # Cache is keyed by date range so different windows never collide.
    start_key = start[:7].replace("-", "")
    end_key = end[:7].replace("-", "")
    return DATA_DIR / f"momentum_prices_{start_key}_{end_key}.pkl"


def _cache_is_fresh(cache_path) -> bool:
    if not cache_path.exists():
        return False
    age_days = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days
    return age_days < CACHE_TTL_DAYS


def fetch_stock_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download monthly adjusted close prices for all tickers, cached for 7 days."""
    cache_path = _get_cache_path(start, end)

    if _cache_is_fresh(cache_path):
        cached = pd.read_pickle(cache_path)
        coverage = len(set(cached.columns) & set(tickers)) / max(len(tickers), 1)
        if coverage >= 0.85:
            logger.info(f"Using cached prices ({coverage:.0%} coverage, {len(cached)} months)")
            return cached

    logger.info(f"Downloading prices for {len(tickers)} tickers ({start} to {end})...")
    total_batches = (len(tickers) - 1) // BATCH_SIZE + 1
    frames = []

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        try:
            raw = yf.download(
                batch, start=start, end=end,
                auto_adjust=True, threads=True, progress=False,
            )
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw.to_frame(name=batch[0]) if len(batch) == 1 else raw
            frames.append(prices)
            logger.info(f"  Batch {batch_num}/{total_batches}: {prices.shape[1]} tickers")
        except Exception as e:
            logger.warning(f"  Batch {batch_num} failed: {e}")

    if not frames:
        return pd.DataFrame()

    all_prices = pd.concat(frames, axis=1)
    all_prices = all_prices.loc[:, ~all_prices.columns.duplicated()]
    all_prices.index = pd.DatetimeIndex(all_prices.index)
    all_prices = all_prices.resample("ME").last()

    all_prices.to_pickle(cache_path)
    logger.info(f"Prices cached: {all_prices.shape[0]} months x {all_prices.shape[1]} tickers")
    return all_prices


def _compute_stats(returns: pd.Series, name: str) -> Dict:
    if len(returns) < 3:
        return {"name": name, "error": "Insufficient data"}
    equity = (1 + returns).cumprod()
    n_years = len(returns) / 12
    ann_return = float(equity.iloc[-1] ** (1 / n_years) - 1)
    ann_vol = float(returns.std() * np.sqrt(12))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    peak = equity.cummax()
    max_dd = float(((equity - peak) / peak).min())
    return {
        "name": name,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_dd,
        "total_return": float(equity.iloc[-1] - 1),
        "best_month": float(returns.max()),
        "worst_month": float(returns.min()),
        "pct_positive_months": float((returns > 0).mean()),
        "n_months": int(len(returns)),
    }


def run_single_backtest(
    years: int,
    prices: pd.DataFrame,
    constituent_df: pd.DataFrame,
    spy_daily: Optional[pd.Series] = None,
    growth_map: Optional[dict] = None,
    revenue_threshold: float = REVENUE_GROWTH_THRESHOLD,
) -> Dict:
    """Run cross-sectional momentum backtest for the given lookback window."""
    end_date = prices.index[-1]
    start_date = end_date - pd.DateOffset(years=years)

    if start_date < prices.index[0] + pd.DateOffset(months=14):
        return {"error": f"Insufficient price history for {years}Y backtest"}

    logger.info(f"  {years}Y backtest: {start_date.date()} → {end_date.date()}")

    monthly_returns = prices.pct_change()
    rebalance_dates = prices.loc[start_date:].index

    portfolio_returns = []
    holdings_history = []
    months_in_ief = 0
    months_in_stocks = 0
    prev_holdings: set = set()  # tracks last month's positions for tx cost

    for rebal_date in rebalance_dates:
        future = monthly_returns.index[monthly_returns.index > rebal_date]
        if len(future) == 0:
            break
        next_date = future[0]

        # ── 200-day MA regime filter ──────────────────────────────────────
        if spy_daily is not None and not is_above_200d_ma(rebal_date, spy_daily):
            ief_ret = monthly_returns.loc[next_date, "IEF"] if "IEF" in monthly_returns.columns else 0.0
            if pd.isna(ief_ret):
                ief_ret = 0.0
            # Tx cost: if switching from stocks to IEF, charge for liquidating
            tx_cost = (len(prev_holdings) / 50) * 0.001 if prev_holdings and prev_holdings != {"IEF"} else 0.0
            prev_holdings = {"IEF"}
            net_ret = float(np.clip(ief_ret, -0.40, 0.50)) - tx_cost
            portfolio_returns.append({"date": next_date, "return": net_ret})
            holdings_history.append({"date": rebal_date, "holdings": ["IEF"]})
            months_in_ief += 1
            continue

        # ── 12-1 momentum signal ──────────────────────────────────────────
        signal_end = rebal_date - pd.DateOffset(months=1)
        signal_start = rebal_date - pd.DateOffset(months=13)

        constituents = get_constituents(rebal_date, constituent_df)
        available = [t for t in constituents if t in prices.columns]
        if not available:
            continue

        price_slice = prices.loc[signal_start:signal_end, available]
        scores = {}
        for ticker in available:
            col = price_slice[ticker].dropna()
            if len(col) >= 11:
                scores[ticker] = float(col.iloc[-1] / col.iloc[0] - 1)

        if len(scores) < 50:
            continue

        # ── Positive momentum filter ──────────────────────────────────────
        scores = {t: s for t, s in scores.items() if s > 0}

        if len(scores) < 50:
            continue

        # ── Revenue growth filter ─────────────────────────────────────────
        if growth_map:
            scores = apply_revenue_filter(scores, growth_map, revenue_threshold)

        if len(scores) < 50:
            continue

        top50 = sorted(scores, key=scores.__getitem__, reverse=True)[:50]
        holdings_history.append({"date": rebal_date, "holdings": top50})
        months_in_stocks += 1

        # ── Volatility-adjusted weights ───────────────────────────────────
        avail_rets = monthly_returns.loc[next_date, top50].dropna()
        avail_rets = avail_rets.clip(-0.40, 0.50)
        if len(avail_rets) < 5:
            continue

        vol_start = rebal_date - pd.DateOffset(months=3)
        vol_window = monthly_returns.loc[vol_start:rebal_date, avail_rets.index]
        vols = {}
        for ticker in avail_rets.index:
            ticker_rets = vol_window[ticker].dropna()
            if len(ticker_rets) >= 2:
                vols[ticker] = max(float(ticker_rets.std()), 0.001)
            else:
                fallback = monthly_returns[ticker].dropna()
                vols[ticker] = max(float(fallback.std()) if len(fallback) > 1 else 0.01, 0.001)

        inv_vols = {t: 1.0 / v for t, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        weights = pd.Series({t: iv / total_inv_vol for t, iv in inv_vols.items()})
        weighted_ret = float((avail_rets * weights).sum())

        # ── Transaction cost (10bps round-trip per traded position) ───────
        top50_set = set(top50)
        if not prev_holdings:
            tx_cost = 0.0  # first rebalance, no prior state
        elif prev_holdings == {"IEF"}:
            tx_cost = 0.001  # exiting IEF: sell IEF + buy full portfolio
        else:
            n_changed = len(top50_set.symmetric_difference(prev_holdings))
            tx_cost = (n_changed / 50) * 0.001
        prev_holdings = top50_set

        portfolio_returns.append({"date": next_date, "return": weighted_ret - tx_cost})

    if len(portfolio_returns) < 6:
        return {"error": f"Only {len(portfolio_returns)} months of data for {years}Y backtest"}

    ret_series = (
        pd.DataFrame(portfolio_returns).set_index("date")["return"].sort_index()
    )
    equity = (1 + ret_series).cumprod()
    current_holdings = holdings_history[-1]["holdings"] if holdings_history else []

    return {
        "years": years,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "stats": _compute_stats(ret_series, f"{years}Y Momentum"),
        "equity_curve": {
            "dates": [d.strftime("%Y-%m-%d") for d in equity.index],
            "values": [round(float(v), 4) for v in equity.values],
        },
        "current_holdings": current_holdings,
        "months_in_ief": months_in_ief,
        "months_in_stocks": months_in_stocks,
    }


def run_all_backtests(end_date: str = None, revenue_threshold: float = None) -> Dict:
    """
    Main entry point. Downloads prices once for the 10Y window,
    then runs 2Y / 5Y / 10Y backtests with regime filter and revenue screener.

    Args:
        end_date: Optional cutoff date string like "2014-12-31". Defaults to today.
        revenue_threshold: YoY revenue growth minimum. Defaults to settings value.
    """
    if revenue_threshold is None:
        revenue_threshold = REVENUE_GROWTH_THRESHOLD

    constituent_df = load_constituent_data()

    if end_date is not None:
        end_dt = pd.Timestamp(end_date).normalize()
    else:
        end_dt = pd.Timestamp.today().normalize()

    data_start = end_dt - pd.DateOffset(years=10) - pd.DateOffset(months=15)

    all_tickers = list(get_universe_for_window(data_start, end_dt, constituent_df))
    if "IEF" not in all_tickers:
        all_tickers.append("IEF")
    logger.info(f"10Y universe: {len(all_tickers)} unique tickers (incl. IEF)")

    prices = fetch_stock_prices(
        all_tickers,
        start=data_start.strftime("%Y-%m-%d"),
        end=(end_dt + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
    )

    if prices.empty:
        return {"error": "Failed to fetch price data from Yahoo Finance"}

    # Daily SPY prices — needs extra 10-month lead so the MA is computable
    # at the very start of the backtest window
    spy_start = data_start - pd.DateOffset(months=10)
    spy_daily = fetch_spy_daily(
        spy_start.strftime("%Y-%m-%d"),
        (end_dt + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
    )

    # Revenue growth map from FMP (cached 90 days, fetches up to 240 tickers/run)
    growth_map = build_growth_map(all_tickers, FMP_API_KEY) if FMP_API_KEY else {}

    results = {}
    for years in [2, 5, 10]:
        try:
            results[f"{years}y"] = run_single_backtest(
                years, prices, constituent_df,
                spy_daily=spy_daily if not spy_daily.empty else None,
                growth_map=growth_map or None,
                revenue_threshold=revenue_threshold,
            )
        except Exception as e:
            logger.error(f"{years}Y backtest failed: {e}", exc_info=True)
            results[f"{years}y"] = {"error": str(e)}

    current_holdings = (
        results.get("2y", {}).get("current_holdings")
        or results.get("5y", {}).get("current_holdings")
        or []
    )

    return {
        "backtests": results,
        "current_holdings": current_holdings,
        "currently_in_ief": current_holdings == ["IEF"],
        "universe_size": len(all_tickers),
        "revenue_threshold": revenue_threshold,
        "computed_at": datetime.now().isoformat(),
        "note": (
            f"Gross returns. Equal-weight top-50 by 12-1 momentum. "
            f"200-day MA regime filter: moves to IEF when SPY < 200d MA. "
            f"Revenue screener: {revenue_threshold:.0%} YoY growth threshold "
            f"(hard filter if 50+ qualify, soft 1.2x weight otherwise). "
            f"Universe: point-in-time S&P 500 constituents (1996-2026)."
        ),
    }
