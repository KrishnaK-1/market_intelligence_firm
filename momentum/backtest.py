"""
Cross-Sectional Momentum Backtest
===================================
Monthly rebalancing on 12-1 momentum signal across S&P 500 constituents.
Universe is point-in-time accurate using sp500_ticker_start_end.csv.

v2 additions:
  - 200-day MA regime filter: moves to IEF (or cash) when SPY is below 200-day MA
  - FMP fundamental screener: filters by YoY revenue growth before ranking
  - Volatility-adjusted position sizing (inverse-vol weights)
  - Positive momentum filter
  - Transaction cost model (10 bps per traded position)
  - Holdings history with per-stock returns, added/dropped diffs
  - Comparison simulations: cash vs bonds, equal-weight vs vol-adjusted
  - Benchmarks: SPY buy-and-hold, equal-weight S&P 500
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
    use_bonds: bool = True,
    vol_adjusted: bool = True,
) -> Dict:
    """
    Run cross-sectional momentum backtest for the given lookback window.

    Args:
        use_bonds: True = move to IEF when SPY < 200d MA. False = hold cash (0% return).
        vol_adjusted: True = inverse-volatility weights. False = equal weight.
    """
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
    months_in_cash = 0
    prev_holdings: set = set()

    for rebal_date in rebalance_dates:
        future = monthly_returns.index[monthly_returns.index > rebal_date]
        if len(future) == 0:
            break
        next_date = future[0]

        # ── 200-day MA regime filter ──────────────────────────────────────
        if spy_daily is not None and not is_above_200d_ma(rebal_date, spy_daily):
            if use_bonds:
                ief_ret = monthly_returns.loc[next_date, "IEF"] if "IEF" in monthly_returns.columns else 0.0
                if pd.isna(ief_ret):
                    ief_ret = 0.0
                tx_cost = (len(prev_holdings) / 50) * 0.001 if prev_holdings and prev_holdings != {"IEF"} else 0.0
                net_ret = float(np.clip(ief_ret, -0.40, 0.50)) - tx_cost

                added_h = [] if prev_holdings == {"IEF"} else ["IEF"]
                dropped_h = sorted(prev_holdings - {"IEF"})[:20] if prev_holdings and prev_holdings != {"IEF"} else []
                holdings_history.append({
                    "date": rebal_date.strftime("%Y-%m-%d"),
                    "holdings": ["IEF"],
                    "returns": {"IEF": round(float(np.clip(ief_ret, -0.40, 0.50)), 4)},
                    "weights": {"IEF": 1.0},
                    "added": added_h,
                    "dropped": dropped_h,
                    "in_ief": True,
                    "in_cash": False,
                    "portfolio_return": round(net_ret, 4),
                })
                prev_holdings = {"IEF"}
                months_in_ief += 1
            else:
                tx_cost = (len(prev_holdings) / 50) * 0.001 if prev_holdings and prev_holdings not in [set(), {"CASH"}, {"IEF"}] else 0.0
                dropped_h = sorted(prev_holdings - {"CASH"})[:20] if prev_holdings and prev_holdings not in [set(), {"CASH"}] else []
                net_ret = -tx_cost
                holdings_history.append({
                    "date": rebal_date.strftime("%Y-%m-%d"),
                    "holdings": ["CASH"],
                    "returns": {},
                    "weights": {},
                    "added": [],
                    "dropped": dropped_h,
                    "in_ief": False,
                    "in_cash": True,
                    "portfolio_return": round(net_ret, 4),
                })
                prev_holdings = {"CASH"}
                months_in_cash += 1

            portfolio_returns.append({"date": next_date, "return": net_ret})
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
        months_in_stocks += 1

        # ── Next-month returns ────────────────────────────────────────────
        avail_rets = monthly_returns.loc[next_date, top50].dropna()
        avail_rets = avail_rets.clip(-0.40, 0.50)
        if len(avail_rets) < 5:
            continue

        # ── Weights: vol-adjusted or equal ───────────────────────────────
        if vol_adjusted:
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
        else:
            n = len(avail_rets)
            weights = pd.Series(1.0 / n, index=avail_rets.index)

        weighted_ret = float((avail_rets * weights).sum())

        # ── Transaction cost ──────────────────────────────────────────────
        top50_set = set(top50)
        if not prev_holdings:
            tx_cost = 0.0
        elif prev_holdings in [{"IEF"}, {"CASH"}]:
            tx_cost = 0.001
        else:
            n_changed = len(top50_set.symmetric_difference(prev_holdings))
            tx_cost = (n_changed / 50) * 0.001

        # ── Holdings history entry ────────────────────────────────────────
        prev_was_stocks = prev_holdings and prev_holdings not in [{"IEF"}, {"CASH"}]
        added_h = sorted(top50_set - prev_holdings)[:20] if prev_was_stocks else []
        dropped_h = sorted(prev_holdings - top50_set)[:20] if prev_was_stocks else []

        rets_display = {t: round(float(avail_rets[t]), 4) for t in avail_rets.index}
        wts_display = {t: round(float(weights[t]), 4) for t in weights.index if t in avail_rets.index}
        # Sort holdings by return descending for display
        top50_by_ret = sorted(top50, key=lambda t: rets_display.get(t, float("-inf")), reverse=True)

        holdings_history.append({
            "date": rebal_date.strftime("%Y-%m-%d"),
            "holdings": top50_by_ret,
            "returns": rets_display,
            "weights": wts_display,
            "added": added_h,
            "dropped": dropped_h,
            "in_ief": False,
            "in_cash": False,
            "portfolio_return": round(weighted_ret - tx_cost, 4),
        })
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
        "holdings_history": holdings_history[-6:],
        "months_in_ief": months_in_ief,
        "months_in_stocks": months_in_stocks,
        "months_in_cash": months_in_cash,
    }


def _spy_benchmark(prices: pd.DataFrame, years: int, spy_daily: Optional[pd.Series] = None) -> Dict:
    """Buy-and-hold SPY benchmark. Uses monthly prices if cached, otherwise resamples daily data."""
    end_date = prices.index[-1]
    start_date = end_date - pd.DateOffset(years=years)

    if "SPY" in prices.columns:
        spy_monthly = prices["SPY"].loc[start_date:]
    elif spy_daily is not None and not spy_daily.empty:
        # Resample daily → month-end so the index aligns with the rest of the backtest
        spy_monthly = spy_daily.resample("ME").last().loc[start_date:]
    else:
        return {"error": "SPY data not available"}

    spy_rets = spy_monthly.pct_change().dropna().clip(-0.40, 0.50)
    if len(spy_rets) < 3:
        return {"error": "Insufficient SPY data"}
    equity = (1 + spy_rets).cumprod()
    return {
        "name": "SPY Buy & Hold",
        "stats": _compute_stats(spy_rets, "SPY Buy & Hold"),
        "equity_curve": {
            "dates": [d.strftime("%Y-%m-%d") for d in equity.index],
            "values": [round(float(v), 4) for v in equity.values],
        },
    }


def _equal_weight_sp500(prices: pd.DataFrame, constituent_df: pd.DataFrame, years: int) -> Dict:
    """Equal-weight all S&P 500 constituents, monthly rebalanced."""
    end_date = prices.index[-1]
    start_date = end_date - pd.DateOffset(years=years)
    monthly_rets = prices.pct_change()
    rebal_dates = prices.loc[start_date:].index

    port_returns = []
    for rebal_date in rebal_dates:
        future = monthly_rets.index[monthly_rets.index > rebal_date]
        if len(future) == 0:
            break
        next_date = future[0]
        constituents = get_constituents(rebal_date, constituent_df)
        available = [t for t in constituents if t in prices.columns]
        if not available:
            continue
        rets = monthly_rets.loc[next_date, available].dropna().clip(-0.40, 0.50)
        if len(rets) < 10:
            continue
        port_returns.append({"date": next_date, "return": float(rets.mean())})

    if len(port_returns) < 6:
        return {"error": "Insufficient data for equal-weight benchmark"}

    ret_series = pd.DataFrame(port_returns).set_index("date")["return"].sort_index()
    equity = (1 + ret_series).cumprod()
    return {
        "name": "S&P 500 Equal-Weight",
        "stats": _compute_stats(ret_series, "S&P 500 Equal-Weight"),
        "equity_curve": {
            "dates": [d.strftime("%Y-%m-%d") for d in equity.index],
            "values": [round(float(v), 4) for v in equity.values],
        },
    }


def run_all_backtests(end_date: str = None, revenue_threshold: float = None) -> Dict:
    """
    Main entry point. Downloads prices once for the 10Y window,
    then runs 2Y / 5Y / 10Y backtests plus comparison simulations and benchmarks.
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
    if "SPY" not in all_tickers:
        all_tickers.append("SPY")
    logger.info(f"10Y universe: {len(all_tickers)} unique tickers (incl. IEF, SPY)")

    prices = fetch_stock_prices(
        all_tickers,
        start=data_start.strftime("%Y-%m-%d"),
        end=(end_dt + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
    )

    if prices.empty:
        return {"error": "Failed to fetch price data from Yahoo Finance"}

    spy_start = data_start - pd.DateOffset(months=10)
    spy_daily = fetch_spy_daily(
        spy_start.strftime("%Y-%m-%d"),
        (end_dt + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
    )

    growth_map = build_growth_map(all_tickers, FMP_API_KEY) if FMP_API_KEY else {}
    spy_d = spy_daily if not spy_daily.empty else None
    gm = growth_map or None

    # ── Primary backtests (current strategy) ─────────────────────────────
    results = {}
    for years in [2, 5, 10]:
        try:
            results[f"{years}y"] = run_single_backtest(
                years, prices, constituent_df,
                spy_daily=spy_d, growth_map=gm,
                revenue_threshold=revenue_threshold,
                use_bonds=True, vol_adjusted=True,
            )
        except Exception as e:
            logger.error(f"{years}Y backtest failed: {e}", exc_info=True)
            results[f"{years}y"] = {"error": str(e)}

    # ── Comparison simulations ────────────────────────────────────────────
    comparisons = {}
    for years in [2, 5, 10]:
        ykey = f"{years}y"

        try:
            comparisons[f"cash_{ykey}"] = run_single_backtest(
                years, prices, constituent_df,
                spy_daily=spy_d, growth_map=gm,
                revenue_threshold=revenue_threshold,
                use_bonds=False, vol_adjusted=True,
            )
        except Exception as e:
            comparisons[f"cash_{ykey}"] = {"error": str(e)}

        try:
            comparisons[f"equal_wt_{ykey}"] = run_single_backtest(
                years, prices, constituent_df,
                spy_daily=spy_d, growth_map=gm,
                revenue_threshold=revenue_threshold,
                use_bonds=True, vol_adjusted=False,
            )
        except Exception as e:
            comparisons[f"equal_wt_{ykey}"] = {"error": str(e)}

        try:
            comparisons[f"spy_{ykey}"] = _spy_benchmark(prices, years, spy_daily=spy_d)
        except Exception as e:
            comparisons[f"spy_{ykey}"] = {"error": str(e)}

        try:
            comparisons[f"ew_sp500_{ykey}"] = _equal_weight_sp500(prices, constituent_df, years)
        except Exception as e:
            comparisons[f"ew_sp500_{ykey}"] = {"error": str(e)}

    holdings_history = (
        results.get("2y", {}).get("holdings_history")
        or results.get("5y", {}).get("holdings_history")
        or []
    )
    current_holdings = (
        results.get("2y", {}).get("current_holdings")
        or results.get("5y", {}).get("current_holdings")
        or []
    )

    return {
        "backtests": results,
        "comparisons": comparisons,
        "holdings_history": holdings_history,
        "current_holdings": current_holdings,
        "currently_in_ief": current_holdings == ["IEF"],
        "universe_size": len(all_tickers),
        "revenue_threshold": revenue_threshold,
        "computed_at": datetime.now().isoformat(),
        "note": (
            f"Gross returns. Vol-adjusted top-50 by 12-1 momentum. "
            f"200-day MA regime filter: moves to IEF when SPY < 200d MA. "
            f"Revenue screener: {revenue_threshold:.0%} YoY growth threshold. "
            f"Universe: point-in-time S&P 500 constituents."
        ),
    }
