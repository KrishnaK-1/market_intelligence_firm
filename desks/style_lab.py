"""
STYLE LAB — experimental, NOT part of the production firm.

Prototypes two non-trend styles and measures whether they diversify the
3-desk trend firm (commodities + equities + rates):

  1. Mean reversion ("buy the dip in an uptrend"): on the equities universe,
     overweight assets whose 5-day move is a sharp DOWN outlier while the
     asset remains above its 200d SMA. Opposite reflex to trend: it buys
     what trend-followers are nervously watching fall.

  2. Relative value (ratio reversion): five economically-linked pairs
     (gold/silver, WTI/Brent, corn/wheat, staples/discretionary, 7-10y/20y
     Treasuries). When the log price ratio is >1 std from its 1-year norm,
     overweight the CHEAP leg. Long/flat handicap: we cannot short the rich
     leg, so half the classic trade (and its market-neutrality) is missing.

Both run under the firm's standard machinery: inverse-vol sizing, 12% vol
target, 5% no-trade band, 10 bps costs, optional crisis rule, long/flat.

Usage: python desks/style_lab.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from commodities.data.fetcher import fetch_tickers
from desks.engine import run_firm, _fetch_named
from desks.universes import DESKS, DATA_START, EVAL_START
from desks.signals import ewma_vol, ewma_cov, portfolio_vol, crisis_flag, TRADING_DAYS
from desks.strategy import apply_trade_band, perf_stats, TRANSACTION_COST, SLEEVE_TARGET_VOL

RV_PAIRS = [
    ("GLD", "SLV"), ("USL", "BNO"), ("CORN", "WEAT"),
    ("XLP", "XLY"), ("IEF", "TLT"),
]
Z_ENTRY = 1.0        # ratio must be >1 std from norm before a position opens
Z_CAP = 3.0
MR_DIP_ENTRY = 0.5   # 5d move must be >0.5 std below zero
MR_DIP_CAP = 3.0


def finalize_sleeve(raw: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """Shared tail of sleeve construction: normalize, vol-target, band."""
    total = raw.sum(axis=1)
    w_norm = raw.div(total.where(total > 0), axis=0).fillna(0.0)
    vol_est = portfolio_vol(w_norm, ewma_cov(rets))
    scale = (SLEEVE_TARGET_VOL / vol_est.replace(0.0, np.nan)).clip(upper=1.0).fillna(0.0)
    return apply_trade_band(w_norm.mul(scale, axis=0))


def net_returns(weights: pd.DataFrame, prices: pd.DataFrame, bil: pd.Series) -> pd.Series:
    rets = prices.pct_change()
    held = weights.shift(1).fillna(0.0)
    gross = (held * rets).sum(axis=1)
    cash = (1.0 - held.sum(axis=1)).clip(lower=0.0)
    turnover = (weights - held).abs().sum(axis=1)
    return gross + cash * bil.reindex(gross.index).fillna(0.0) - turnover * TRANSACTION_COST


def mean_reversion_sleeve(prices: pd.DataFrame, vix: pd.Series, use_crisis: bool) -> pd.DataFrame:
    rets = prices.pct_change()
    ann_vol = ewma_vol(rets)
    daily_vol = ann_vol / np.sqrt(TRADING_DAYS)
    z5 = prices.pct_change(5).div(daily_vol * np.sqrt(5))
    uptrend = (prices > prices.rolling(200).mean()).astype(float)
    dip = (-z5 - MR_DIP_ENTRY).clip(lower=0.0, upper=MR_DIP_CAP)
    raw = (dip * uptrend / ann_vol).fillna(0.0)
    w = finalize_sleeve(raw, rets)
    if use_crisis:
        w = w.mask(crisis_flag(prices, vix), 0.0)
    return w


def relative_value_sleeve(prices: pd.DataFrame, vix: pd.Series, use_crisis: bool) -> pd.DataFrame:
    rets = prices.pct_change()
    ann_vol = ewma_vol(rets)
    raw = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for a, b in RV_PAIRS:
        ratio = np.log(prices[a] / prices[b])
        z = (ratio - ratio.rolling(252).mean()) / (ratio.rolling(252).std() + 1e-8)
        # z high = a rich vs b -> b is cheap; z low = a cheap
        raw[b] = raw[b] + (z - Z_ENTRY).clip(lower=0.0, upper=Z_CAP - Z_ENTRY)
        raw[a] = raw[a] + (-z - Z_ENTRY).clip(lower=0.0, upper=Z_CAP - Z_ENTRY)
    raw = (raw / ann_vol).fillna(0.0)
    w = finalize_sleeve(raw, rets)
    if use_crisis:
        w = w.mask(crisis_flag(prices, vix), 0.0)
    return w


def report(name: str, r: pd.Series, firm: pd.Series):
    corr = float(r.loc[EVAL_START:].corr(firm.loc[EVAL_START:]))
    blend = 0.75 * firm + 0.25 * r   # style added as a 4th equal-weight desk
    line = f"  {name:<26}"
    for start, tag in [(EVAL_START, "full"), ("2020-01-01", "2020+")]:
        s = perf_stats(r.loc[start:])
        line += f" | {tag} sharpe {s['sharpe']:+.2f} ret {s['return']:+7.1%} DD {s['max_dd']:.0%}"
    b = perf_stats(blend.loc[EVAL_START:])
    line += f" | corr {corr:+.2f} | blended-firm sharpe {b['sharpe']:+.2f}"
    print(line)
    return {"corr": corr, "blend_sharpe": b["sharpe"]}


def main():
    print("Running baseline 3-desk trend firm...")
    base = run_firm(include=["commodities", "equities", "rates"], verbose=False)
    firm = base["_series"]["firm"]["equal_weight"]
    fs = perf_stats(firm.loc[EVAL_START:])
    print(f"  baseline firm: full sharpe {fs['sharpe']:+.2f}  "
          f"2020+ sharpe {perf_stats(firm.loc['2020-01-01':])['sharpe']:+.2f}\n")

    vix = fetch_tickers(["^VIX"], "desks_vix", start=DATA_START)["^VIX"]
    bil = fetch_tickers(["BIL"], "commodities_bil", start=DATA_START)["BIL"].pct_change()

    eq_prices = _fetch_named(DESKS["equities"]["sleeves"]["equities"],
                             "desk_equities_equities", DATA_START)
    rv_tickers = sorted({t for pair in RV_PAIRS for t in pair})
    rv_prices = fetch_tickers(rv_tickers, f"style_rv_{'_'.join(rv_tickers)}",
                              start=DATA_START).dropna(how="all")

    print("Style sleeves (standalone and blended 25% into the firm):")
    for use_crisis in [False, True]:
        tag = "with crisis rule" if use_crisis else "no crisis rule"
        w_mr = mean_reversion_sleeve(eq_prices, vix, use_crisis)
        report(f"mean-reversion ({tag})", net_returns(w_mr, eq_prices, bil), firm)
        w_rv = relative_value_sleeve(rv_prices, vix, use_crisis)
        report(f"relative-value ({tag})", net_returns(w_rv, rv_prices, bil), firm)


if __name__ == "__main__":
    main()
