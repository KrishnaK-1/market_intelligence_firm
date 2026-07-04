"""
Generic desk runner + firm-level manager.

A desk = one or more sleeves of ETFs run through the shared rules
(desks/signals.py, desks/strategy.py). The firm combines desk return streams
with either equal weights or inverse-vol (risk parity across asset classes),
charges costs once on the final asset-level weights, and parks idle cash in
T-bills. No shorting, no leverage anywhere.

Usage: python desks/engine.py           (runs the firm backtest, saves JSON)
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from commodities.data.fetcher import fetch_tickers, fetch_curve_pairs
from desks.signals import (
    curve_spread, carry_multiplier, xs_momentum_multiplier, crisis_flag,
    TRADING_DAYS,
)
from desks.strategy import (
    build_sleeve_weights, apply_trade_band, combine, perf_stats,
    TRANSACTION_COST,
)
from desks.universes import DESKS, DATA_START, EVAL_START, YIELD_CURVE_CARRY_ASSETS

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIRM_VOL_FLOOR = 0.04   # alloc floor so a becalmed desk can't swallow the firm


def _fetch_named(assets: dict, cache_prefix: str, start: str) -> pd.DataFrame:
    tickers = sorted(set(assets.values()))
    cache_name = f"{cache_prefix}_{'_'.join(t.replace('^', '') for t in tickers)}"
    closes = fetch_tickers(tickers, cache_name, start=start)
    closes = closes.rename(columns={v: k for k, v in assets.items()})
    return closes[list(assets.keys())].dropna(how="all")


def _normalize_yield(series: pd.Series) -> pd.Series:
    """CBOE yield indices are sometimes quoted x10 (44.0 = 4.4%). Post-2015 US
    yields never exceeded 15%, so a median above that means the x10 form."""
    recent = series.loc["2015":].dropna()
    return series / 10.0 if len(recent) and recent.median() > 15 else series


def build_carry(kind: str, start: str) -> dict:
    """Desk-specific carry multipliers, keyed by asset logical name."""
    if kind == "energy_curve":
        curve = fetch_curve_pairs(start=start)
        ung = fetch_tickers(["UNG"], "desks_ung", start=start)["UNG"]
        wti = curve_spread(curve["WTI_FRONT"], curve["WTI_12M"])
        ng = curve_spread(ung, curve["NG_12M"])
        return {
            "WTI_12M": carry_multiplier(wti),
            "BRENT": carry_multiplier(wti),   # WTI curve proxies Brent (no deferred Brent ETF)
            "NG": carry_multiplier(ng),
        }
    if kind == "yield_curve":
        y = fetch_tickers(["^TNX", "^IRX"], "desks_yields", start=start)
        slope = _normalize_yield(y["^TNX"]) - _normalize_yield(y["^IRX"])
        mult = carry_multiplier(slope.ffill())
        return {asset: mult for asset in YIELD_CURVE_CARRY_ASSETS}
    raise ValueError(f"Unknown carry kind: {kind}")


def run_desk(name: str, spec: dict, vix: pd.Series,
             data_start: str = DATA_START) -> dict:
    """Run one desk. Returns its price frame, final asset weights (banded,
    cost-free), and standalone net returns for reporting/allocation."""
    sleeves = {
        sleeve: _fetch_named(assets, f"desk_{name}_{sleeve}", data_start)
        for sleeve, assets in spec["sleeves"].items()
    }
    carry = build_carry(spec["carry"], data_start) if "carry" in spec else {}

    all_prices = pd.concat(sleeves.values(), axis=1).dropna(how="all")
    xs = xs_momentum_multiplier(all_prices)

    weights = {}
    for sleeve, prices in sleeves.items():
        w = build_sleeve_weights(prices, carry_mults=carry or None,
                                 tilt_mults=xs[prices.columns])
        flag = crisis_flag(prices, vix)
        weights[sleeve] = w.mask(flag, 0.0)

    combo = combine(weights, all_prices)   # equal-weight sleeves, costs once
    return {
        "prices": all_prices,
        "weights": combo["weights"],
        "net_returns": combo["returns"],
        "turnover": combo["turnover"],
    }


def yearly_stats(returns: pd.Series, eval_start: str = EVAL_START) -> dict:
    r = returns.loc[eval_start:].dropna()
    out = {}
    for year, chunk in r.groupby(r.index.year):
        label = str(year) if len(chunk) >= 200 else f"{year} YTD"
        out[label] = perf_stats(chunk)
    return out


def run_firm(data_start: str = DATA_START, eval_start: str = EVAL_START,
             include: list = None, verbose: bool = True) -> dict:
    vix = fetch_tickers(["^VIX"], "desks_vix", start=data_start)["^VIX"]
    bil = fetch_tickers(["BIL"], "commodities_bil", start=data_start)["BIL"].pct_change()

    specs = {n: s for n, s in DESKS.items() if include is None or n in include}
    desks = {name: run_desk(name, spec, vix, data_start)
             for name, spec in specs.items()}

    all_prices = pd.concat([d["prices"] for d in desks.values()], axis=1)
    all_prices = all_prices.dropna(how="all")
    all_rets = all_prices.pct_change()
    desk_rets = pd.DataFrame({n: d["net_returns"] for n, d in desks.items()})

    # Firm allocation across desks: equal weight, and inverse trailing vol
    # (risk parity across asset classes — desk vols range ~4% to ~15%)
    n = len(desks)
    vol = desk_rets.rolling(63, min_periods=20).std() * np.sqrt(TRADING_DAYS)
    inv = 1.0 / vol.clip(lower=FIRM_VOL_FLOOR)
    rp_alloc = inv.div(inv.sum(axis=1), axis=0).fillna(1.0 / n)
    eq_alloc = pd.DataFrame(1.0 / n, index=all_prices.index, columns=list(desks))

    results, series = {}, {}
    for label, alloc in [("equal_weight", eq_alloc), ("risk_parity", rp_alloc)]:
        firm_w = pd.DataFrame(0.0, index=all_prices.index, columns=all_prices.columns)
        for name, d in desks.items():
            w = d["weights"].reindex(all_prices.index).fillna(0.0)
            firm_w[w.columns] = firm_w[w.columns] + w.mul(
                alloc[name].reindex(all_prices.index).fillna(1.0 / n), axis=0)
        firm_w = apply_trade_band(firm_w)

        held = firm_w.shift(1).fillna(0.0)
        gross = (held * all_rets).sum(axis=1)
        cash = (1.0 - held.sum(axis=1)).clip(lower=0.0)
        turnover = (firm_w - held).abs().sum(axis=1)
        net = gross + cash * bil.reindex(gross.index).fillna(0.0) - turnover * TRANSACTION_COST

        results[label] = {
            "overall": perf_stats(net.loc[eval_start:]),
            "years": yearly_stats(net, eval_start),
            "avg_daily_turnover": float(turnover.loc[eval_start:].mean()),
        }
        series[label] = net

    # Benchmarks: SPY and daily-rebalanced 60/40 (SPY/AGG)
    bench = fetch_tickers(["SPY", "AGG"], "desks_benchmarks", start=data_start)
    spy = bench["SPY"].pct_change()
    b6040 = 0.6 * spy + 0.4 * bench["AGG"].pct_change()
    benchmarks = {
        "SPY": {"overall": perf_stats(spy.loc[eval_start:]), "years": yearly_stats(spy, eval_start)},
        "60/40": {"overall": perf_stats(b6040.loc[eval_start:]), "years": yearly_stats(b6040, eval_start)},
    }

    corr = desk_rets.loc[eval_start:].corr().round(2)
    spy_corr = {n: round(float(desk_rets[n].loc[eval_start:].corr(spy.loc[eval_start:])), 2)
                for n in desks}

    out = {
        "desks": {n: {"overall": perf_stats(d["net_returns"].loc[eval_start:]),
                      "years": yearly_stats(d["net_returns"], eval_start)}
                  for n, d in desks.items()},
        "firm": results,
        "benchmarks": benchmarks,
        "desk_correlations": corr.to_dict(),
        "spy_correlation": spy_corr,
        "_series": {"desks": desks, "firm": series, "desk_rets": desk_rets},
    }

    if verbose:
        print(f"\n===== DESKS ({eval_start[:4]}+) =====")
        for name, d in out["desks"].items():
            o = d["overall"]
            print(f"  {name:<14} sharpe {o['sharpe']:+.2f}  return {o['return']:+7.1%}  "
                  f"vol {o['vol']:.1%}  maxDD {o['max_dd']:.1%}  (corr to SPY: {spy_corr[name]:+.2f})")
        print(f"\n  Desk cross-correlations:\n{corr.to_string()}")
        for label, res in results.items():
            o = res["overall"]
            years = res["years"]
            full = [y for y in years if "YTD" not in y]
            print(f"\n===== FIRM ({label}) =====")
            print(f"  sharpe {o['sharpe']:+.2f}  return {o['return']:+.1%}  vol {o['vol']:.1%}  "
                  f"maxDD {o['max_dd']:.1%}  turnover {res['avg_daily_turnover']:.2%}/day")
            print(f"  profitable years: {sum(1 for y in full if years[y]['return'] > 0)}/{len(full)}")
        for name, b in benchmarks.items():
            o = b["overall"]
            print(f"  benchmark {name:<6} sharpe {o['sharpe']:+.2f}  return {o['return']:+9.1%}  maxDD {o['max_dd']:.1%}")

    return out


def main():
    out = run_firm()
    serializable = {k: v for k, v in out.items() if not k.startswith("_")}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "firm.json").write_text(json.dumps(serializable, indent=2))
    print(f"\nSaved -> {RESULTS_DIR / 'firm.json'}")


if __name__ == "__main__":
    main()
