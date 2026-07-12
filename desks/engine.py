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
             data_start: str = DATA_START,
             allow_short: bool = False, max_gross: float = 1.0) -> dict:
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
                                 tilt_mults=xs[prices.columns],
                                 allow_short=allow_short, max_gross=max_gross)
        flag = crisis_flag(prices, vix)
        weights[sleeve] = w.mask(flag, 0.0)

    combo = combine(weights, all_prices)   # equal-weight sleeves, costs once
    return {
        "prices": all_prices,
        "weights": combo["weights"],
        "net_returns": combo["returns"],
        "turnover": combo["turnover"],
    }


# Value data reaches back to 2000 so the 5-year lookback is live by the 2008
# eval start for the pre-2006 ETF cohort; younger ETFs enter the value
# universe five years after inception.
VALUE_DATA_START = "2000-01-01"
VALUE_LOOKBACK = 5 * 252


def run_value_desk(vix: pd.Series, data_start: str = DATA_START,
                   allow_short: bool = False, max_gross: float = 1.0) -> dict:
    """
    Value desk (experimental 4th desk): the standard non-equity value signal
    is long-term reversal — cheapness = the NEGATIVE of the past 5-year
    return (Asness, Moskowitz & Pedersen 2013). Assets are ranked WITHIN
    their desk group (commodities vs commodities, sectors vs sectors), the
    cheap half is overweighted, sized inverse-vol, vol-targeted, banded.
    Deliberately NO trend filter — buying what trend sold is exactly where
    the negative correlation to the rest of the firm comes from.
    """
    groups = {}
    for desk_name, spec in DESKS.items():
        if desk_name == "fx":
            continue   # FX desk is shelved
        # "V_" prefix keeps value-desk columns distinct from the trend desks'
        # when the firm concatenates weights (same underlying ETFs; positions
        # are tracked per-desk rather than netted — slightly conservative on
        # costs, much simpler to attribute)
        merged = {}
        for sleeve, assets in spec["sleeves"].items():
            merged.update({f"V_{k}": v for k, v in assets.items()})
        groups[desk_name] = _fetch_named(merged, f"desk_{desk_name}_value",
                                         VALUE_DATA_START)

    all_prices = pd.concat(groups.values(), axis=1).dropna(how="all")
    rets = all_prices.pct_change()
    from desks.signals import ewma_vol, ewma_cov, portfolio_vol
    from desks.strategy import SLEEVE_TARGET_VOL
    ann_vol = ewma_vol(rets)

    score = pd.DataFrame(index=all_prices.index, columns=all_prices.columns,
                         dtype=float)
    for name, px in groups.items():
        cheap = -(px / px.shift(VALUE_LOOKBACK) - 1.0)
        pct = cheap.rank(axis=1, pct=True)          # 1.0 = cheapest in group
        if allow_short:
            score[px.columns] = (2.0 * pct - 1.0)   # rich half shorted
        else:
            score[px.columns] = ((pct - 0.5) * 2.0).clip(lower=0.0)

    raw = (score / ann_vol).fillna(0.0)
    total = raw.abs().sum(axis=1)
    w_norm = raw.div(total.where(total > 0), axis=0).fillna(0.0)
    vol_est = portfolio_vol(w_norm, ewma_cov(rets))
    scale = (SLEEVE_TARGET_VOL / vol_est.replace(0.0, np.nan)).clip(upper=max_gross).fillna(0.0)
    w = apply_trade_band(w_norm.mul(scale, axis=0))
    w = w.mask(crisis_flag(all_prices, vix), 0.0)

    w = w.loc[data_start:]
    prices = all_prices.loc[data_start:]
    combo = combine({"value": w}, prices)
    return {
        "prices": prices,
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


FINANCING_SPREAD = 0.005   # borrow at T-bill + 50 bps for gross exposure > 1


def run_firm(data_start: str = DATA_START, eval_start: str = EVAL_START,
             include: list = None, verbose: bool = True,
             allow_short: bool = False, max_gross: float = 1.0) -> dict:
    """`include` may list desk names from DESKS plus the special name
    "value" (the experimental 5-year-reversal desk). allow_short/max_gross
    are the compliance switches — both off by default; when max_gross > 1,
    leverage financing is charged at T-bill + FINANCING_SPREAD. Shorting is
    modeled frictionless (no borrow fees) — treat short results as an
    upper-bound estimate."""
    vix = fetch_tickers(["^VIX"], "desks_vix", start=data_start)["^VIX"]
    bil = fetch_tickers(["BIL"], "commodities_bil", start=data_start)["BIL"].pct_change()

    specs = {n: s for n, s in DESKS.items() if include is None or n in include}
    desks = {name: run_desk(name, spec, vix, data_start,
                            allow_short=allow_short, max_gross=max_gross)
             for name, spec in specs.items()}
    if include is not None and "value" in include:
        desks["value"] = run_value_desk(vix, data_start,
                                        allow_short=allow_short, max_gross=max_gross)

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

    results, series, firm_weights = {}, {}, {}
    for label, alloc in [("equal_weight", eq_alloc), ("risk_parity", rp_alloc)]:
        firm_w = pd.DataFrame(0.0, index=all_prices.index, columns=all_prices.columns)
        for name, d in desks.items():
            w = d["weights"].reindex(all_prices.index).fillna(0.0)
            firm_w[w.columns] = firm_w[w.columns] + w.mul(
                alloc[name].reindex(all_prices.index).fillna(1.0 / n), axis=0)
        firm_w = apply_trade_band(firm_w)

        held = firm_w.shift(1).fillna(0.0)
        gross = (held * all_rets).sum(axis=1)
        gross_exposure = held.abs().sum(axis=1)
        cash = (1.0 - gross_exposure).clip(lower=0.0)
        borrowed = (gross_exposure - 1.0).clip(lower=0.0)
        turnover = (firm_w - held).abs().sum(axis=1)
        bil_d = bil.reindex(gross.index).fillna(0.0)
        net = (gross + cash * bil_d
               - borrowed * (bil_d + FINANCING_SPREAD / TRADING_DAYS)
               - turnover * TRANSACTION_COST)

        results[label] = {
            "overall": perf_stats(net.loc[eval_start:]),
            "years": yearly_stats(net, eval_start),
            "avg_daily_turnover": float(turnover.loc[eval_start:].mean()),
        }
        series[label] = net
        firm_weights[label] = firm_w   # asset-level targets (live trading reads the last row)

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
        "_series": {"desks": desks, "firm": series, "desk_rets": desk_rets,
                    "firm_weights": firm_weights},
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
