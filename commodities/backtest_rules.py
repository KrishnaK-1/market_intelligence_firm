"""
Walk-forward evaluation of the rules-based desk (trend + carry + vol targeting
+ risk parity) on the same 2020-2025 calendar-year folds as the PPO desk, with
the same cost model and Sharpe convention, so the two are directly comparable.

There is no training step: parameters are fixed a priori (signals.py), so the
whole window is evaluated in one pass and sliced by year. If the price cache
extends into 2026, a bonus "2026 YTD" row is reported — genuinely untouched
data, seen by neither the RL desk nor the research that motivated this design.

Universe differences vs the PPO desk (deliberate, see fetcher.CURVE_TICKERS):
  - Energy trades USL (12-month WTI) instead of USO (front-month; contaminated
    by the April 2020 forced restructurings and chronic contango drag).
  - Agriculture/metals sleeves are pure commodity ETFs (no BDRY/TIP overlay).

Usage: python commodities/backtest_rules.py
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from commodities.data.fetcher import fetch_curve_pairs, fetch_tickers, fetch_macro
from commodities.rules.signals import (
    curve_spread, carry_multiplier, xs_momentum_multiplier, crisis_flag,
)
from commodities.rules.strategy import (
    build_sleeve_weights, sleeve_gross_returns, risk_parity_alloc,
    combine, perf_stats, TRANSACTION_COST,
)

DATA_START = "2015-01-01"   # warmup for 200d SMA + vol estimators
EVAL_START = "2020-01-01"
RESULTS_DIR = Path(__file__).resolve().parent / "walkforward_results"

# The rules desk's own universe — independent of the RL desk's COMMODITY_GROUPS
# so neither system's changes break the other. Every ticker is a still-trading,
# futures/physical-backed ETF with pre-2015 inception (full signal warmup and
# a complete 2020+ eval window). Most single-commodity softs/livestock ETNs
# (coffee, cotton, cocoa, cattle) were delisted in 2023; DBA's basket is the
# closest still-tradeable proxy for those exposures.
RULES_UNIVERSE = {
    "energy": {
        "WTI_12M": "USL",     # 12-month WTI ladder (replaces contaminated USO)
        "BRENT": "BNO",
        "NG": "UNG",
        "GASOLINE": "UGA",    # crack-spread exposure distinct from crude
    },
    "agriculture": {
        "CORN": "CORN",
        "WHEAT": "WEAT",
        "SOY": "SOYB",
        "SUGAR": "CANE",      # Teucrium sugar — the one surviving softs ETF
        "AG_BASKET": "DBA",   # grains + softs + livestock basket
    },
    "metals": {
        "GOLD": "GLD",
        "SILVER": "SLV",
        "COPPER": "CPER",
        "PLATINUM": "PPLT",
        "PALLADIUM": "PALL",
        "BASE_METALS": "DBB",  # aluminum/zinc/copper index
    },
}

# Original 10-asset universe, kept for ablation comparison
BASELINE_UNIVERSE = {
    sleeve: {k: v for k, v in assets.items()
             if v not in ("UGA", "CANE", "DBA", "PALL", "DBB")}
    for sleeve, assets in RULES_UNIVERSE.items()
}


def load_universe(universe=RULES_UNIVERSE, start=DATA_START):
    """Assemble sleeve price frames and energy carry multipliers. Sleeves keep
    rows where ANY asset is alive — assets born mid-backtest simply enter the
    universe at their inception (their weight is zero until then)."""
    curve = fetch_curve_pairs(start=start)  # WTI_FRONT, WTI_12M, NG_12M, WTI_OPT

    sleeves = {}
    for name, assets in universe.items():
        tickers = sorted(set(assets.values()))
        cache_name = f"rules_{name}_{'_'.join(tickers)}"  # cache keyed by ticker set
        closes = fetch_tickers(tickers, cache_name, start=start)
        closes = closes.rename(columns={v: k for k, v in assets.items()})
        sleeves[name] = closes[list(assets.keys())].dropna(how="all")

    wti_spread = curve_spread(curve["WTI_FRONT"], curve["WTI_12M"])
    ng_spread = curve_spread(sleeves["energy"]["NG"], curve["NG_12M"])
    carry_mults = {
        "WTI_12M": carry_multiplier(wti_spread),
        # WTI and Brent curves are tightly linked; no deferred Brent ETF exists,
        # so the WTI curve signal proxies for Brent as well.
        "BRENT": carry_multiplier(wti_spread),
        "NG": carry_multiplier(ng_spread),
    }
    return sleeves, carry_mults


def yearly_stats(returns: pd.Series, eval_start: str = EVAL_START) -> dict:
    """Per-calendar-year perf stats from EVAL_START onward. A partial final
    year is labeled '<year> YTD'."""
    r = returns.loc[eval_start:].dropna()
    out = {}
    for year, chunk in r.groupby(r.index.year):
        label = str(year)
        if chunk.index[-1] < pd.Timestamp(f"{year}-12-15"):
            label = f"{year} YTD"
        out[label] = perf_stats(chunk)
    return out


def sleeve_net_returns(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """Sleeve returns net of transaction costs (for standalone comparison
    against the PPO sleeves, which also pay costs internally)."""
    gross = sleeve_gross_returns(weights, prices.pct_change())
    turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    return gross - turnover * TRANSACTION_COST


def load_ppo_results(group: str) -> dict:
    path = RESULTS_DIR / f"{group}.json"
    if not path.exists():
        return {}
    return {r["year"]: r for r in json.loads(path.read_text())}


def print_sleeve_comparison(name: str, rules_years: dict, ppo: dict):
    print(f"\n----- {name.upper()} — rules vs PPO (per OOS year) -----")
    print(f"  {'Year':<10}{'Rules ret':>11}{'Rules Sharpe':>14}{'PPO ret':>11}{'PPO Sharpe':>12}")
    for year, s in rules_years.items():
        p = ppo.get(year.split()[0], {})
        ppo_ret = f"{p['return']:+.1%}" if p else "—"
        ppo_sh = f"{p['sharpe']:+.2f}" if p else "—"
        print(f"  {year:<10}{s['return']:>+11.1%}{s['sharpe']:>14.2f}{ppo_ret:>11}{ppo_sh:>12}")
    full_years = [y for y in rules_years if "YTD" not in y]
    mean_sh = np.mean([rules_years[y]["sharpe"] for y in full_years])
    prof = sum(1 for y in full_years if rules_years[y]["return"] > 0)
    print(f"  Mean Sharpe (full years): {mean_sh:+.2f}   profitable: {prof}/{len(full_years)}")


def run_backtest(universe=RULES_UNIVERSE, use_xs_momentum=True,
                 use_crisis_override=True, verbose=True,
                 data_start=DATA_START, eval_start=EVAL_START):
    sleeves, carry_mults = load_universe(universe, start=data_start)

    # Cross-sectional momentum is ranked across the WHOLE desk (all sleeves),
    # not within-sleeve: "is gold strong relative to sugar" is the signal.
    xs_mults = None
    if use_xs_momentum:
        xs_mults = xs_momentum_multiplier(pd.concat(sleeves.values(), axis=1))

    vix = None
    if use_crisis_override:
        macro = fetch_macro(start=data_start)
        vix = (macro["Close"] if "Close" in macro else macro.xs("Close", axis=1, level=0))["^VIX"]

    # Per-sleeve weights and standalone net returns
    weights, net_rets = {}, {}
    for name, prices in sleeves.items():
        mults = carry_mults if name == "energy" else None
        tilt = xs_mults[prices.columns] if xs_mults is not None else None
        weights[name] = build_sleeve_weights(prices, carry_mults=mults, tilt_mults=tilt)
        if use_crisis_override:
            # Same per-sleeve liquidation the RL envs enforce: each sleeve
            # goes to cash when ITS OWN basket vol (or VIX) is in crisis.
            flag = crisis_flag(prices, vix)
            weights[name] = weights[name].mask(flag, 0.0)
        net_rets[name] = sleeve_net_returns(weights[name], prices)
        if verbose:
            print_sleeve_comparison(name, yearly_stats(net_rets[name], eval_start), load_ppo_results(name))

    # Combined portfolio — costs charged once on final asset weights
    all_prices = pd.concat(sleeves.values(), axis=1).dropna(how="all")
    sleeve_gross = pd.DataFrame({
        n: sleeve_gross_returns(weights[n].reindex(all_prices.index).fillna(0.0), all_prices.pct_change())
        for n in sleeves
    })
    # T-bill returns for the cash remainder (BIL ETF) — real desks don't hold
    # idle cash at zero yield, especially through the 2023-2025 rate regime.
    bil = fetch_tickers(["BIL"], "commodities_bil", start=data_start)["BIL"].pct_change()

    results = {}
    combos = {}
    for label, alloc, cash in [
        ("equal_weight", None, None),
        ("risk_parity", risk_parity_alloc(sleeve_gross), None),
        ("equal_weight + T-bill cash", None, bil),
    ]:
        combo = combine(weights, all_prices, alloc=alloc, cash_returns=cash)
        combos[label] = combo
        eval_rets = combo["returns"].loc[eval_start:]
        stats = perf_stats(eval_rets)
        years = yearly_stats(combo["returns"], eval_start)
        results[label] = {
            "overall": stats,
            "years": years,
            "avg_daily_turnover": float(combo["turnover"].loc[eval_start:].mean()),
        }
        if not verbose:
            continue
        print(f"\n===== COMBINED PORTFOLIO ({label}) — {eval_start[:4]}+ =====")
        for year, s in years.items():
            print(f"  {year:<10} return {s['return']:+7.1%}   sharpe {s['sharpe']:+.2f}   maxDD {s['max_dd']:.1%}")
        full = [y for y in years if "YTD" not in y]
        print(f"  Overall: return {stats['return']:+.1%}  sharpe {stats['sharpe']:+.2f}  "
              f"vol {stats['vol']:.1%}  maxDD {stats['max_dd']:.1%}")
        print(f"  Mean yearly Sharpe: {np.mean([years[y]['sharpe'] for y in full]):+.2f}   "
              f"profitable: {sum(1 for y in full if years[y]['return'] > 0)}/{len(full)}   "
              f"avg daily turnover: {results[label]['avg_daily_turnover']:.2%}")

    return {
        "sleeves": {n: yearly_stats(net_rets[n], eval_start) for n in sleeves},
        "combined": results,
        # Raw series for downstream consumers (dashboard export); not JSON'd
        "_series": {
            "combos": combos,
            "sleeve_net_returns": net_rets,
            "sleeve_weights": weights,
            "all_prices": all_prices,
        },
    }


def main():
    out = run_backtest(RULES_UNIVERSE, use_xs_momentum=True, verbose=True)
    serializable = {k: v for k, v in out.items() if not k.startswith("_")}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "rules.json").write_text(json.dumps(serializable, indent=2))
    print(f"\nSaved -> {RESULTS_DIR / 'rules.json'}")


if __name__ == "__main__":
    main()
