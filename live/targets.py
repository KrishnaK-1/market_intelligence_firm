"""
Computes today's target portfolio weights by running the exact same engine
as the backtest (desks/engine.py) on freshly downloaded data.

Freshness + correctness: every price cache the engine reads is re-downloaded
here with an EXCLUSIVE end date of today, so (a) the data always includes
yesterday's completed close, and (b) a partial intraday bar for today can
never leak into the signals no matter what time the script runs. The engine
then reads these caches untouched — live targets and backtest come from one
code path.
"""
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf

from commodities.data.fetcher import CACHE_DIR, CURVE_TICKERS
from desks.universes import DESKS
from desks.engine import run_firm, VALUE_DATA_START
from live.config import INCLUDE_DESKS, FIRM_VARIANT


def _cache_specs() -> dict:
    """Every cache file the engine touches for the live desk set, with its
    ticker list — names must match desks/engine.py and fetcher.py exactly."""
    specs = {}
    for desk, spec in DESKS.items():
        if desk not in INCLUDE_DESKS:
            continue
        for sleeve, assets in spec["sleeves"].items():
            tickers = sorted(set(assets.values()))
            name = f"desk_{desk}_{sleeve}_{'_'.join(t.replace('^', '') for t in tickers)}"
            specs[name] = tickers
    specs["commodities_curve"] = list(CURVE_TICKERS.values())
    specs["desks_ung"] = ["UNG"]
    specs["desks_yields"] = ["^TNX", "^IRX"]
    specs["desks_vix"] = ["^VIX"]
    specs["commodities_bil"] = ["BIL"]
    specs["desks_benchmarks"] = ["SPY", "AGG"]
    return specs


def refresh_caches(verbose: bool = True) -> str:
    """Re-download every engine cache through yesterday's close. Returns the
    exclusive end date used."""
    end = date.today().isoformat()   # yf `end` is exclusive -> data through yesterday
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name, tickers in _cache_specs().items():
        df = yf.download(tickers, start=VALUE_DATA_START, end=end,
                         auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            raise RuntimeError(f"Download returned no data for {name} ({tickers}) — "
                               "refusing to trade on missing data.")
        df.to_pickle(CACHE_DIR / f"{name}.pkl")
        if verbose:
            last = df.index[-1].date()
            print(f"  refreshed {name}: {len(df)} rows, last bar {last}")
    return end


def compute_targets() -> dict:
    """Run the engine and return today's target weights BY TICKER, plus
    diagnostics. Weights are fractions of account equity; the remainder is
    cash (parked in T-bills by the strategy design; left as cash at Alpaca)."""
    out = run_firm(include=INCLUDE_DESKS, verbose=False)
    firm_w = out["_series"]["firm_weights"][FIRM_VARIANT]
    last_row = firm_w.iloc[-1]
    asof = firm_w.index[-1].date().isoformat()

    # logical asset name -> ticker
    name_to_ticker = {}
    for desk, spec in DESKS.items():
        if desk not in INCLUDE_DESKS:
            continue
        for sleeve, assets in spec["sleeves"].items():
            name_to_ticker.update(assets)

    targets = {}
    for logical, w in last_row.items():
        t = name_to_ticker[logical]
        targets[t] = targets.get(t, 0.0) + float(w)
    targets = {t: w for t, w in targets.items() if abs(w) > 1e-6}

    if min(targets.values(), default=0.0) < 0:
        raise RuntimeError("Negative target weight with shorting disabled — aborting.")
    gross = sum(targets.values())
    if gross > 1.0 + 1e-6:
        raise RuntimeError(f"Target gross exposure {gross:.3f} exceeds 1.0 — aborting.")

    desk_gross = {n: float(d["weights"].iloc[-1].abs().sum())
                  for n, d in out["_series"]["desks"].items()}
    return {
        "asof": asof,                 # date of the close the signals used
        "weights": targets,           # {ticker: fraction of equity}
        "cash_weight": 1.0 - gross,
        "desk_gross_exposure": desk_gross,
        "by_logical": {k: float(v) for k, v in last_row.items() if abs(v) > 1e-6},
    }
