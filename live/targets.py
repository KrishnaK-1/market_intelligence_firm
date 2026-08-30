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
import time
import logging
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf

from commodities.data.fetcher import CACHE_DIR, CURVE_TICKERS
from desks.universes import DESKS
from desks.engine import run_firm, VALUE_DATA_START
from live.config import INCLUDE_DESKS, FIRM_VARIANT

log = logging.getLogger("live")

# Data-quality guards. A PARTIAL yfinance download (some tickers missing
# their most recent bars) is far more dangerous than an empty one: the
# affected assets get a NaN price, their trend gate evaluates to 0, and the
# engine silently reallocates the whole portfolio away from them. That
# happened on 2026-08-20 — HYG's target collapsed 20.8% -> 4.0% and MBB/TIP
# absorbed it — causing a ~$40k round trip that reverted the next day.
MAX_STALE_BARS = 3      # a ticker's last close may lag the frame's last row by this much
MIN_HISTORY_ROWS = 500  # a truncated download is a failed download
DOWNLOAD_RETRIES = 3
FFILL_LIMIT = 3         # patch isolated interior gaps so one hole can't zero an asset


def _download(tickers: list, end: str, name: str) -> pd.DataFrame:
    """Download with retries. Raises if every attempt comes back unusable."""
    last_err = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            df = yf.download(tickers, start=VALUE_DATA_START, end=end,
                             auto_adjust=True, progress=False)
            if df is not None and len(df) >= MIN_HISTORY_ROWS:
                return df
            last_err = f"got {0 if df is None else len(df)} rows (need >= {MIN_HISTORY_ROWS})"
        except Exception as e:
            last_err = str(e)
        if attempt < DOWNLOAD_RETRIES:
            time.sleep(5 * attempt)
    raise RuntimeError(f"Download failed for {name} {tickers}: {last_err}")


def _validate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Reject a download where any ticker's most recent close is stale or
    missing, then forward-fill small interior gaps. Refusing to trade is
    always safer than trading a silently-reweighted portfolio."""
    close = df["Close"] if "Close" in df else df.xs("Close", axis=1, level=0)
    close = close.dropna(how="all")           # all-NaN rows are holidays, not data loss
    if len(close) < MIN_HISTORY_ROWS:
        raise RuntimeError(f"{name}: only {len(close)} usable rows after cleaning")

    problems = []
    for ticker in close.columns:
        series = close[ticker].dropna()
        if series.empty:
            problems.append(f"{ticker}: no data at all")
            continue
        stale_bars = len(close.loc[series.index[-1]:]) - 1
        if stale_bars > MAX_STALE_BARS:
            problems.append(f"{ticker}: last close {series.index[-1].date()} "
                            f"({stale_bars} bars stale)")
    if problems:
        raise RuntimeError(
            f"Stale/incomplete data in {name} — refusing to trade: " + "; ".join(problems))

    return df.ffill(limit=FFILL_LIMIT)


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
        df = _validate(_download(tickers, end, name), name)
        # Only overwrite the cache once the data has passed validation, so a
        # bad download can never replace a good cache.
        df.to_pickle(CACHE_DIR / f"{name}.pkl")
        if verbose:
            print(f"  refreshed {name}: {len(df)} rows, last bar {df.index[-1].date()}")
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
