"""
Runs the full-history (2008+) walk-forward of the rules-based commodity desk
and exports everything the web dashboard needs as one JSON payload:
equity curves vs DBC/SPY benchmarks, drawdown, yearly stats, per-sleeve
curves, and monthly sector allocation.

The 2008 start is the honest maximum: GLD/SLV/USL/UNG/UGA/DBA/DBB trade from
2006-2008; later ETFs (BNO 2010, CORN 2010, CPER/CANE/WEAT/SOYB late 2011)
enter the universe at their inception. A true "20 years" is impossible —
most commodity ETFs simply didn't exist.

Usage: python commodities/export_dashboard.py
"""
import sys
import json
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from commodities.backtest_rules import run_backtest, RULES_UNIVERSE, RESULTS_DIR
from commodities.data.fetcher import fetch_tickers
from commodities.rules.strategy import perf_stats

DATA_START = "2006-01-01"
EVAL_START = "2008-01-01"
RECENT_START = "2020-01-01"
HEADLINE = "equal_weight + T-bill cash"
OUT_PATH = RESULTS_DIR / "dashboard.json"


def weekly(series: pd.Series) -> dict:
    """Resample a daily series to weekly (Friday) for a compact payload."""
    s = series.resample("W-FRI").last().dropna()
    return {"dates": [d.strftime("%Y-%m-%d") for d in s.index],
            "values": [round(float(v), 4) for v in s.values]}


def equity(returns: pd.Series, start: str) -> pd.Series:
    r = returns.loc[start:].dropna()
    return (1 + r).cumprod()


def window_stats(returns: pd.Series, start: str) -> dict:
    stats = perf_stats(returns.loc[start:])
    r = returns.loc[start:].dropna()
    yearly = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    full_years = yearly[[y for y in yearly.index if len(r.loc[str(y)]) > 200]]
    stats["years_profitable"] = int((full_years > 0).sum())
    stats["years_total"] = int(len(full_years))
    n_years = len(r) / 252
    stats["annualized_return"] = float((1 + stats["return"]) ** (1 / n_years) - 1)
    return stats


def main():
    print("Running 2008+ walk-forward (a few minutes)...")
    out = run_backtest(RULES_UNIVERSE, verbose=False,
                       data_start=DATA_START, eval_start=EVAL_START)
    combo = out["_series"]["combos"][HEADLINE]
    desk_rets = combo["returns"]

    bench = fetch_tickers(["DBC", "SPY"], "rules_benchmarks", start=DATA_START)
    dbc_rets = bench["DBC"].pct_change()
    spy_rets = bench["SPY"].pct_change()

    desk_eq = equity(desk_rets, EVAL_START)
    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "eval_start": EVAL_START,
            "eval_end": desk_eq.index[-1].strftime("%Y-%m-%d"),
            "variant": HEADLINE,
            "universe": {s: sorted(a.values()) for s, a in RULES_UNIVERSE.items()},
            "rules": [
                "Multi-horizon trend (50/100/200d SMA), long/flat",
                "Carry tilt from futures-curve ETF pairs (energy)",
                "Cross-sectional 12-1 momentum across all 15 markets",
                "Volatility targeting: 12% annualized per sleeve",
                "Crisis liquidation: VIX > 35 or sleeve vol z-score > 3",
                "Idle cash earns T-bills (BIL); costs 10 bps on turnover",
            ],
        },
        "stats": {
            "full": window_stats(desk_rets, EVAL_START),
            "recent": window_stats(desk_rets, RECENT_START),
            "benchmarks": {
                "DBC": {"full": window_stats(dbc_rets, EVAL_START),
                        "recent": window_stats(dbc_rets, RECENT_START)},
                "SPY": {"full": window_stats(spy_rets, EVAL_START),
                        "recent": window_stats(spy_rets, RECENT_START)},
            },
        },
        "equity_curves": {
            "desk": weekly(desk_eq),
            "DBC": weekly(equity(dbc_rets, EVAL_START)),
            "SPY": weekly(equity(spy_rets, EVAL_START)),
        },
        "drawdown": weekly(desk_eq / desk_eq.cummax() - 1.0),
        "sleeve_curves": {
            name: weekly(equity(r, EVAL_START))
            for name, r in out["_series"]["sleeve_net_returns"].items()
        },
    }

    # Yearly table: desk return/sharpe/maxDD alongside benchmark returns
    r = desk_rets.loc[EVAL_START:].dropna()
    years = []
    for year, chunk in r.groupby(r.index.year):
        s = perf_stats(chunk)
        row = {"year": int(year), "partial": bool(len(chunk) < 200),
               "return": s["return"], "sharpe": s["sharpe"], "max_dd": s["max_dd"]}
        for label, br in [("dbc", dbc_rets), ("spy", spy_rets)]:
            b = br.loc[str(year)].dropna()
            row[f"{label}_return"] = float((1 + b).prod() - 1) if len(b) else None
        years.append(row)
    payload["years"] = years

    # Monthly sector allocation (gross exposure per sleeve + cash remainder)
    weights = combo["weights"].loc[EVAL_START:]
    alloc = pd.DataFrame({
        name: weights[[c for c in w.columns if c in weights.columns]].sum(axis=1)
        for name, w in out["_series"]["sleeve_weights"].items()
    })
    alloc["cash"] = (1.0 - alloc.sum(axis=1)).clip(lower=0.0)
    alloc_m = alloc.resample("ME").last().dropna()
    payload["allocation"] = {
        "dates": [d.strftime("%Y-%m") for d in alloc_m.index],
        **{c: [round(float(v), 4) for v in alloc_m[c]] for c in alloc.columns},
    }

    OUT_PATH.write_text(json.dumps(payload))
    print(f"Saved -> {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
    f, rec = payload["stats"]["full"], payload["stats"]["recent"]
    print(f"Full 2008+ : sharpe {f['sharpe']:+.2f}  ann.return {f['annualized_return']:+.1%}  maxDD {f['max_dd']:.1%}")
    print(f"Recent 2020+: sharpe {rec['sharpe']:+.2f}  ann.return {rec['annualized_return']:+.1%}  maxDD {rec['max_dd']:.1%}")


if __name__ == "__main__":
    main()
