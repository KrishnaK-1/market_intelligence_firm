"""
Backtest: Gradual vs Steep Slowdown Equity Reduction
=====================================================
Tests Lead=9mo with different Slowdown equity exposure levels.
Patches REGIME_SUB_ALLOCATIONS at runtime to vary how aggressively
the portfolio de-risks when Slowdown is detected.

Variants:
  - Baseline:  No lead window, current Slowdown alloc (25% eq)
  - Steep:     Lead=9mo, current Slowdown alloc (25% eq)
  - Moderate:  Lead=9mo, Slowdown alloc at 35% eq
  - Gentle:    Lead=9mo, Slowdown alloc at 38% eq

Usage:
    python scripts/compare_slowdown_alloc.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import numpy as np
import pandas as pd

from data.ingestion import fetch_all_fred, fetch_shiller_cape, fetch_yahoo_prices, build_feature_matrix
from models.regime_classifier import EnsembleRegimeClassifier
from models.feedback import DEFAULT_PARAMS
from backtest.engine import BacktestEngine
from config import settings
from config.settings import N_REGIMES

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("alloc_compare")
logger.setLevel(logging.INFO)

# Original Slowdown sub-allocations (sums to ~0.80, rest is cash)
ORIGINAL_SLOWDOWN_SUB = dict(settings.REGIME_SUB_ALLOCATIONS[1])

# Different Slowdown allocation profiles
# Key idea: gradually reduce equity, keep some growth exposure
SLOWDOWN_PROFILES = {
    "Steep (25% eq)": {
        # Current: heavy defensive, 25% eq, 20% cash
        "SPY": 0.10, "XLV": 0.05, "XLP": 0.05, "XLU": 0.05,
        "TLT": 0.10, "IEF": 0.10, "TIP": 0.05, "LQD": 0.05,
        "GLD": 0.07, "DBC": 0.03,
        "EFA": 0.10, "EEM": 0.05,
    },
    "Moderate (35% eq)": {
        # Moderate: keep more equity, less cash
        "SPY": 0.15, "XLV": 0.05, "XLP": 0.05, "XLU": 0.04, "XLK": 0.03, "XLI": 0.03,
        "TLT": 0.08, "IEF": 0.08, "TIP": 0.04, "LQD": 0.04,
        "GLD": 0.06, "DBC": 0.03,
        "EFA": 0.08, "EEM": 0.04,
    },
    "Gentle (38% eq)": {
        # Gentle: barely de-risk from expansion, tilt defensive sectors
        "SPY": 0.15, "XLV": 0.06, "XLP": 0.06, "XLU": 0.05, "XLK": 0.03, "XLI": 0.03,
        "TLT": 0.06, "IEF": 0.06, "TIP": 0.04, "LQD": 0.04,
        "GLD": 0.06, "DBC": 0.03,
        "EFA": 0.07, "EEM": 0.04,
    },
}


def run_backtest_with_alloc(feature_matrix, asset_returns, label_params, slowdown_sub, name):
    """Train model and run backtest with custom Slowdown sub-allocations."""
    logger.info(f"Training [{name}]...")
    model = EnsembleRegimeClassifier(n_regimes=N_REGIMES)
    model._label_params = label_params
    model.fit(feature_matrix)

    # Patch the sub-allocations for Slowdown (regime 1)
    original = dict(settings.REGIME_SUB_ALLOCATIONS[1])
    settings.REGIME_SUB_ALLOCATIONS[1] = slowdown_sub

    logger.info(f"Running backtest [{name}]...")
    engine = BacktestEngine(n_splits=5)
    results = engine.run(
        feature_matrix=feature_matrix,
        asset_returns=asset_returns,
        classifier_class=EnsembleRegimeClassifier,
        label_params=label_params,
    )

    # Restore original
    settings.REGIME_SUB_ALLOCATIONS[1] = original
    return results


def run_comparison():
    print("=" * 80)
    print("  BACKTEST: Gradual vs Steep Slowdown De-Risking (Lead=9mo)")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    fred_data = fetch_all_fred()
    shiller_data = fetch_shiller_cape()
    asset_prices = fetch_yahoo_prices()
    feature_matrix = build_feature_matrix(fred_data, shiller_data)
    asset_returns = asset_prices.pct_change().dropna(how="all")
    print(f"  Features: {feature_matrix.shape}, Returns: {asset_returns.shape}")

    baseline_params = dict(DEFAULT_PARAMS)
    baseline_params["pre_recession_lead_months"] = 0

    lead9_params = dict(DEFAULT_PARAMS)
    lead9_params["pre_recession_lead_months"] = 9

    all_results = {}

    # 1. Baseline (no lead, current alloc)
    print("\n[2] Baseline (no lead window, current allocations)...")
    all_results["Baseline (0mo)"] = run_backtest_with_alloc(
        feature_matrix, asset_returns,
        baseline_params, ORIGINAL_SLOWDOWN_SUB, "Baseline"
    )

    # 2-4. Lead=9mo with different Slowdown profiles
    for i, (profile_name, sub_alloc) in enumerate(SLOWDOWN_PROFILES.items()):
        eq_sum = sum(v for k, v in sub_alloc.items()
                     if k in ["SPY", "XLK", "XLF", "XLE", "XLV", "XLI",
                              "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"])
        name = f"Lead=9mo + {profile_name}"
        print(f"\n[{i+3}] {name} (equity sum: {eq_sum:.0%})...")
        all_results[name] = run_backtest_with_alloc(
            feature_matrix, asset_returns,
            lead9_params, sub_alloc, name
        )

    # Results
    print("\n" + "=" * 80)
    print("  RESULTS")
    print("=" * 80)

    # Build header
    names = list(all_results.keys())
    short_names = ["Baseline", "9mo+Steep", "9mo+Moderate", "9mo+Gentle"]

    print(f"\n  {'Metric':<20}", end="")
    for sn in short_names:
        print(f" {sn:>13}", end="")
    print()
    print(f"  {'-' * (20 + 14 * len(short_names))}")

    metrics = [
        ("Sharpe Ratio", "sharpe_ratio", False),
        ("Max Drawdown", "max_drawdown", True),
        ("Total Return", "total_return", True),
        ("Annual Return", "annual_return", True),
        ("Volatility", "annual_vol", True),
    ]

    for label, key, is_pct in metrics:
        row = f"  {label:<20}"
        for name in names:
            val = all_results[name].get("strategy_stats", {}).get(key, 0)
            if is_pct:
                row += f" {val:>12.1%}"
            else:
                row += f" {val:>12.2f}"
        print(row)

    # Benchmark
    bm = all_results[names[0]].get("benchmark_stats", {})
    print(f"\n  Benchmark (SPY): Sharpe={bm.get('sharpe_ratio', 0):.2f}, MaxDD={bm.get('max_drawdown', 0):.1%}")

    # Delta vs baseline
    base_stats = all_results[names[0]].get("strategy_stats", {})
    print(f"\n  DELTA vs BASELINE:")
    print(f"  {'Metric':<20}", end="")
    for sn in short_names[1:]:
        print(f" {sn:>13}", end="")
    print()
    print(f"  {'-' * (20 + 14 * len(short_names[1:]))}")

    for label, key, is_pct in metrics:
        row = f"  {label:<20}"
        base_val = base_stats.get(key, 0)
        for name in names[1:]:
            val = all_results[name].get("strategy_stats", {}).get(key, 0)
            delta = val - base_val
            if is_pct:
                row += f" {delta:>+12.1%}"
            else:
                row += f" {delta:>+12.2f}"
        print(row)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_comparison()