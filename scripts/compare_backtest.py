"""
Backtest Comparison: Baseline vs Option 2 (Forward-Shifted Labels)
===================================================================
Runs walk-forward backtest on both models and compares Sharpe, MaxDD,
annual return, and per-split performance.

Usage:
    python scripts/compare_backtest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import numpy as np
import pandas as pd

from data.ingestion import fetch_all_fred, fetch_shiller_cape, fetch_yahoo_prices, build_feature_matrix
from models.regime_classifier import EnsembleRegimeClassifier
from models.feedback import ParameterOptimizer, DEFAULT_PARAMS
from backtest.engine import BacktestEngine
from config.settings import N_REGIMES

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("backtest_compare")
logger.setLevel(logging.INFO)


def train_and_backtest(feature_matrix, asset_returns, params, name):
    """Train model with given params and run backtest."""
    logger.info(f"Training [{name}]...")
    model = EnsembleRegimeClassifier(n_regimes=N_REGIMES)
    model._label_params = params
    model.fit(feature_matrix)

    logger.info(f"Running backtest [{name}]...")
    engine = BacktestEngine(n_splits=5)
    results = engine.run(
        feature_matrix=feature_matrix,
        asset_returns=asset_returns,
        classifier_class=EnsembleRegimeClassifier,
        label_params=params,
    )
    return model, results


def print_results(name, results):
    """Print backtest summary for one model."""
    stats = results.get("strategy_stats", {})
    bm = results.get("benchmark_stats", {})
    splits = results.get("split_details", [])

    print(f"\n  {name}")
    print(f"  {'-' * 50}")
    print(f"  {'Annual Return:':<25} {stats.get('annual_return', 0):>8.1%}")
    print(f"  {'Sharpe Ratio:':<25} {stats.get('sharpe_ratio', 0):>8.2f}")
    print(f"  {'Max Drawdown:':<25} {stats.get('max_drawdown', 0):>8.1%}")
    print(f"  {'Volatility:':<25} {stats.get('annual_vol', 0):>8.1%}")
    print(f"  {'Total Return:':<25} {stats.get('total_return', 0):>8.1%}")

    if bm:
        print(f"\n  Benchmark (SPY Buy & Hold):")
        print(f"  {'Annual Return:':<25} {bm.get('annual_return', 0):>8.1%}")
        print(f"  {'Sharpe Ratio:':<25} {bm.get('sharpe_ratio', 0):>8.2f}")
        print(f"  {'Max Drawdown:':<25} {bm.get('max_drawdown', 0):>8.1%}")

    if splits:
        print(f"\n  Per-Split Detail:")
        print(f"  {'Split':<8} {'Period':<22} {'Return':>10} {'Sharpe':>10} {'Accuracy':>10}")
        print(f"  {'-' * 62}")
        for s in splits:
            period = f"{s.get('test_start', '?')} - {s.get('test_end', '?')}"
            print(f"  {s.get('split', '?'):<8} {period:<22} "
                  f"{s.get('strategy_return', 0):>9.1%} "
                  f"{s.get('sharpe', 0):>10.2f} "
                  f"{s.get('accuracy', 0):>9.1%}")


def run_comparison():
    print("=" * 70)
    print("  BACKTEST COMPARISON: Baseline vs Option 2")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    fred_data = fetch_all_fred()
    shiller_data = fetch_shiller_cape()
    asset_prices = fetch_yahoo_prices()
    feature_matrix = build_feature_matrix(fred_data, shiller_data)
    asset_returns = asset_prices.pct_change().dropna(how="all")
    print(f"  Features: {feature_matrix.shape}, Returns: {asset_returns.shape}")

    # Baseline params (no lead window)
    optimizer = ParameterOptimizer()
    baseline_params = dict(DEFAULT_PARAMS)
    baseline_params["pre_recession_lead_months"] = 0

    # Test multiple lead windows
    lead_windows = [0, 6, 9, 12]
    all_results = {}

    for i, lead in enumerate(lead_windows):
        params = dict(baseline_params)
        params["pre_recession_lead_months"] = lead
        name = f"Lead={lead}mo" if lead > 0 else "Baseline (0mo)"
        print(f"\n[{i+2}/{len(lead_windows)+1}] Training & backtesting {name}...")
        _, results = train_and_backtest(feature_matrix, asset_returns, params, name)
        all_results[lead] = results

    # Print individual results
    print(f"\n[{len(lead_windows)+1}/{len(lead_windows)+1}] Results")
    print("=" * 70)

    for lead, results in all_results.items():
        name = f"Lead={lead}mo" if lead > 0 else "BASELINE (0mo)"
        print_results(name, results)

    # Side-by-side summary table
    print("\n" + "=" * 80)
    print("  SIDE-BY-SIDE COMPARISON (all lead windows)")
    print("=" * 80)

    header = f"  {'Metric':<20}"
    for lead in lead_windows:
        header += f" {'Lead='+str(lead)+'mo':>12}" if lead > 0 else f" {'Baseline':>12}"
    print(header)
    print(f"  {'-' * (20 + 13 * len(lead_windows))}")

    metrics = [
        ("Sharpe Ratio", "sharpe_ratio", False),
        ("Max Drawdown", "max_drawdown", True),
        ("Total Return", "total_return", True),
        ("Annual Return", "annual_return", True),
        ("Volatility", "annual_vol", True),
    ]
    for label, key, is_pct in metrics:
        row = f"  {label:<20}"
        for lead in lead_windows:
            val = all_results[lead].get("strategy_stats", {}).get(key, 0)
            if is_pct:
                row += f" {val:>11.1%}"
            else:
                row += f" {val:>11.2f}"
        print(row)

    # Benchmark row
    bm = all_results[0].get("benchmark_stats", {})
    print(f"\n  Benchmark (SPY):")
    print(f"  {'Sharpe':<20} {bm.get('sharpe_ratio', 0):>11.2f}")
    print(f"  {'Max Drawdown':<20} {bm.get('max_drawdown', 0):>11.1%}")

    print("=" * 80)


if __name__ == "__main__":
    run_comparison()