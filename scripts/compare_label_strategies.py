"""
A/B Comparison: Current vs Forward-Shifted Slowdown Labels
==========================================================
Trains two models on the same data:
  A (baseline):     Current label strategy (coincident slowdown signals)
  B (experimental): Forward-shifted labels (pre-recession months = Slowdown)

Compares:
  1. Ground truth accuracy (NBER recession + known expansion matching)
  2. Return alignment (regime calls vs S&P 500 returns)
  3. Transition detection lead time (months before recession the model
     first signals Slowdown — the key metric for a forecasting system)
  4. Label distribution (ensures no regime is eliminated)
  5. Regime stability (average duration, noisy switching)

Usage:
    python scripts/compare_label_strategies.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import numpy as np
import pandas as pd
from copy import deepcopy

from data.ingestion import fetch_all_fred, fetch_shiller_cape, build_feature_matrix
from models.regime_classifier import EnsembleRegimeClassifier
from models.feedback import (
    NBER_RECESSIONS,
    KNOWN_EXPANSIONS,
    RETURN_EXPECTATIONS,
    ParameterOptimizer,
)
from config.settings import REGIME_LABELS, N_REGIMES

logging.basicConfig(
    level=logging.WARNING,  # Suppress training noise
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("compare")
logger.setLevel(logging.INFO)


# ── Metrics ──────────────────────────────────────────────────────────


def ground_truth_accuracy(preds: pd.Series, features: pd.DataFrame) -> dict:
    """Check predicted regimes against NBER recession/expansion dates."""
    rec_correct, rec_total = 0, 0
    for start, end in NBER_RECESSIONS:
        mask = (preds.index >= pd.Timestamp(start)) & (preds.index <= pd.Timestamp(end))
        period = preds[mask]
        if len(period) == 0:
            continue
        correct = period.isin(["Contraction", "Crisis"]).sum()
        rec_total += len(period)
        rec_correct += correct

    exp_correct, exp_total = 0, 0
    for start, end in KNOWN_EXPANSIONS:
        mask = (preds.index >= pd.Timestamp(start)) & (preds.index <= pd.Timestamp(end))
        period = preds[mask]
        if len(period) == 0:
            continue
        correct = period.isin(["Expansion", "Recovery"]).sum()
        exp_total += len(period)
        exp_correct += correct

    rec_acc = rec_correct / rec_total if rec_total > 0 else 0
    exp_acc = exp_correct / exp_total if exp_total > 0 else 0
    total = rec_total + exp_total
    overall = (rec_correct + exp_correct) / total if total > 0 else 0

    return {
        "recession_accuracy": rec_acc,
        "expansion_accuracy": exp_acc,
        "overall_accuracy": overall,
    }


def return_alignment(preds: pd.Series, features: pd.DataFrame) -> dict:
    """Check if regime calls align with S&P 500 YoY returns."""
    if "sp500_yoy" not in features.columns:
        return {"overall_alignment": 0}

    aligned, total = 0, 0
    for date in preds.index:
        if date not in features.index:
            continue
        sp500 = features.loc[date, "sp500_yoy"]
        if pd.isna(sp500):
            continue
        total += 1
        pred = preds[date]
        exp = RETURN_EXPECTATIONS.get(pred, {})
        if exp.get("sp500_yoy_min", -999) <= sp500 <= exp.get("sp500_yoy_max", 999):
            aligned += 1

    return {"overall_alignment": aligned / total if total > 0 else 0}


def transition_detection_lead(preds: pd.Series) -> dict:
    """
    For each NBER recession, measure how many months BEFORE the recession
    start the model first signals Slowdown (or Contraction/Crisis).

    This is the KEY metric: a forecasting model should detect trouble
    months before NBER officially dates the recession.
    """
    results = []
    for start_str, end_str in NBER_RECESSIONS:
        rec_start = pd.Timestamp(start_str)
        # Look at the 18 months before each recession
        lookback = pd.DateOffset(months=18)
        pre_window = preds[
            (preds.index >= rec_start - lookback) & (preds.index < rec_start)
        ]
        if len(pre_window) == 0:
            continue

        # Find first non-Expansion signal
        warning_signals = pre_window[pre_window.isin(["Slowdown", "Contraction", "Crisis"])]
        if len(warning_signals) > 0:
            first_warning = warning_signals.index[0]
            lead_months = (rec_start.year - first_warning.year) * 12 + (
                rec_start.month - first_warning.month
            )
            results.append({
                "recession": start_str,
                "first_warning": first_warning.strftime("%Y-%m"),
                "lead_months": lead_months,
            })
        else:
            results.append({
                "recession": start_str,
                "first_warning": "NONE",
                "lead_months": 0,
            })

    avg_lead = np.mean([r["lead_months"] for r in results]) if results else 0
    detected = sum(1 for r in results if r["lead_months"] > 0)

    return {
        "avg_lead_months": avg_lead,
        "recessions_detected_early": detected,
        "total_recessions": len(results),
        "detection_rate": detected / len(results) if results else 0,
        "details": results,
    }


def regime_stability(preds: pd.Series) -> dict:
    """Measure regime switching frequency."""
    changes = (preds != preds.shift()).sum()
    avg_duration = len(preds) / max(changes, 1)
    return {
        "total_changes": int(changes),
        "avg_duration_months": round(avg_duration, 1),
        "is_noisy": avg_duration < 4,
    }


def label_distribution(preds: pd.Series) -> dict:
    """Count of each regime in the classification."""
    counts = preds.value_counts().to_dict()
    return {k: int(v) for k, v in sorted(counts.items())}


# ── Model Training ───────────────────────────────────────────────────


def train_model(feature_matrix: pd.DataFrame, label_params: dict, name: str):
    """Train a fresh classifier with given label params."""
    logger.info(f"Training model [{name}]...")
    model = EnsembleRegimeClassifier(n_regimes=N_REGIMES)
    model._label_params = label_params
    model.fit(feature_matrix)

    # Get predictions
    features_only = feature_matrix.drop(columns=["recession"], errors="ignore")
    regime_proba = model.predict_proba(features_only)
    regime_idx = regime_proba.values.argmax(axis=1)
    preds = pd.Series(
        [REGIME_LABELS[i] for i in regime_idx],
        index=regime_proba.index,
        name="regime",
    )
    return model, preds, regime_proba


# ── Main Comparison ──────────────────────────────────────────────────


def run_comparison():
    print("=" * 70)
    print("  A/B COMPARISON: Current vs Forward-Shifted Slowdown Labels")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    fred_data = fetch_all_fred()
    shiller_data = fetch_shiller_cape()
    feature_matrix = build_feature_matrix(fred_data, shiller_data)
    print(f"  Feature matrix: {feature_matrix.shape}")

    # Get current optimized params as baseline
    optimizer = ParameterOptimizer()
    baseline_params = optimizer.get_params()

    # Experimental params: same as baseline + forward-shifted labels
    experimental_params = dict(baseline_params)
    experimental_params["pre_recession_lead_months"] = 9  # new param

    # ── Train Model A (Baseline) ──
    print("\n[2/4] Training Model A (baseline — current labels)...")
    model_a, preds_a, proba_a = train_model(feature_matrix, baseline_params, "Baseline")

    # ── Train Model B (Experimental) ──
    print("\n[3/4] Training Model B (experimental — forward-shifted labels)...")
    model_b, preds_b, proba_b = train_model(feature_matrix, experimental_params, "Experimental")

    # ── Compare ──
    print("\n[4/4] Computing comparison metrics...")
    features_only = feature_matrix.drop(columns=["recession"], errors="ignore")

    metrics_a = {
        "ground_truth": ground_truth_accuracy(preds_a, feature_matrix),
        "returns": return_alignment(preds_a, feature_matrix),
        "transition_lead": transition_detection_lead(preds_a),
        "stability": regime_stability(preds_a),
        "distribution": label_distribution(preds_a),
    }
    metrics_b = {
        "ground_truth": ground_truth_accuracy(preds_b, feature_matrix),
        "returns": return_alignment(preds_b, feature_matrix),
        "transition_lead": transition_detection_lead(preds_b),
        "stability": regime_stability(preds_b),
        "distribution": label_distribution(preds_b),
    }

    # ── Print Results ──
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print("\n  GROUND TRUTH ACCURACY (higher is better)")
    print(f"  {'Metric':<30} {'Baseline':>12} {'Experimental':>12} {'Delta':>10}")
    print(f"  {'-'*64}")
    for key in ["recession_accuracy", "expansion_accuracy", "overall_accuracy"]:
        a_val = metrics_a["ground_truth"][key]
        b_val = metrics_b["ground_truth"][key]
        delta = b_val - a_val
        marker = " **" if abs(delta) > 0.02 else ""
        print(f"  {key:<30} {a_val:>11.1%} {b_val:>12.1%} {delta:>+9.1%}{marker}")

    print("\n  RETURN ALIGNMENT (higher is better)")
    a_val = metrics_a["returns"]["overall_alignment"]
    b_val = metrics_b["returns"]["overall_alignment"]
    delta = b_val - a_val
    print(f"  {'overall_alignment':<30} {a_val:>11.1%} {b_val:>12.1%} {delta:>+9.1%}")

    print("\n  TRANSITION DETECTION (higher lead = better forecasting)")
    print(f"  {'Metric':<30} {'Baseline':>12} {'Experimental':>12} {'Delta':>10}")
    print(f"  {'-'*64}")
    for key in ["avg_lead_months", "detection_rate"]:
        a_val = metrics_a["transition_lead"][key]
        b_val = metrics_b["transition_lead"][key]
        delta = b_val - a_val
        if key == "detection_rate":
            print(f"  {key:<30} {a_val:>11.1%} {b_val:>12.1%} {delta:>+9.1%}")
        else:
            print(f"  {key:<30} {a_val:>11.1f} {b_val:>12.1f} {delta:>+9.1f}")

    # Composite score (what the optimizer uses + transition lead bonus)
    def composite(m):
        gt = m["ground_truth"]["overall_accuracy"]
        ret = m["returns"]["overall_alignment"]
        lead = min(m["transition_lead"]["avg_lead_months"] / 12, 1.0)  # cap at 12mo
        # Weight: 40% ground truth, 30% returns, 30% transition lead
        return 0.40 * gt + 0.30 * ret + 0.30 * lead

    score_a = composite(metrics_a)
    score_b = composite(metrics_b)

    print(f"\n  COMPOSITE SCORE (40% GT + 30% Returns + 30% Lead Time)")
    print(f"  {'Baseline':<30} {score_a:>11.3f}")
    print(f"  {'Experimental':<30} {score_b:>11.3f}")
    print(f"  {'Delta':<30} {score_b - score_a:>+11.3f}")

    print("\n  REGIME STABILITY")
    print(f"  {'Metric':<30} {'Baseline':>12} {'Experimental':>12}")
    print(f"  {'-'*54}")
    print(f"  {'avg_duration_months':<30} {metrics_a['stability']['avg_duration_months']:>12.1f} {metrics_b['stability']['avg_duration_months']:>12.1f}")
    print(f"  {'total_changes':<30} {metrics_a['stability']['total_changes']:>12d} {metrics_b['stability']['total_changes']:>12d}")

    print("\n  LABEL DISTRIBUTION")
    print(f"  {'Regime':<30} {'Baseline':>12} {'Experimental':>12}")
    print(f"  {'-'*54}")
    all_regimes = sorted(set(list(metrics_a["distribution"].keys()) + list(metrics_b["distribution"].keys())))
    for regime in all_regimes:
        a_val = metrics_a["distribution"].get(regime, 0)
        b_val = metrics_b["distribution"].get(regime, 0)
        print(f"  {regime:<30} {a_val:>12d} {b_val:>12d}")

    print("\n  PER-RECESSION LEAD TIME DETAIL")
    print(f"  {'Recession':<15} {'Baseline Lead':>15} {'Experimental Lead':>18}")
    print(f"  {'-'*50}")
    for det_a, det_b in zip(
        metrics_a["transition_lead"]["details"],
        metrics_b["transition_lead"]["details"],
    ):
        a_lead = f"{det_a['lead_months']}mo" if det_a["lead_months"] > 0 else "MISSED"
        b_lead = f"{det_b['lead_months']}mo" if det_b["lead_months"] > 0 else "MISSED"
        print(f"  {det_a['recession']:<15} {a_lead:>15} {b_lead:>18}")

    # ── Verdict ──
    print("\n" + "=" * 70)
    if score_b > score_a + 0.01:
        print("  VERDICT: Experimental model is BETTER (keep the change)")
    elif score_b < score_a - 0.01:
        print("  VERDICT: Baseline model is BETTER (revert the change)")
    else:
        print("  VERDICT: Models are EQUIVALENT (check per-recession details)")
    print("=" * 70)

    return metrics_a, metrics_b


if __name__ == "__main__":
    run_comparison()