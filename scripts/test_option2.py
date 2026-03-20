"""
Option 2 Test: Optimizer-Tuned Pre-Recession Lead Window
=========================================================
Adds pre_recession_lead_months to the optimizer search space,
runs optimization, trains the best model, and shows:
  - Current regime classification (full ensemble)
  - Regime probabilities
  - Nowcast regime (XGBoost real-time)
  - Transition detection performance
  - Comparison to baseline (no lead window)

Usage:
    python scripts/test_option2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import numpy as np
import pandas as pd

from data.ingestion import fetch_all_fred, fetch_shiller_cape, build_feature_matrix
from data.nowcast import run_nowcast
from models.regime_classifier import EnsembleRegimeClassifier
from models.feedback import (
    NBER_RECESSIONS,
    KNOWN_EXPANSIONS,
    RETURN_EXPECTATIONS,
    ParameterOptimizer,
    MIN_LABEL_COUNTS,
)
from config.settings import REGIME_LABELS, N_REGIMES

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("option2")
logger.setLevel(logging.INFO)


def ground_truth_accuracy(preds, features):
    rec_correct, rec_total = 0, 0
    for start, end in NBER_RECESSIONS:
        mask = (preds.index >= pd.Timestamp(start)) & (preds.index <= pd.Timestamp(end))
        period = preds[mask]
        if len(period) == 0:
            continue
        rec_correct += period.isin(["Contraction", "Crisis"]).sum()
        rec_total += len(period)

    exp_correct, exp_total = 0, 0
    for start, end in KNOWN_EXPANSIONS:
        mask = (preds.index >= pd.Timestamp(start)) & (preds.index <= pd.Timestamp(end))
        period = preds[mask]
        if len(period) == 0:
            continue
        exp_correct += period.isin(["Expansion", "Recovery"]).sum()
        exp_total += len(period)

    total = rec_total + exp_total
    return {
        "recession_acc": rec_correct / rec_total if rec_total else 0,
        "expansion_acc": exp_correct / exp_total if exp_total else 0,
        "overall": (rec_correct + exp_correct) / total if total else 0,
    }


def return_alignment(preds, features):
    if "sp500_yoy" not in features.columns:
        return 0
    aligned, total = 0, 0
    for date in preds.index:
        if date not in features.index:
            continue
        sp500 = features.loc[date, "sp500_yoy"]
        if pd.isna(sp500):
            continue
        total += 1
        exp = RETURN_EXPECTATIONS.get(preds[date], {})
        if exp.get("sp500_yoy_min", -999) <= sp500 <= exp.get("sp500_yoy_max", 999):
            aligned += 1
    return aligned / total if total else 0


def transition_detection(preds):
    results = []
    for start_str, end_str in NBER_RECESSIONS:
        rec_start = pd.Timestamp(start_str)
        pre = preds[(preds.index >= rec_start - pd.DateOffset(months=18)) & (preds.index < rec_start)]
        warnings = pre[pre.isin(["Slowdown", "Contraction", "Crisis"])]
        if len(warnings) > 0:
            first = warnings.index[0]
            lead = (rec_start.year - first.year) * 12 + (rec_start.month - first.month)
            results.append({"recession": start_str, "lead": lead})
        else:
            results.append({"recession": start_str, "lead": 0})
    return results


def check_label_distribution(label_counts):
    for regime_idx, min_count in MIN_LABEL_COUNTS.items():
        if label_counts.get(regime_idx, 0) < min_count:
            return False
    return True


def get_preds(model, feature_matrix):
    features_only = feature_matrix.drop(columns=["recession"], errors="ignore")
    proba = model.predict_proba(features_only)
    idx = proba.values.argmax(axis=1)
    return pd.Series([REGIME_LABELS[i] for i in idx], index=proba.index), proba


def run_test():
    print("=" * 70)
    print("  OPTION 2: Optimizer-Tuned Pre-Recession Lead Window")
    print("=" * 70)

    # ── Load data ──
    print("\n[1/5] Loading data...")
    fred_data = fetch_all_fred()
    shiller_data = fetch_shiller_cape()
    feature_matrix = build_feature_matrix(fred_data, shiller_data)
    features_only = feature_matrix.drop(columns=["recession"], errors="ignore")
    print(f"  Features: {feature_matrix.shape}")

    # ── Train baseline ──
    print("\n[2/5] Training baseline model (current params, no lead window)...")
    optimizer = ParameterOptimizer()
    baseline_params = optimizer.get_params()
    # Force lead to 0 for baseline
    baseline_params["pre_recession_lead_months"] = 0

    model_baseline = EnsembleRegimeClassifier(n_regimes=N_REGIMES)
    model_baseline._label_params = baseline_params
    model_baseline.fit(feature_matrix)
    preds_baseline, proba_baseline = get_preds(model_baseline, feature_matrix)

    # ── Optimize with lead window in search space ──
    print("\n[3/5] Running optimizer with lead window in search space (40 iterations)...")

    search_space = {
        "yield_curve_threshold": [-1.0, -0.7, -0.5, -0.3, -0.1, 0.2, 0.5],
        "indpro_threshold": [-0.5, 0.0, 0.5, 0.8, 1.5, 2.0],
        "slowdown_signal_count": [2, 3, 4, 5],
        "sentiment_threshold": [-12.0, -8.0, -5.0, -3.0],
        "payems_threshold": [-0.1, -0.05, 0.0],
        "recovery_months": [4, 6, 8],
        "unrate_accel_threshold": [0.1, 0.2, 0.3, 0.5],
        "sp500_mom3m_threshold": [-10.0, -5.0, -3.0, 0.0],
        "credit_accel_threshold": [0.1, 0.2, 0.3, 0.5],
        "umcsent_level_threshold": [55.0, 60.0, 65.0, 70.0, 75.0],
        "pre_recession_lead_months": [0, 6, 9, 12],  # THE NEW PARAM
    }

    rng = np.random.RandomState(42)
    best_score = 0
    best_params = dict(baseline_params)
    n_iterations = 40

    for i in range(n_iterations):
        trial_params = dict(baseline_params)
        for param, values in search_space.items():
            trial_params[param] = rng.choice(values)

        try:
            model = EnsembleRegimeClassifier(n_regimes=N_REGIMES)
            model._label_params = trial_params
            model.fit(feature_matrix)

            # Check label distribution
            X_unscaled = features_only.ffill().bfill()
            X_unscaled = X_unscaled[X_unscaled.notna().all(axis=1)]
            target = feature_matrix["recession"].reindex(X_unscaled.index).dropna()
            common = X_unscaled.index.intersection(target.index)
            test_labels = model._create_5class_labels(X_unscaled.loc[common], target.loc[common])
            label_counts = test_labels.value_counts().to_dict()

            if not check_label_distribution(label_counts):
                continue

            preds, _ = get_preds(model, feature_matrix)
            gt = ground_truth_accuracy(preds, feature_matrix)
            ret = return_alignment(preds, feature_matrix)

            # Transition detection bonus in scoring
            trans = transition_detection(preds)
            avg_lead = np.mean([t["lead"] for t in trans])
            detect_rate = sum(1 for t in trans if t["lead"] > 0) / len(trans)
            lead_bonus = min(avg_lead / 12, 1.0)

            # Composite: 35% GT + 30% returns + 35% transition lead
            score = 0.35 * gt["overall"] + 0.30 * ret + 0.35 * lead_bonus

            lead_val = trial_params["pre_recession_lead_months"]
            if score > best_score:
                best_score = score
                best_params = dict(trial_params)
                print(f"  Iter {i:2d}: score={score:.3f} (GT={gt['overall']:.1%}, "
                      f"Ret={ret:.1%}, Lead={avg_lead:.1f}mo, Det={detect_rate:.0%}) "
                      f"pre_rec_lead={lead_val} ** BEST **")

        except Exception as e:
            pass

    print(f"\n  Best score: {best_score:.3f}")
    print(f"  Best pre_recession_lead_months: {best_params['pre_recession_lead_months']}")

    # ── Train final model with best params ──
    print("\n[4/5] Training final model with optimized params...")
    model_final = EnsembleRegimeClassifier(n_regimes=N_REGIMES)
    model_final._label_params = best_params
    model_final.fit(feature_matrix)
    preds_final, proba_final = get_preds(model_final, feature_matrix)

    # ── Run nowcast with both models ──
    print("\n[5/5] Running nowcast with both models...")
    nowcast_baseline = run_nowcast(fred_monthly=fred_data, classifier=model_baseline)
    nowcast_final = run_nowcast(fred_monthly=fred_data, classifier=model_final)

    # ══════════════════════════════════════════════════════════════════
    #  RESULTS
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # Current regime from full ensemble
    current_proba_baseline = proba_baseline.iloc[-1]
    current_regime_baseline = current_proba_baseline.idxmax()
    current_conf_baseline = current_proba_baseline.max()

    current_proba_final = proba_final.iloc[-1]
    current_regime_final = current_proba_final.idxmax()
    current_conf_final = current_proba_final.max()

    print("\n  +-----------------------------------------------------------+")
    print("  |  REGIME CLASSIFICATION (Full Ensemble -- Latest Month)   |")
    print("  +-----------------------------------------------------------+")
    print(f"  |  Baseline:     {current_regime_baseline:<14} ({current_conf_baseline:.1%} confidence)  |")
    print(f"  |  Optimized:    {current_regime_final:<14} ({current_conf_final:.1%} confidence)  |")
    print("  +-----------------------------------------------------------+")

    print("\n  Regime Probabilities (Latest Month - Full Ensemble):")
    print(f"  {'Regime':<15} {'Baseline':>10} {'Optimized':>10}")
    print(f"  {'-'*37}")
    for col in proba_baseline.columns:
        bv = current_proba_baseline[col]
        fv = current_proba_final[col]
        marker = " <--" if col == current_regime_final else ""
        print(f"  {col:<15} {bv:>9.1%} {fv:>10.1%}{marker}")

    print("\n  +-----------------------------------------------------------+")
    print("  |  NOWCAST (XGBoost Real-Time -- Current Data)             |")
    print("  +-----------------------------------------------------------+")
    nc_b = nowcast_baseline
    nc_f = nowcast_final
    print(f"  |  Baseline:     {nc_b.get('regime', '?'):<14} ({nc_b.get('confidence', 0):.1%} confidence)  |")
    print(f"  |  Optimized:    {nc_f.get('regime', '?'):<14} ({nc_f.get('confidence', 0):.1%} confidence)  |")
    print("  +-----------------------------------------------------------+")

    print("\n  Nowcast Probabilities:")
    print(f"  {'Regime':<15} {'Baseline':>10} {'Optimized':>10}")
    print(f"  {'-'*37}")
    for regime in ["Expansion", "Slowdown", "Contraction", "Recovery", "Crisis"]:
        bv = nc_b.get("probabilities", {}).get(regime, 0)
        fv = nc_f.get("probabilities", {}).get(regime, 0)
        print(f"  {regime:<15} {bv:>9.1%} {fv:>10.1%}")

    if nc_f.get("signals_firing"):
        print("\n  Nowcast Signals Firing:")
        for sig in nc_f["signals_firing"]:
            print(f"    - {sig}")

    # Transition detection comparison
    print("\n  TRANSITION DETECTION COMPARISON")
    trans_b = transition_detection(preds_baseline)
    trans_f = transition_detection(preds_final)
    print(f"  {'Recession':<15} {'Baseline':>12} {'Optimized':>12}")
    print(f"  {'-'*41}")
    for tb, tf in zip(trans_b, trans_f):
        bl = f"{tb['lead']}mo" if tb["lead"] > 0 else "MISSED"
        fl = f"{tf['lead']}mo" if tf["lead"] > 0 else "MISSED"
        print(f"  {tb['recession']:<15} {bl:>12} {fl:>12}")

    avg_b = np.mean([t["lead"] for t in trans_b])
    avg_f = np.mean([t["lead"] for t in trans_f])
    det_b = sum(1 for t in trans_b if t["lead"] > 0)
    det_f = sum(1 for t in trans_f if t["lead"] > 0)
    print(f"  {'Avg lead':<15} {avg_b:>11.1f}mo {avg_f:>11.1f}mo")
    print(f"  {'Detected':<15} {det_b:>10}/{len(trans_b)} {det_f:>10}/{len(trans_f)}")

    # Accuracy comparison
    gt_b = ground_truth_accuracy(preds_baseline, feature_matrix)
    gt_f = ground_truth_accuracy(preds_final, feature_matrix)
    ret_b = return_alignment(preds_baseline, feature_matrix)
    ret_f = return_alignment(preds_final, feature_matrix)

    print("\n  ACCURACY COMPARISON")
    print(f"  {'Metric':<25} {'Baseline':>10} {'Optimized':>10} {'Delta':>10}")
    print(f"  {'-'*57}")
    print(f"  {'GT Overall':<25} {gt_b['overall']:>9.1%} {gt_f['overall']:>10.1%} {gt_f['overall']-gt_b['overall']:>+9.1%}")
    print(f"  {'GT Recession':<25} {gt_b['recession_acc']:>9.1%} {gt_f['recession_acc']:>10.1%} {gt_f['recession_acc']-gt_b['recession_acc']:>+9.1%}")
    print(f"  {'GT Expansion':<25} {gt_b['expansion_acc']:>9.1%} {gt_f['expansion_acc']:>10.1%} {gt_f['expansion_acc']-gt_b['expansion_acc']:>+9.1%}")
    print(f"  {'Return Alignment':<25} {ret_b:>9.1%} {ret_f:>10.1%} {ret_f-ret_b:>+9.1%}")

    # Best params summary
    print("\n  OPTIMIZED PARAMETERS:")
    print(f"    pre_recession_lead_months: {best_params['pre_recession_lead_months']}")
    print(f"    slowdown_signal_count:     {best_params['slowdown_signal_count']}")
    print(f"    yield_curve_threshold:     {best_params['yield_curve_threshold']}")
    print(f"    recovery_months:           {best_params['recovery_months']}")

    print("\n" + "=" * 70)
    if best_params["pre_recession_lead_months"] > 0:
        print(f"  Optimizer chose pre_recession_lead_months = {best_params['pre_recession_lead_months']}")
        print("  The lead window IS helping — implement option 2.")
    else:
        print("  Optimizer chose pre_recession_lead_months = 0 (disabled)")
        print("  The lead window did NOT help with these params — keep baseline.")
    print("=" * 70)


if __name__ == "__main__":
    run_test()