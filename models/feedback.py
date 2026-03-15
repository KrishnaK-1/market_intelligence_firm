"""
Reinforcement Feedback Loop & Parameter Optimizer
===================================================
Validates regime classifications against NBER dates, known expansions,
and market return profiles. Persists optimized parameters to disk.

Key constraint: optimizer rejects parameter sets that eliminate
any regime class (minimum count per regime enforced).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import MODEL_DIR, N_REGIMES, REGIME_LABELS

logger = logging.getLogger(__name__)

PARAMS_FILE = MODEL_DIR / "optimized_params.json"
HISTORY_FILE = MODEL_DIR / "optimization_history.json"

# ── Known economic history ──

NBER_RECESSIONS = [
    ("1953-07", "1954-05"),
    ("1957-08", "1958-04"),
    ("1960-04", "1961-02"),
    ("1969-12", "1970-11"),
    ("1973-11", "1975-03"),
    ("1980-01", "1980-07"),
    ("1981-07", "1982-11"),
    ("1990-07", "1991-03"),
    ("2001-03", "2001-11"),
    ("2007-12", "2009-06"),
    ("2020-02", "2020-04"),
]

KNOWN_EXPANSIONS = [
    ("1983-01", "1990-06"),
    ("1991-04", "2001-02"),
    ("2003-07", "2007-11"),
    ("2010-07", "2019-12"),
    ("2020-06", "2024-06"),
]

RETURN_EXPECTATIONS = {
    "Expansion":   {"sp500_yoy_min": 5,    "sp500_yoy_max": 45},
    "Slowdown":    {"sp500_yoy_min": -25,  "sp500_yoy_max": 20},
    "Contraction": {"sp500_yoy_min": -50,  "sp500_yoy_max": 5},
    "Recovery":    {"sp500_yoy_min": 0,    "sp500_yoy_max": 50},
    "Crisis":      {"sp500_yoy_min": -60,  "sp500_yoy_max": -5},
}

# Minimum label counts -- optimizer rejects solutions below these
MIN_LABEL_COUNTS = {
    0: 350,   # Expansion: most of history
    1: 40,    # Slowdown: real slowdowns exist
    2: 60,    # Contraction: NBER recessions
    3: 25,    # Recovery: post-recession rebounds
    4: 8,     # Crisis: rare but real
}

DEFAULT_PARAMS = {
    "yield_curve_threshold": -0.2,
    "indpro_threshold": 0.5,
    "sentiment_threshold": -8.0,
    "payems_threshold": -0.05,
    "sp500_threshold": -5.0,
    "credit_spread_quantile": 0.75,
    "slowdown_signal_count": 3,
    "crisis_vix_quantile": 0.95,
    "crisis_spread_quantile": 0.95,
    "recovery_months": 6,
    "unrate_accel_threshold": 0.3,
    "sp500_mom3m_threshold": -5.0,
    "credit_accel_threshold": 0.3,
    "umcsent_level_threshold": 65.0,
}


class ParameterOptimizer:
    """Manages calibrated parameters. Persists to disk."""

    def __init__(self):
        self._params = dict(DEFAULT_PARAMS)
        self._best_accuracy = 0.0
        self._best_params: Optional[Dict] = None
        self._history: List[Dict] = []
        self._load_from_disk()

    def _load_from_disk(self):
        if PARAMS_FILE.exists():
            try:
                with open(PARAMS_FILE, "r") as f:
                    saved = json.load(f)
                self._best_params = saved.get("best_params")
                self._best_accuracy = saved.get("best_accuracy", 0)
                self._params = saved.get("current_params", dict(DEFAULT_PARAMS))
                # Ensure new params exist in loaded params
                for k, v in DEFAULT_PARAMS.items():
                    if self._best_params and k not in self._best_params:
                        self._best_params[k] = v
                    if k not in self._params:
                        self._params[k] = v
                logger.info(
                    f"Loaded saved params (accuracy={self._best_accuracy:.1%}): "
                    f"yc={self._best_params.get('yield_curve_threshold', '?') if self._best_params else '?'}, "
                    f"signals>={self._best_params.get('slowdown_signal_count', '?') if self._best_params else '?'}"
                )
            except Exception as e:
                logger.warning(f"Failed to load saved params: {e}")

        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r") as f:
                    self._history = json.load(f)
            except Exception:
                pass

    def _save_to_disk(self):
        try:
            def to_native(obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                return obj

            save_data = {
                "best_params": {k: to_native(v) for k, v in self._best_params.items()} if self._best_params else None,
                "best_accuracy": float(self._best_accuracy),
                "current_params": {k: to_native(v) for k, v in self._params.items()},
                "last_updated": datetime.now().isoformat(),
            }
            with open(PARAMS_FILE, "w") as f:
                json.dump(save_data, f, indent=2)

            history_to_save = []
            for h in self._history[-100:]:
                record = dict(h)
                if "params" in record:
                    record["params"] = {k: to_native(v) for k, v in record["params"].items()}
                if "accuracy" in record:
                    record["accuracy"] = float(record["accuracy"])
                history_to_save.append(record)
            with open(HISTORY_FILE, "w") as f:
                json.dump(history_to_save, f, indent=2)

            logger.info(f"Saved optimized params to {PARAMS_FILE}")
        except Exception as e:
            logger.error(f"Failed to save params: {e}")

    def get_params(self) -> Dict:
        if self._best_params:
            return dict(self._best_params)
        return dict(self._params)

    def record_accuracy(self, accuracy: float, params: Optional[Dict] = None):
        p = params or dict(self._params)
        self._history.append({
            "timestamp": datetime.now().isoformat(),
            "params": dict(p),
            "accuracy": float(accuracy),
        })
        if accuracy > self._best_accuracy:
            self._best_accuracy = accuracy
            self._best_params = dict(p)
            logger.info(f"  New best accuracy: {accuracy:.1%}")
            self._save_to_disk()

    def update_best(self, params: Dict, accuracy: float):
        self._best_params = dict(params)
        self._best_accuracy = float(accuracy)
        self._params = dict(params)
        self._save_to_disk()

    def get_history(self) -> List[Dict]:
        return self._history


class FeedbackLoop:
    """Validation + optimization + persistence."""

    def __init__(self, param_optimizer: Optional[ParameterOptimizer] = None):
        self._param_optimizer = param_optimizer or ParameterOptimizer()

    @property
    def param_optimizer(self) -> ParameterOptimizer:
        return self._param_optimizer

    def run_feedback(
        self,
        feature_matrix: pd.DataFrame,
        regime_history: pd.DataFrame,
        sp500_returns: Optional[pd.Series] = None,
        classifier_class=None,
        optimize: bool = False,
        n_iterations: int = 30,
    ) -> Dict:
        logger.info("Running reinforcement feedback validation...")

        if regime_history is None or len(regime_history) == 0:
            return {"error": "No regime history available"}

        regime_idx = regime_history.values.argmax(axis=1)
        predicted_regimes = pd.Series(
            [REGIME_LABELS[i] for i in regime_idx],
            index=regime_history.index,
        )

        gt_report = self._validate_ground_truth(predicted_regimes, feature_matrix)
        ret_report = self._validate_returns(predicted_regimes, feature_matrix)
        recommendations = self._generate_recommendations(predicted_regimes, feature_matrix, gt_report, ret_report)
        stability = self._check_stability(predicted_regimes)

        self._param_optimizer.record_accuracy(gt_report.get("accuracy", 0))

        optimization_report = None
        if optimize and classifier_class is not None:
            optimization_report = self._run_optimization(
                feature_matrix, classifier_class, n_iterations
            )

        return {
            "ground_truth_validation": gt_report,
            "return_validation": ret_report,
            "recommendations": recommendations,
            "stability": stability,
            "optimization": optimization_report,
            "saved_best_params": self._param_optimizer.get_params(),
            "timestamp": datetime.now().isoformat(),
        }

    def _validate_ground_truth(self, preds: pd.Series, features: pd.DataFrame) -> Dict:
        results = {"recession_accuracy": 0, "expansion_accuracy": 0, "details": []}

        recession_correct = 0
        recession_total = 0
        for start, end in NBER_RECESSIONS:
            mask = (preds.index >= pd.Timestamp(start)) & (preds.index <= pd.Timestamp(end))
            period = preds[mask]
            if len(period) == 0:
                continue
            correct = period.isin(["Contraction", "Crisis"]).sum()
            recession_total += len(period)
            recession_correct += correct
            acc = correct / len(period)
            if acc < 0.7:
                results["details"].append(f"Recession {start}->{end}: {acc:.0%} ({correct}/{len(period)})")

        results["recession_accuracy"] = recession_correct / recession_total if recession_total > 0 else 0

        expansion_correct = 0
        expansion_total = 0
        for start, end in KNOWN_EXPANSIONS:
            mask = (preds.index >= pd.Timestamp(start)) & (preds.index <= pd.Timestamp(end))
            period = preds[mask]
            if len(period) == 0:
                continue
            correct = period.isin(["Expansion", "Recovery"]).sum()
            expansion_total += len(period)
            expansion_correct += correct
            acc = correct / len(period)
            if acc < 0.7:
                results["details"].append(f"Expansion {start}->{end}: {acc:.0%} ({correct}/{len(period)})")

        results["expansion_accuracy"] = expansion_correct / expansion_total if expansion_total > 0 else 0

        total = recession_total + expansion_total
        correct = recession_correct + expansion_correct
        results["accuracy"] = correct / total if total > 0 else 0

        logger.info(
            f"  Ground truth: recession={results['recession_accuracy']:.1%}, "
            f"expansion={results['expansion_accuracy']:.1%}, "
            f"overall={results['accuracy']:.1%}"
        )
        return results

    def _validate_returns(self, preds: pd.Series, features: pd.DataFrame) -> Dict:
        if "sp500_yoy" not in features.columns:
            return {"overall_alignment": 0}

        aligned = 0
        total = 0
        mismatches = []

        for date in preds.index:
            if date not in features.index:
                continue
            pred = preds[date]
            sp500 = features.loc[date, "sp500_yoy"]
            if pd.isna(sp500):
                continue
            total += 1

            expectations = RETURN_EXPECTATIONS.get(pred, {})
            sp_min = expectations.get("sp500_yoy_min", -999)
            sp_max = expectations.get("sp500_yoy_max", 999)

            if sp_min <= sp500 <= sp_max:
                aligned += 1
            elif sp500 > sp_max + 10 or sp500 < sp_min - 10:
                mismatches.append({
                    "date": date.strftime("%Y-%m"),
                    "predicted": pred,
                    "sp500_yoy": round(float(sp500), 1),
                })

        alignment = aligned / total if total > 0 else 0
        logger.info(f"  Return alignment: {alignment:.1%} ({aligned}/{total}), {len(mismatches)} major mismatches")
        return {"overall_alignment": alignment, "aligned": aligned, "total": total, "major_mismatches": mismatches[:20]}

    def _generate_recommendations(self, preds, features, gt_report, ret_report) -> List[str]:
        recs = []
        exp_acc = gt_report.get("expansion_accuracy", 1.0)
        if exp_acc < 0.75:
            recs.append(f"Expansion accuracy is {exp_acc:.0%}. Too many expansion months classified as Slowdown.")
        rec_acc = gt_report.get("recession_accuracy", 1.0)
        if rec_acc < 0.80:
            recs.append(f"Recession accuracy is {rec_acc:.0%}. Some recession months missed.")
        alignment = ret_report.get("overall_alignment", 1.0)
        if alignment < 0.70:
            recs.append(f"Return alignment is {alignment:.0%}. Regime calls don't match market behavior.")
        if not recs:
            recs.append("Model validation looks good. No major issues detected.")
        return recs

    def _check_stability(self, preds: pd.Series) -> Dict:
        changes = (preds != preds.shift()).sum()
        avg_duration = len(preds) / max(changes, 1)
        return {
            "total_changes": int(changes),
            "avg_regime_duration_months": float(avg_duration),
            "is_noisy": avg_duration < 4,
        }

    def _check_label_distribution(self, label_counts: Dict[int, int]) -> bool:
        """
        Check if a label distribution meets minimum count constraints.
        Returns True if valid, False if any regime is under-represented.
        """
        for regime_idx, min_count in MIN_LABEL_COUNTS.items():
            actual = label_counts.get(regime_idx, 0)
            if actual < min_count:
                return False
        return True

    def _run_optimization(self, feature_matrix, classifier_class, n_iterations) -> Dict:
        """
        Iterative parameter optimization with minimum label constraints.
        Rejects parameter sets that eliminate any regime class.
        """
        logger.info(f"  Running parameter optimization ({n_iterations} iterations)...")

        best_score = 0
        best_params = self._param_optimizer.get_params()

        # Expanded search space including new signals
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
        }

        rng = np.random.RandomState(42)
        iteration_results = []
        skipped = 0

        for i in range(min(n_iterations, 40)):
            trial_params = dict(best_params)
            for param, values in search_space.items():
                trial_params[param] = rng.choice(values)

            try:
                model = classifier_class(n_regimes=N_REGIMES)
                model._label_params = trial_params
                model.fit(feature_matrix)

                # CHECK LABEL DISTRIBUTION - reject if any regime eliminated
                features_only = feature_matrix.drop(columns=["recession"], errors="ignore")
                X_unscaled = features_only.ffill().bfill()
                X_unscaled = X_unscaled[X_unscaled.notna().all(axis=1)]
                target = feature_matrix["recession"].reindex(X_unscaled.index).dropna()
                common = X_unscaled.index.intersection(target.index)
                test_labels = model._create_5class_labels(
                    X_unscaled.loc[common], target.loc[common]
                )
                label_counts = test_labels.value_counts().to_dict()

                if not self._check_label_distribution(label_counts):
                    skipped += 1
                    logger.info(f"    Iter {i}: SKIPPED (label constraint violated: {label_counts})")
                    continue

                # Validate
                regime_proba = model.predict_proba(features_only)
                regime_idx = regime_proba.values.argmax(axis=1)
                preds = pd.Series([REGIME_LABELS[j] for j in regime_idx], index=regime_proba.index)

                gt = self._validate_ground_truth(preds, feature_matrix)
                ret = self._validate_returns(preds, feature_matrix)
                score = 0.6 * gt["accuracy"] + 0.4 * ret["overall_alignment"]

                iteration_results.append({
                    "iteration": i,
                    "score": round(float(score), 4),
                    "gt_accuracy": round(float(gt["accuracy"]), 4),
                    "return_alignment": round(float(ret["overall_alignment"]), 4),
                    "label_dist": {str(k): int(v) for k, v in label_counts.items()},
                })

                if score > best_score:
                    best_score = score
                    best_params = dict(trial_params)
                    logger.info(
                        f"    Iter {i}: score={score:.3f} (GT={gt['accuracy']:.1%}, "
                        f"Ret={ret['overall_alignment']:.1%}) labels={label_counts} ** NEW BEST **"
                    )

            except Exception as e:
                logger.warning(f"    Iter {i} failed: {e}")

        # Persist best
        self._param_optimizer.update_best(best_params, best_score)

        logger.info(f"  Optimization complete. Best score: {best_score:.3f}, skipped {skipped} constraint violations")
        logger.info(f"  Best params saved to disk: {PARAMS_FILE}")

        return {
            "best_score": float(best_score),
            "best_params": {k: float(v) if isinstance(v, (np.integer, np.floating)) else v for k, v in best_params.items()},
            "iterations_run": len(iteration_results),
            "skipped_constraint": skipped,
            "top_5": sorted(iteration_results, key=lambda x: -x["score"])[:5],
        }
