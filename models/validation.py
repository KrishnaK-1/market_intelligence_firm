"""
Regime Validation & Reinforcement Feedback System
===================================================
Three-stage pipeline:
  1. VALIDATE: Cross-check regime classifications against known economic history
     and actual market returns to identify misclassifications
  2. SCORE: Compute a regime accuracy score by comparing predicted regimes
     against what market returns actually looked like in each period
  3. FEEDBACK: Auto-adjust label creation thresholds and retrain the model
     based on validation results

The key insight: if the model calls "Slowdown" but equity returns were +20% YoY
and GDP was growing 2.8%, that's a misclassification. The system detects these
and adjusts the sensitivity parameters that drive label creation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    N_REGIMES,
    REGIME_LABELS,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Known Economic History (ground truth for validation)
# ═══════════════════════════════════════════════════════════════════════

# Post-war NBER recessions (peak -> trough)
NBER_RECESSIONS = [
    ("1953-07", "1954-05"),  # Korean War recession
    ("1957-08", "1958-04"),  # Eisenhower recession
    ("1960-04", "1961-02"),  # Rolling adjustment
    ("1969-12", "1970-11"),  # Nixon recession
    ("1973-11", "1975-03"),  # Oil crisis
    ("1980-01", "1980-07"),  # Volcker shock 1
    ("1981-07", "1982-11"),  # Volcker shock 2
    ("1990-07", "1991-03"),  # S&L crisis
    ("2001-03", "2001-11"),  # Dot-com bust
    ("2007-12", "2009-06"),  # Great Financial Crisis
    ("2020-02", "2020-04"),  # COVID
]

# Known crisis periods (subset of recessions with extreme financial stress)
CRISIS_PERIODS = [
    ("1973-11", "1974-12"),  # Oil embargo + market crash
    ("2008-09", "2009-03"),  # Lehman/AIG/banking collapse
    ("2020-02", "2020-04"),  # COVID crash
]

# Known strong expansion years (GDP > 2.5%, strong equity returns)
STRONG_EXPANSIONS = [
    ("1954-06", "1957-07"),  # Post-Korean War boom
    ("1961-03", "1969-11"),  # 1960s expansion
    ("1983-01", "1990-06"),  # Reagan expansion
    ("1991-04", "2001-02"),  # 1990s bull market
    ("2003-07", "2007-11"),  # Mid-2000s expansion
    ("2010-07", "2019-12"),  # Post-GFC expansion
    ("2020-06", "2024-12"),  # Post-COVID expansion (strong GDP, strong markets)
]

# Expected regime return profiles (annualized monthly return ranges)
# If actual returns fall outside these ranges, the regime call is suspect
REGIME_RETURN_PROFILES = {
    "Expansion": {"sp500_yoy_min": 0, "sp500_yoy_typical": 12, "gdp_growth_min": 1.5},
    "Slowdown":  {"sp500_yoy_min": -15, "sp500_yoy_typical": 3, "gdp_growth_min": 0},
    "Contraction": {"sp500_yoy_min": -40, "sp500_yoy_typical": -10, "gdp_growth_min": -5},
    "Recovery":  {"sp500_yoy_min": 5, "sp500_yoy_typical": 20, "gdp_growth_min": 1},
    "Crisis":    {"sp500_yoy_min": -50, "sp500_yoy_typical": -25, "gdp_growth_min": -8},
}


@dataclass
class ValidationResult:
    """Result of validating a single period's regime classification."""
    date: str
    predicted_regime: str
    expected_regime: str
    is_correct: bool
    reason: str
    sp500_yoy: Optional[float] = None
    indpro_yoy: Optional[float] = None
    confidence: float = 0.0


@dataclass
class CalibrationAdjustment:
    """Recommended adjustment to a label creation parameter."""
    parameter: str
    current_value: float
    recommended_value: float
    reason: str
    impact_periods: int  # how many months this would affect


class RegimeValidator:
    """
    Validates regime classifications against economic history and market returns.
    Produces a validation report and calibration recommendations.
    """

    def __init__(self):
        self.validation_results: List[ValidationResult] = []
        self.calibration_adjustments: List[CalibrationAdjustment] = []
        self.accuracy_by_regime: Dict[str, float] = {}
        self.confusion_matrix: Optional[pd.DataFrame] = None

    def validate(
        self,
        regime_predictions: pd.Series,
        feature_matrix: pd.DataFrame,
    ) -> Dict:
        """
        Run full validation pipeline.

        Args:
            regime_predictions: Series with DatetimeIndex, values are regime labels
            feature_matrix: Features including sp500_yoy, indpro_yoy, recession, etc.
        """
        logger.info("=" * 60)
        logger.info("REGIME VALIDATION & CALIBRATION")
        logger.info("=" * 60)

        self.validation_results = []

        # Stage 1: Validate against NBER recession dates
        self._validate_nber_recessions(regime_predictions, feature_matrix)

        # Stage 2: Validate against known expansion periods
        self._validate_expansions(regime_predictions, feature_matrix)

        # Stage 3: Validate against crisis periods
        self._validate_crises(regime_predictions, feature_matrix)

        # Stage 4: Validate using market return profiles
        self._validate_return_profiles(regime_predictions, feature_matrix)

        # Stage 5: Build confusion matrix and accuracy metrics
        self._compute_accuracy_metrics(regime_predictions, feature_matrix)

        # Stage 6: Generate calibration recommendations
        self._generate_calibration_recommendations(regime_predictions, feature_matrix)

        # Compile report
        report = self._compile_report()
        logger.info(f"Validation complete: {report['overall_accuracy']:.1%} accuracy")
        return report

    # ───────────────────────────────────────────────────────────────
    #  Stage 1: NBER Recession Validation
    # ───────────────────────────────────────────────────────────────

    def _validate_nber_recessions(self, preds: pd.Series, features: pd.DataFrame):
        """Check that NBER recession periods are classified as Contraction or Crisis."""
        logger.info("Validating NBER recession periods...")
        for start, end in NBER_RECESSIONS:
            try:
                mask = (preds.index >= pd.Timestamp(start)) & (preds.index <= pd.Timestamp(end))
                period_preds = preds[mask]
                if len(period_preds) == 0:
                    continue

                correct_calls = period_preds.isin(["Contraction", "Crisis"]).sum()
                total = len(period_preds)
                accuracy = correct_calls / total

                for date, pred in period_preds.items():
                    is_correct = pred in ["Contraction", "Crisis"]
                    sp500 = features.loc[date, "sp500_yoy"] if "sp500_yoy" in features.columns and date in features.index else None
                    self.validation_results.append(ValidationResult(
                        date=date.strftime("%Y-%m"),
                        predicted_regime=pred,
                        expected_regime="Contraction",
                        is_correct=is_correct,
                        reason=f"NBER recession {start} to {end}",
                        sp500_yoy=sp500,
                    ))

                if accuracy < 0.7:
                    logger.warning(f"  Recession {start}->{end}: only {accuracy:.0%} correct ({correct_calls}/{total})")
                else:
                    logger.info(f"  Recession {start}->{end}: {accuracy:.0%} correct")
            except Exception as e:
                logger.warning(f"  Error validating {start}-{end}: {e}")

    # ───────────────────────────────────────────────────────────────
    #  Stage 2: Expansion Validation
    # ───────────────────────────────────────────────────────────────

    def _validate_expansions(self, preds: pd.Series, features: pd.DataFrame):
        """Check that known expansion periods are classified as Expansion or Recovery."""
        logger.info("Validating expansion periods...")
        for start, end in STRONG_EXPANSIONS:
            try:
                mask = (preds.index >= pd.Timestamp(start)) & (preds.index <= pd.Timestamp(end))
                period_preds = preds[mask]
                if len(period_preds) == 0:
                    continue

                correct_calls = period_preds.isin(["Expansion", "Recovery"]).sum()
                total = len(period_preds)
                accuracy = correct_calls / total

                for date, pred in period_preds.items():
                    is_correct = pred in ["Expansion", "Recovery"]
                    sp500 = features.loc[date, "sp500_yoy"] if "sp500_yoy" in features.columns and date in features.index else None
                    self.validation_results.append(ValidationResult(
                        date=date.strftime("%Y-%m"),
                        predicted_regime=pred,
                        expected_regime="Expansion",
                        is_correct=is_correct,
                        reason=f"Known expansion {start} to {end}",
                        sp500_yoy=sp500,
                    ))

                if accuracy < 0.7:
                    logger.warning(f"  Expansion {start}->{end}: only {accuracy:.0%} correct ({correct_calls}/{total})")
                else:
                    logger.info(f"  Expansion {start}->{end}: {accuracy:.0%} correct")
            except Exception as e:
                logger.warning(f"  Error validating {start}-{end}: {e}")

    # ───────────────────────────────────────────────────────────────
    #  Stage 3: Crisis Validation
    # ───────────────────────────────────────────────────────────────

    def _validate_crises(self, preds: pd.Series, features: pd.DataFrame):
        """Check that crisis periods are classified as Crisis."""
        logger.info("Validating crisis periods...")
        for start, end in CRISIS_PERIODS:
            try:
                mask = (preds.index >= pd.Timestamp(start)) & (preds.index <= pd.Timestamp(end))
                period_preds = preds[mask]
                if len(period_preds) == 0:
                    continue

                correct_calls = (period_preds == "Crisis").sum()
                # Also accept Contraction for crisis periods
                acceptable = period_preds.isin(["Crisis", "Contraction"]).sum()
                total = len(period_preds)

                if correct_calls / total < 0.5:
                    logger.warning(f"  Crisis {start}->{end}: only {correct_calls}/{total} classified as Crisis")
            except Exception:
                pass

    # ───────────────────────────────────────────────────────────────
    #  Stage 4: Return Profile Validation
    # ───────────────────────────────────────────────────────────────

    def _validate_return_profiles(self, preds: pd.Series, features: pd.DataFrame):
        """
        Check if predicted regimes match what market returns actually did.
        If the model says 'Slowdown' but S&P was up 20% YoY, something's wrong.
        """
        logger.info("Validating against market return profiles...")

        if "sp500_yoy" not in features.columns:
            logger.warning("  No sp500_yoy in features, skipping return validation")
            return

        mismatches = 0
        total = 0

        for date in preds.index:
            if date not in features.index:
                continue
            pred = preds[date]
            sp500 = features.loc[date, "sp500_yoy"] if pd.notna(features.loc[date, "sp500_yoy"]) else None
            indpro = features.loc[date, "indpro_yoy"] if "indpro_yoy" in features.columns and pd.notna(features.loc[date, "indpro_yoy"]) else None

            if sp500 is None:
                continue
            total += 1

            # Detect clear mismatches
            is_mismatch = False
            expected = None

            if pred == "Slowdown" and sp500 > 15 and (indpro is None or indpro > 3):
                is_mismatch = True
                expected = "Expansion"
            elif pred == "Expansion" and sp500 < -15:
                is_mismatch = True
                expected = "Contraction"
            elif pred == "Contraction" and sp500 > 15 and (indpro is None or indpro > 2):
                is_mismatch = True
                expected = "Expansion"
            elif pred == "Recovery" and sp500 < -20:
                is_mismatch = True
                expected = "Contraction"
            elif pred == "Crisis" and sp500 > 10 and (indpro is None or indpro > 2):
                is_mismatch = True
                expected = "Expansion"

            if is_mismatch:
                mismatches += 1

        mismatch_rate = mismatches / total if total > 0 else 0
        logger.info(f"  Return profile mismatches: {mismatches}/{total} ({mismatch_rate:.1%})")

    # ───────────────────────────────────────────────────────────────
    #  Stage 5: Accuracy Metrics
    # ───────────────────────────────────────────────────────────────

    def _compute_accuracy_metrics(self, preds: pd.Series, features: pd.DataFrame):
        """Compute per-regime accuracy from validation results."""
        if not self.validation_results:
            return

        results_df = pd.DataFrame([
            {"predicted": r.predicted_regime, "expected": r.expected_regime, "correct": r.is_correct}
            for r in self.validation_results
        ])

        # Per-regime accuracy
        for regime in REGIME_LABELS.values():
            expected_mask = results_df["expected"] == regime
            if expected_mask.sum() > 0:
                acc = results_df.loc[expected_mask, "correct"].mean()
                self.accuracy_by_regime[regime] = acc

        # Confusion matrix
        regimes = list(REGIME_LABELS.values())
        matrix = pd.DataFrame(0, index=regimes, columns=regimes)
        for _, row in results_df.iterrows():
            if row["predicted"] in regimes and row["expected"] in regimes:
                matrix.loc[row["expected"], row["predicted"]] += 1
        self.confusion_matrix = matrix

        logger.info("Per-regime accuracy:")
        for regime, acc in self.accuracy_by_regime.items():
            logger.info(f"  {regime}: {acc:.1%}")

    # ───────────────────────────────────────────────────────────────
    #  Stage 6: Calibration Recommendations
    # ───────────────────────────────────────────────────────────────

    def _generate_calibration_recommendations(self, preds: pd.Series, features: pd.DataFrame):
        """
        Analyze misclassification patterns and recommend threshold adjustments.
        This is the 'reinforcement feedback' loop.
        """
        logger.info("Generating calibration recommendations...")
        self.calibration_adjustments = []

        if not self.validation_results:
            return

        results_df = pd.DataFrame([vars(r) for r in self.validation_results])

        # Pattern 1: Expansion periods misclassified as Slowdown
        exp_as_slow = results_df[
            (results_df["expected_regime"] == "Expansion") &
            (results_df["predicted_regime"] == "Slowdown")
        ]
        if len(exp_as_slow) > 10:
            # Check what features were triggering false slowdowns
            false_slow_dates = pd.DatetimeIndex([pd.Timestamp(d) for d in exp_as_slow["date"]])
            common_dates = false_slow_dates.intersection(features.index)

            if len(common_dates) > 0 and "spread_10y2y" in features.columns:
                avg_spread = features.loc[common_dates, "spread_10y2y"].mean()
                if avg_spread > -0.5:  # Not deeply inverted
                    self.calibration_adjustments.append(CalibrationAdjustment(
                        parameter="slowdown_yield_curve_threshold",
                        current_value=-0.2,
                        recommended_value=-0.5,
                        reason=f"{len(exp_as_slow)} expansion months misclassified as Slowdown. "
                               f"Avg yield spread was {avg_spread:.2f} — threshold too sensitive.",
                        impact_periods=len(exp_as_slow),
                    ))

            if len(common_dates) > 0 and "indpro_yoy" in features.columns:
                avg_indpro = features.loc[common_dates, "indpro_yoy"].mean()
                if avg_indpro > 1.0:
                    self.calibration_adjustments.append(CalibrationAdjustment(
                        parameter="slowdown_indpro_threshold",
                        current_value=0.5,
                        recommended_value=0.0,
                        reason=f"Industrial production averaged {avg_indpro:.1f}% during false "
                               f"slowdowns — threshold should require actual contraction.",
                        impact_periods=len(exp_as_slow),
                    ))

        # Pattern 2: Contraction not caught
        cont_missed = results_df[
            (results_df["expected_regime"] == "Contraction") &
            (~results_df["predicted_regime"].isin(["Contraction", "Crisis"]))
        ]
        if len(cont_missed) > 5:
            self.calibration_adjustments.append(CalibrationAdjustment(
                parameter="contraction_sensitivity",
                current_value=0,
                recommended_value=0,
                reason=f"{len(cont_missed)} recession months not classified as Contraction/Crisis. "
                       f"Model may need more weight on NBER labels in training.",
                impact_periods=len(cont_missed),
            ))

        # Pattern 3: Too much regime switching (noisy)
        regime_changes = (preds != preds.shift()).sum()
        months_per_regime = len(preds) / max(regime_changes, 1)
        if months_per_regime < 4:
            self.calibration_adjustments.append(CalibrationAdjustment(
                parameter="regime_stability",
                current_value=months_per_regime,
                recommended_value=6.0,
                reason=f"Average regime duration is only {months_per_regime:.1f} months. "
                       f"Too much switching — consider HMM smoothing or minimum duration filter.",
                impact_periods=regime_changes,
            ))

        for adj in self.calibration_adjustments:
            logger.info(f"  RECOMMENDATION: {adj.parameter}")
            logger.info(f"    Current: {adj.current_value} -> Recommended: {adj.recommended_value}")
            logger.info(f"    Reason: {adj.reason}")

    # ───────────────────────────────────────────────────────────────
    #  Report
    # ───────────────────────────────────────────────────────────────

    def _compile_report(self) -> Dict:
        """Compile validation results into a report dict."""
        total = len(self.validation_results)
        correct = sum(1 for r in self.validation_results if r.is_correct)
        overall_accuracy = correct / total if total > 0 else 0

        return {
            "overall_accuracy": overall_accuracy,
            "total_validated": total,
            "correct": correct,
            "accuracy_by_regime": self.accuracy_by_regime,
            "confusion_matrix": (
                self.confusion_matrix.to_dict() if self.confusion_matrix is not None else None
            ),
            "calibration_adjustments": [
                {
                    "parameter": a.parameter,
                    "current_value": a.current_value,
                    "recommended_value": a.recommended_value,
                    "reason": a.reason,
                    "impact_periods": a.impact_periods,
                }
                for a in self.calibration_adjustments
            ],
            "n_adjustments": len(self.calibration_adjustments),
            "misclassification_details": [
                {
                    "date": r.date,
                    "predicted": r.predicted_regime,
                    "expected": r.expected_regime,
                    "reason": r.reason,
                    "sp500_yoy": r.sp500_yoy,
                }
                for r in self.validation_results if not r.is_correct
            ][:50],  # Limit to 50 most recent
        }


# ═══════════════════════════════════════════════════════════════════════
#  Adaptive Label Calibrator
# ═══════════════════════════════════════════════════════════════════════


class AdaptiveLabelCalibrator:
    """
    Takes validation feedback and adjusts the label creation parameters.
    This is the 'learning from output' piece.

    The calibrator adjusts thresholds used in _create_5class_labels
    based on historical validation accuracy.
    """

    def __init__(self):
        # Current thresholds (matches what's in regime_classifier.py)
        self.params = {
            "slowdown_yield_curve_threshold": -0.2,
            "slowdown_indpro_threshold": 0.5,
            "slowdown_sentiment_threshold": -8,
            "slowdown_payems_threshold": -0.05,
            "slowdown_sp500_threshold": -5,
            "slowdown_credit_threshold_quantile": 0.75,
            "slowdown_min_signals": 3,
            "crisis_vix_quantile": 0.95,
            "crisis_credit_quantile": 0.95,
            "recovery_window_months": 6,
        }
        self._history: List[Dict] = []

    def calibrate(self, validation_report: Dict, features: pd.DataFrame) -> Dict:
        """
        Adjust parameters based on validation feedback.
        Returns updated parameter dict.
        """
        logger.info("Running adaptive calibration...")
        adjustments = validation_report.get("calibration_adjustments", [])

        for adj in adjustments:
            param = adj["parameter"]
            if param in self.params:
                old_val = self.params[param]
                new_val = adj["recommended_value"]
                # Apply gradual adjustment (don't jump all the way)
                # Use 50% step toward recommendation to avoid overcorrection
                blended = old_val + 0.5 * (new_val - old_val)
                self.params[param] = blended
                logger.info(f"  Adjusted {param}: {old_val} -> {blended:.4f} (target: {new_val})")

        # Additional data-driven adjustments
        self._auto_adjust_from_features(features, validation_report)

        self._history.append({
            "timestamp": datetime.now().isoformat(),
            "params": dict(self.params),
            "accuracy": validation_report.get("overall_accuracy", 0),
        })

        logger.info(f"Calibration complete. Current params: {self.params}")
        return dict(self.params)

    def _auto_adjust_from_features(self, features: pd.DataFrame, report: Dict):
        """
        Data-driven auto-adjustments based on feature distributions.
        Adapts thresholds to current market regime characteristics.
        """
        accuracy = report.get("overall_accuracy", 0)

        # If accuracy is already high, make smaller adjustments
        if accuracy > 0.85:
            logger.info("  Accuracy > 85%, minimal adjustments needed")
            return

        # If too many expansion periods are being called Slowdown
        regime_acc = report.get("accuracy_by_regime", {})
        expansion_acc = regime_acc.get("Expansion", 1.0)

        if expansion_acc < 0.7:
            # Tighten slowdown criteria
            logger.info(f"  Expansion accuracy low ({expansion_acc:.0%}), tightening slowdown criteria")
            self.params["slowdown_min_signals"] = min(4, self.params["slowdown_min_signals"] + 0.5)
            self.params["slowdown_yield_curve_threshold"] -= 0.1
            self.params["slowdown_indpro_threshold"] -= 0.2

        contraction_acc = regime_acc.get("Contraction", 1.0)
        if contraction_acc < 0.7:
            # Loosen contraction sensitivity
            logger.info(f"  Contraction accuracy low ({contraction_acc:.0%}), may need more NBER weight")

    def get_params(self) -> Dict:
        """Return current calibrated parameters."""
        return dict(self.params)

    def get_history(self) -> List[Dict]:
        """Return calibration history for tracking convergence."""
        return self._history
