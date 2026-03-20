"""
Ensemble Economic Regime Classifier
====================================
Three-model ensemble:
  1. HMM (hmmlearn) -- captures sequential regime transitions
  2. GMM (sklearn) -- captures distributional clustering
  3. XGBoost -- supervised classifier trained on NBER recession labels

All models are ML-driven; no hard-coded rules.
Outputs: regime probabilities, transition matrix, confidence scores.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from imblearn.over_sampling import SMOTE
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config.settings import (
    FORECAST_HORIZONS,
    MODEL_DIR,
    N_REGIMES,
    REGIME_LABELS,
)

logger = logging.getLogger(__name__)


class EnsembleRegimeClassifier:
    """
    5-state economic regime classifier using HMM + GMM + XGBoost ensemble.
    """

    def __init__(self, n_regimes: int = N_REGIMES, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state

        # Models
        self.hmm: Optional[GaussianHMM] = None
        self.gmm: Optional[GaussianMixture] = None
        self.xgb: Optional[XGBClassifier] = None
        self.scaler = StandardScaler()

        # Ensemble weights (calibrated during training)
        self.ensemble_weights = {"hmm": 0.35, "gmm": 0.25, "xgb": 0.40}

        # State alignment mapping
        self._hmm_state_map: Optional[Dict[int, int]] = None
        self._gmm_state_map: Optional[Dict[int, int]] = None

        # Transition matrix from HMM
        self.transition_matrix: Optional[np.ndarray] = None

        # Feature columns used during training
        self.feature_cols: List[str] = []
        self.is_fitted = False
        self._col_medians: Optional[pd.Series] = None

        # Label creation parameters (injectable by optimizer)
        self._label_params: Optional[Dict] = None

    # -------------------------------------------------------------------
    #  Training
    # -------------------------------------------------------------------

    def fit(self, features: pd.DataFrame) -> "EnsembleRegimeClassifier":
        """
        Train the full ensemble on the feature matrix.
        `features` must contain a 'recession' column (NBER labels).
        """
        logger.info("Training ensemble regime classifier...")

        if "recession" not in features.columns:
            raise ValueError("Feature matrix must contain 'recession' column (NBER labels)")

        target = features["recession"].copy()
        X_raw = features.drop(columns=["recession"]).copy()

        self.feature_cols = list(X_raw.columns)

        # Handle NaNs gracefully for pre-1990 data where VIX/CAPE/credit
        # spreads don't exist.  Strategy:
        #   1. ffill + bfill propagates values where series exists
        #   2. Keep rows with at least 60% of columns populated
        #   3. Fill remaining NaNs with column medians so StandardScaler
        #      and all downstream models receive a complete matrix
        #   4. If a column is entirely NaN (series doesn't exist yet in
        #      this training window), fill with 0.0 — after scaling this
        #      represents the neutral/mean value
        X_raw = X_raw.ffill().bfill()
        n_feature_cols = X_raw.shape[1]
        valid_mask = (
            (X_raw.notna().sum(axis=1) >= int(n_feature_cols * 0.6))
            & target.notna()
        )
        X_raw = X_raw[valid_mask]
        target = target[valid_mask]

        # Impute any remaining NaNs with column medians
        col_medians = X_raw.median()
        # For columns that are entirely NaN (series not yet available),
        # median is NaN — replace with 0.0 (neutral after scaling)
        still_nan = col_medians.isna()
        if still_nan.any():
            nan_cols = list(col_medians[still_nan].index)
            logger.info(f"Columns with no data in training window (filled with 0): {nan_cols}")
            col_medians = col_medians.fillna(0.0)
        X_raw = X_raw.fillna(col_medians)
        self._col_medians = col_medians  # save for predict-time imputation

        # Final safety check: ensure no NaNs remain
        remaining_nans = X_raw.isna().sum().sum()
        if remaining_nans > 0:
            logger.warning(f"  {remaining_nans} NaNs still remain after imputation — filling with 0")
            X_raw = X_raw.fillna(0.0)

        logger.info(f"Training data: {len(X_raw)} rows after NaN handling "
                    f"({n_feature_cols} features, "
                    f"{(~valid_mask).sum()} rows dropped)")

        X_scaled = self.scaler.fit_transform(X_raw)
        # StandardScaler produces NaN for zero-variance columns (e.g. a column
        # filled entirely with 0.0 because the series didn't exist yet).
        # Replace those NaNs with 0.0 (the scaled mean).
        if np.isnan(X_scaled).any():
            nan_count = np.isnan(X_scaled).sum()
            logger.info(f"Replacing {nan_count} NaN values from zero-variance columns after scaling")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        X_df = pd.DataFrame(X_scaled, index=X_raw.index, columns=self.feature_cols)

        # Step 1: Train HMM (unsupervised, sequential)
        self._fit_hmm(X_scaled)

        # Step 2: Train GMM (unsupervised, distributional)
        self._fit_gmm(X_scaled)

        # Step 3: Create 5-class supervised labels
        # IMPORTANT: Use UNSCALED features so thresholds work on real values
        X_unscaled = pd.DataFrame(X_raw.values, index=X_raw.index, columns=self.feature_cols)
        labels_5class = self._create_5class_labels(X_unscaled, target)

        # Step 4: Train XGBoost (supervised)
        self._fit_xgboost(X_scaled, labels_5class)

        # Step 5: Align HMM/GMM states to canonical labels
        self._align_states(X_scaled, labels_5class)

        # Step 6: Calibrate ensemble weights via time-series CV
        self._calibrate_weights(X_scaled, labels_5class)

        self.is_fitted = True
        logger.info("Ensemble training complete")
        return self

    def _fit_hmm(self, X: np.ndarray):
        """Train Gaussian HMM with 5 hidden states."""
        logger.info(f"Training HMM with {self.n_regimes} states...")
        self.hmm = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=self.random_state,
            tol=0.001,
        )
        self.hmm.fit(X)
        self.transition_matrix = self.hmm.transmat_
        logger.info(f"HMM converged: {self.hmm.monitor_.converged}")

    def _fit_gmm(self, X: np.ndarray):
        """Train Gaussian Mixture Model with 5 components."""
        logger.info(f"Training GMM with {self.n_regimes} components...")
        self.gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            n_init=5,
            max_iter=200,
            random_state=self.random_state,
        )
        self.gmm.fit(X)

    def _create_5class_labels(self, X_df: pd.DataFrame, recession: pd.Series) -> pd.Series:
        """
        Derive 5-class regime labels from NBER binary + feature-based logic.
        Parameters are read from self._label_params for optimization.

        States:
          0 = Expansion:  Not in recession, positive momentum
          1 = Slowdown:   Not in recession, but leading indicators deteriorating
          2 = Contraction: In NBER recession
          3 = Recovery:    Exiting recession or early cycle rebound
          4 = Crisis:      Extreme stress (VIX spike, credit blowout, deep recession)
        """
        # Read parameters (from optimizer if available, else defaults)
        p = self._label_params if self._label_params else {}
        yc_thresh = p.get("yield_curve_threshold", -0.2)
        indpro_thresh = p.get("indpro_threshold", 0.5)
        sent_thresh = p.get("sentiment_threshold", -8.0)
        payems_thresh = p.get("payems_threshold", -0.05)
        sp500_thresh = p.get("sp500_threshold", -5.0)
        credit_q = p.get("credit_spread_quantile", 0.75)
        slow_count = int(p.get("slowdown_signal_count", 3))
        crisis_vix_q = p.get("crisis_vix_quantile", 0.95)
        crisis_spread_q = p.get("crisis_spread_quantile", 0.95)
        recovery_mo = int(p.get("recovery_months", 6))
        # New parameters for additional slowdown signals
        unrate_accel_thresh = p.get("unrate_accel_threshold", 0.3)
        sp500_mom3m_thresh = p.get("sp500_mom3m_threshold", -5.0)
        credit_accel_thresh = p.get("credit_accel_threshold", 0.3)
        umcsent_level_thresh = p.get("umcsent_level_threshold", 65.0)

        logger.info(f"  Label params: yc={yc_thresh}, indpro={indpro_thresh}, "
                    f"sent={sent_thresh}, payems={payems_thresh}, "
                    f"sp500={sp500_thresh}, signals>={slow_count}, "
                    f"unrate_accel={unrate_accel_thresh}, sp500_3m={sp500_mom3m_thresh}")

        labels = pd.Series(0, index=X_df.index, name="regime")

        # Base: NBER recession -> Contraction
        labels[recession == 1] = 2

        # Identify crisis periods: recession + extreme financial stress
        crisis_mask = recession == 1
        if "vix" in X_df.columns:
            vix_extreme = X_df["vix"] > X_df["vix"].quantile(crisis_vix_q)
            crisis_mask = crisis_mask & vix_extreme
        if "credit_spread" in X_df.columns:
            spread_extreme = X_df["credit_spread"] > X_df["credit_spread"].quantile(crisis_spread_q)
            crisis_mask = crisis_mask | ((recession == 1) & spread_extreme)
        labels[crisis_mask] = 4

        # Recovery: first N months after recession ends
        rec_diff = recession.diff()
        recession_ends = rec_diff[rec_diff == -1].index
        for end_date in recession_ends:
            recovery_window = (X_df.index >= end_date) & (
                X_df.index < end_date + pd.DateOffset(months=recovery_mo)
            )
            labels[recovery_window] = 3

        # Slowdown: non-recession but leading indicators deteriorating
        non_recession = recession == 0
        slow_signals = pd.Series(0.0, index=X_df.index)

        # Yield curve: only count if deeply inverted
        if "spread_10y2y" in X_df.columns:
            slow_signals += (X_df["spread_10y2y"] < yc_thresh).astype(int)
        # Industrial production: must be actually declining
        if "indpro_yoy" in X_df.columns:
            slow_signals += (X_df["indpro_yoy"] < indpro_thresh).astype(int)
        # Consumer sentiment: sharp deterioration
        if "umcsent_chg" in X_df.columns:
            slow_signals += (X_df["umcsent_chg"] < sent_thresh).astype(int)
        # Payrolls: actual job losses
        if "payems_mom" in X_df.columns:
            slow_signals += (X_df["payems_mom"] < payems_thresh).astype(int)
        # S&P 500: negative YoY return
        if "sp500_yoy" in X_df.columns:
            slow_signals += (X_df["sp500_yoy"] < sp500_thresh).astype(int)
        # Credit spreads widening
        if "credit_spread" in X_df.columns:
            slow_signals += (X_df["credit_spread"] > X_df["credit_spread"].quantile(credit_q)).astype(int)

        # NEW SIGNAL 1: Unemployment rate accelerating (rising even if level is low)
        if "unrate_chg_3m" in X_df.columns:
            slow_signals += (X_df["unrate_chg_3m"] > unrate_accel_thresh).astype(int)

        # NEW SIGNAL 2: S&P 500 3-month momentum negative (captures recent declines)
        if "sp500_mom_3m" in X_df.columns:
            slow_signals += (X_df["sp500_mom_3m"] < sp500_mom3m_thresh).astype(int)

        # NEW SIGNAL 3: Credit spread acceleration (spreads widening rapidly)
        if "credit_spread_chg_3m" in X_df.columns:
            slow_signals += (X_df["credit_spread_chg_3m"] > credit_accel_thresh).astype(int)

        # NEW SIGNAL 4: Consumer sentiment below historical average
        if "umcsent" in X_df.columns:
            slow_signals += (X_df["umcsent"] < umcsent_level_thresh).astype(int)

        # Require N+ signals (parameterized)
        slowdown_mask = non_recession & (slow_signals >= slow_count)
        # Don't overwrite recovery or crisis
        slowdown_mask = slowdown_mask & (labels == 0)
        labels[slowdown_mask] = 1

        # ── Forward-shifted Slowdown: label pre-recession months ──
        # If enabled, the N months BEFORE each recession start are labeled
        # Slowdown.  This teaches the model to associate leading indicator
        # patterns with an *upcoming* contraction rather than only detecting
        # slowdowns coincidentally.  Only overwrites Expansion (label 0).
        pre_rec_lead = int(p.get("pre_recession_lead_months", 0))
        if pre_rec_lead > 0:
            rec_diff_starts = recession.diff()
            recession_starts = rec_diff_starts[rec_diff_starts == 1].index
            pre_rec_count = 0
            for start_date in recession_starts:
                pre_window = (
                    (X_df.index >= start_date - pd.DateOffset(months=pre_rec_lead))
                    & (X_df.index < start_date)
                )
                # Only overwrite Expansion labels — don't clobber Recovery
                # from a prior recession or existing signal-based Slowdown
                overwrite_mask = pre_window & (labels == 0)
                pre_rec_count += overwrite_mask.sum()
                labels[overwrite_mask] = 1
            if pre_rec_count > 0:
                logger.info(f"  Forward-shifted: {pre_rec_count} months re-labeled "
                            f"Expansion->Slowdown (lead={pre_rec_lead}mo)")

        logger.info(f"Label distribution:\n{labels.value_counts().sort_index()}")
        return labels

    def _fit_xgboost(self, X: np.ndarray, labels: pd.Series):
        """Train XGBoost classifier with SMOTE for class imbalance."""
        logger.info("Training XGBoost classifier...")

        y = labels.values

        # SMOTE to handle class imbalance
        try:
            smote = SMOTE(random_state=self.random_state, k_neighbors=3)
            X_res, y_res = smote.fit_resample(X, y)
            logger.info(f"SMOTE: {len(X)} -> {len(X_res)} samples")
        except Exception as e:
            logger.warning(f"SMOTE failed ({e}), using class weights instead")
            X_res, y_res = X, y

        # Compute class weights
        classes, counts = np.unique(y, return_counts=True)
        weight_map = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        sample_weights = np.array([weight_map[yi] for yi in y_res])

        self.xgb = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=self.n_regimes,
            random_state=self.random_state,
            eval_metric="mlogloss",
        )
        self.xgb.fit(X_res, y_res, sample_weight=sample_weights)

    def _align_states(self, X: np.ndarray, reference_labels: pd.Series):
        """Map HMM/GMM unsupervised states to canonical regime labels."""
        ref = reference_labels.values

        hmm_states = self.hmm.predict(X)
        self._hmm_state_map = self._compute_state_mapping(hmm_states, ref)

        gmm_states = self.gmm.predict(X)
        self._gmm_state_map = self._compute_state_mapping(gmm_states, ref)

    def _compute_state_mapping(self, predicted: np.ndarray, reference: np.ndarray) -> Dict[int, int]:
        """Find the best mapping from unsupervised states to reference labels."""
        from scipy.optimize import linear_sum_assignment

        cost = np.zeros((self.n_regimes, self.n_regimes))
        for pred_s in range(self.n_regimes):
            for ref_s in range(self.n_regimes):
                cost[pred_s, ref_s] = -np.sum((predicted == pred_s) & (reference == ref_s))

        row_ind, col_ind = linear_sum_assignment(cost)
        return dict(zip(row_ind, col_ind))

    def _calibrate_weights(self, X: np.ndarray, labels: pd.Series):
        """Calibrate ensemble weights using time-series cross-validation."""
        logger.info("Calibrating ensemble weights via TSCV...")
        tscv = TimeSeriesSplit(n_splits=5)
        y = labels.values

        model_accuracies = {"hmm": [], "gmm": [], "xgb": []}

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_te = y[test_idx]

            hmm_pred = self.hmm.predict(X_te)
            hmm_mapped = np.array([self._hmm_state_map.get(s, 0) for s in hmm_pred])
            model_accuracies["hmm"].append(np.mean(hmm_mapped == y_te))

            gmm_pred = self.gmm.predict(X_te)
            gmm_mapped = np.array([self._gmm_state_map.get(s, 0) for s in gmm_pred])
            model_accuracies["gmm"].append(np.mean(gmm_mapped == y_te))

            xgb_pred = self.xgb.predict(X_te)
            model_accuracies["xgb"].append(np.mean(xgb_pred == y_te))

        mean_acc = {k: np.mean(v) for k, v in model_accuracies.items()}
        total = sum(mean_acc.values())
        self.ensemble_weights = {k: v / total for k, v in mean_acc.items()}
        logger.info(f"Calibrated ensemble weights: {self.ensemble_weights}")
        logger.info(f"Model accuracies: {mean_acc}")

    # -------------------------------------------------------------------
    #  Prediction
    # -------------------------------------------------------------------

    def predict_proba(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities for each row of X_raw.
        Returns DataFrame with columns = regime labels, index = dates.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        X_clean = X_raw[self.feature_cols].ffill().bfill()
        # Apply same median imputation used during training
        if hasattr(self, '_col_medians') and self._col_medians is not None:
            X_clean = X_clean.fillna(self._col_medians)
        X_scaled = self.scaler.transform(X_clean)

        hmm_proba_raw = self.hmm.predict_proba(X_scaled)
        hmm_proba = self._remap_probabilities(hmm_proba_raw, self._hmm_state_map)

        gmm_proba_raw = self.gmm.predict_proba(X_scaled)
        gmm_proba = self._remap_probabilities(gmm_proba_raw, self._gmm_state_map)

        xgb_proba = self.xgb.predict_proba(X_scaled)

        for arr_name, arr in [("hmm", hmm_proba), ("gmm", gmm_proba), ("xgb", xgb_proba)]:
            if arr.shape[1] != self.n_regimes:
                logger.warning(f"{arr_name} has {arr.shape[1]} columns, expected {self.n_regimes}")

        # Cap HMM and GMM confidence to prevent stickiness from overwhelming ensemble
        def cap_confidence(proba, max_prob=0.80):
            """Redistribute excess probability from dominant regime to others."""
            capped = proba.copy()
            for i in range(capped.shape[0]):
                max_idx = capped[i].argmax()
                if capped[i, max_idx] > max_prob:
                    excess = capped[i, max_idx] - max_prob
                    capped[i, max_idx] = max_prob
                    other_count = capped.shape[1] - 1
                    for j in range(capped.shape[1]):
                        if j != max_idx:
                            capped[i, j] += excess / other_count
            return capped

        hmm_proba = cap_confidence(hmm_proba, max_prob=0.80)
        gmm_proba = cap_confidence(gmm_proba, max_prob=0.85)





        w = self.ensemble_weights
        ensemble_proba = (
            w["hmm"] * hmm_proba + w["gmm"] * gmm_proba + w["xgb"] * xgb_proba
        )
        ensemble_proba = ensemble_proba / ensemble_proba.sum(axis=1, keepdims=True)

        return pd.DataFrame(
            ensemble_proba,
            index=X_clean.index,
            columns=[REGIME_LABELS[i] for i in range(self.n_regimes)],
        )

    def predict(self, X_raw: pd.DataFrame) -> pd.Series:
        """Return the most likely regime for each observation."""
        proba = self.predict_proba(X_raw)
        regime_idx = proba.values.argmax(axis=1)
        return pd.Series(
            [REGIME_LABELS[i] for i in regime_idx],
            index=proba.index,
            name="regime",
        )

    def _remap_probabilities(self, proba: np.ndarray, state_map: Dict[int, int]) -> np.ndarray:
        """Remap unsupervised model probabilities to canonical regime order."""
        remapped = np.zeros_like(proba)
        for orig_state, canonical_state in state_map.items():
            if orig_state < proba.shape[1] and canonical_state < remapped.shape[1]:
                remapped[:, canonical_state] = proba[:, orig_state]
        row_sums = remapped.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        remapped = remapped / row_sums
        return remapped

    # -------------------------------------------------------------------
    #  Forward Forecasting
    # -------------------------------------------------------------------

    def forecast_regimes(
        self, current_proba: np.ndarray, horizons: List[int] = None
    ) -> Dict[int, np.ndarray]:
        """
        Forecast regime probabilities at future horizons using HMM transition matrix.
        """
        horizons = horizons or FORECAST_HORIZONS
        T = self.transition_matrix

        T_remapped = np.zeros_like(T)
        for i_orig, i_canon in self._hmm_state_map.items():
            for j_orig, j_canon in self._hmm_state_map.items():
                if i_orig < T.shape[0] and j_orig < T.shape[1]:
                    T_remapped[i_canon, j_canon] = T[i_orig, j_orig]

        row_sums = T_remapped.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        T_remapped = T_remapped / row_sums

        forecasts = {}
        for h in horizons:
            T_h = np.linalg.matrix_power(T_remapped, h)
            forecast_proba = current_proba @ T_h
            forecast_proba = forecast_proba / forecast_proba.sum()
            forecasts[h] = forecast_proba

        return forecasts

    # -------------------------------------------------------------------
    #  Confidence & Disagreement
    # -------------------------------------------------------------------

    def compute_confidence(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-observation confidence metrics.
        """
        X_clean = X_raw[self.feature_cols].ffill().bfill()
        # Apply same median imputation used during training
        if hasattr(self, '_col_medians') and self._col_medians is not None:
            X_clean = X_clean.fillna(self._col_medians)
        X_scaled = self.scaler.transform(X_clean)

        hmm_top = np.array([self._hmm_state_map.get(s, 0) for s in self.hmm.predict(X_scaled)])
        gmm_top = np.array([self._gmm_state_map.get(s, 0) for s in self.gmm.predict(X_scaled)])
        xgb_top = self.xgb.predict(X_scaled)

        proba = self.predict_proba(X_raw)
        max_prob = proba.max(axis=1)
        entropy = -(proba * np.log(proba + 1e-10)).sum(axis=1)

        agreement = ((hmm_top == xgb_top).astype(int) +
                      (gmm_top == xgb_top).astype(int) +
                      (hmm_top == gmm_top).astype(int))
        agreement_score = agreement / 3.0

        return pd.DataFrame({
            "max_prob": max_prob,
            "entropy": entropy,
            "model_agreement": agreement_score,
            "hmm_regime": [REGIME_LABELS.get(s, "Unknown") for s in hmm_top],
            "gmm_regime": [REGIME_LABELS.get(s, "Unknown") for s in gmm_top],
            "xgb_regime": [REGIME_LABELS.get(s, "Unknown") for s in xgb_top],
        }, index=proba.index)

    # -------------------------------------------------------------------
    #  Feature Importance
    # -------------------------------------------------------------------

    def feature_importance(self) -> pd.Series:
        """Return XGBoost feature importances."""
        if self.xgb is None:
            return pd.Series(dtype=float)
        return pd.Series(
            self.xgb.feature_importances_,
            index=self.feature_cols,
            name="importance",
        ).sort_values(ascending=False)

    # -------------------------------------------------------------------
    #  Persistence
    # -------------------------------------------------------------------

    def save(self, path: Optional[Path] = None):
        """Save the trained ensemble to disk."""
        path = path or MODEL_DIR / "ensemble_regime_classifier.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "EnsembleRegimeClassifier":
        """Load a trained ensemble from disk."""
        path = path or MODEL_DIR / "ensemble_regime_classifier.pkl"
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model
