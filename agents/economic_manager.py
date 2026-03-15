"""
Multi-Agent Architecture — Base Agent & Economic Regime Manager
================================================================
Hybrid architecture: Managers can read Economic signals but report
independently to the CIO aggregator.

Standard interface per agent:
  - get_signals()       → Dict of named signals with values
  - get_allocations()   → Dict of asset → weight
  - get_confidence()    → float 0-1
  - get_report()        → Dict with full analysis for dashboard/CIO
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import (
    FORECAST_HORIZONS,
    N_REGIMES,
    REGIME_ALLOCATIONS,
    REGIME_COLORS,
    REGIME_LABELS,
    REGIME_SUB_ALLOCATIONS,
)
from models.regime_classifier import EnsembleRegimeClassifier

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Base Agent Interface
# ═══════════════════════════════════════════════════════════════════════


class BaseManager(ABC):
    """Abstract base class for all investment managers."""

    name: str = "BaseManager"
    description: str = "Base investment manager"

    @abstractmethod
    def get_signals(self) -> Dict[str, Any]:
        """Return a dict of named signals with their current values."""
        ...

    @abstractmethod
    def get_allocations(self) -> Dict[str, float]:
        """Return target asset allocations (ticker → weight, sums ≈ 1.0)."""
        ...

    @abstractmethod
    def get_confidence(self) -> float:
        """Return a confidence score from 0.0 (no confidence) to 1.0 (max)."""
        ...

    @abstractmethod
    def get_report(self) -> Dict[str, Any]:
        """Return a full analysis report for the dashboard / CIO."""
        ...

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Refresh signals with new data."""
        ...


# ═══════════════════════════════════════════════════════════════════════
#  Economic Regime Manager
# ═══════════════════════════════════════════════════════════════════════


class EconomicRegimeManager(BaseManager):
    """
    ML-driven economic regime classification and asset allocation.
    Uses the ensemble HMM + GMM + XGBoost classifier.
    """

    name = "Economic Regime Manager"
    description = "Classifies macro environment and recommends regime-based allocations"

    def __init__(self):
        self.classifier: Optional[EnsembleRegimeClassifier] = None
        self.feature_matrix: Optional[pd.DataFrame] = None
        self.current_proba: Optional[np.ndarray] = None
        self.current_regime: Optional[str] = None
        self.regime_history: Optional[pd.DataFrame] = None
        self.forecasts: Optional[Dict[int, np.ndarray]] = None
        self.confidence_data: Optional[pd.DataFrame] = None
        self.transition_matrix: Optional[np.ndarray] = None
        self._last_update: Optional[datetime] = None

    def update(
        self,
        feature_matrix: pd.DataFrame,
        classifier: Optional[EnsembleRegimeClassifier] = None,
    ) -> None:
        """
        Run full regime analysis on updated data.

        Args:
            feature_matrix: DataFrame from build_feature_matrix()
            classifier: Pre-trained classifier (if None, trains a new one)
        """
        self.feature_matrix = feature_matrix

        # Train or use provided classifier
        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = EnsembleRegimeClassifier(n_regimes=N_REGIMES)
            self.classifier.fit(feature_matrix)

        # Predict regime probabilities for full history
        features_only = feature_matrix.drop(columns=["recession"], errors="ignore")
        self.regime_history = self.classifier.predict_proba(features_only)

        # Current regime
        self.current_proba = self.regime_history.iloc[-1].values
        self.current_regime = REGIME_LABELS[self.current_proba.argmax()]

        # Forward forecasts
        self.forecasts = self.classifier.forecast_regimes(self.current_proba)

        # Confidence metrics
        self.confidence_data = self.classifier.compute_confidence(features_only)

        # Transition matrix
        self.transition_matrix = self.classifier.transition_matrix

        self._last_update = datetime.now()
        logger.info(f"Economic Regime Manager updated. Current regime: {self.current_regime}")

    def get_signals(self) -> Dict[str, Any]:
        """Key regime signals."""
        if self.current_proba is None:
            return {}

        signals = {
            "current_regime": self.current_regime,
            "regime_probabilities": {
                REGIME_LABELS[i]: float(self.current_proba[i])
                for i in range(N_REGIMES)
            },
            "forecasts": {},
        }

        if self.forecasts:
            for h, proba in self.forecasts.items():
                signals["forecasts"][f"{h}m"] = {
                    REGIME_LABELS[i]: float(proba[i]) for i in range(N_REGIMES)
                }

        # Confidence
        if self.confidence_data is not None and len(self.confidence_data) > 0:
            latest = self.confidence_data.iloc[-1]
            signals["model_agreement"] = float(latest["model_agreement"])
            signals["prediction_entropy"] = float(latest["entropy"])
            signals["hmm_says"] = latest["hmm_regime"]
            signals["gmm_says"] = latest["gmm_regime"]
            signals["xgb_says"] = latest["xgb_regime"]

        return signals

    def get_allocations(self) -> Dict[str, float]:
        """
        Probability-weighted asset allocation across all regimes.
        Each regime has a template; final weights = Σ(regime_prob × regime_allocation).
        """
        if self.current_proba is None:
            return {}

        # Use forward forecasts to blend allocations (weight toward 1m and 3m)
        if self.forecasts:
            # Blend: 50% current, 30% 1m forecast, 20% 3m forecast
            blended_proba = (
                0.50 * self.current_proba
                + 0.30 * self.forecasts.get(1, self.current_proba)
                + 0.20 * self.forecasts.get(3, self.current_proba)
            )
            blended_proba = blended_proba / blended_proba.sum()
        else:
            blended_proba = self.current_proba

        # Compute probability-weighted sub-allocations
        allocations = {}
        for regime_idx in range(N_REGIMES):
            weight = blended_proba[regime_idx]
            sub_alloc = REGIME_SUB_ALLOCATIONS.get(regime_idx, {})
            for ticker, ticker_weight in sub_alloc.items():
                allocations[ticker] = allocations.get(ticker, 0.0) + weight * ticker_weight

        # Normalize to sum to 1
        total = sum(allocations.values())
        if total > 0:
            allocations = {k: v / total for k, v in allocations.items()}

        return allocations

    def get_asset_class_allocations(self) -> Dict[str, float]:
        """High-level asset class allocation (probability-weighted)."""
        if self.current_proba is None:
            return {}

        blended = self.current_proba
        if self.forecasts:
            blended = (
                0.50 * self.current_proba
                + 0.30 * self.forecasts.get(1, self.current_proba)
                + 0.20 * self.forecasts.get(3, self.current_proba)
            )
            blended = blended / blended.sum()

        alloc = {}
        for regime_idx in range(N_REGIMES):
            w = blended[regime_idx]
            for ac, ac_w in REGIME_ALLOCATIONS.get(regime_idx, {}).items():
                alloc[ac] = alloc.get(ac, 0.0) + w * ac_w

        return alloc

    def get_confidence(self) -> float:
        """Overall confidence score (0-1)."""
        if self.confidence_data is None or len(self.confidence_data) == 0:
            return 0.0
        latest = self.confidence_data.iloc[-1]
        # Combine max_prob and model_agreement
        return float(0.6 * latest["max_prob"] + 0.4 * latest["model_agreement"])

    def get_report(self) -> Dict[str, Any]:
        """Full analysis report for dashboard/CIO consumption."""
        signals = self.get_signals()
        allocations = self.get_allocations()
        ac_allocations = self.get_asset_class_allocations()
        confidence = self.get_confidence()

        # Regime transition matrix (remapped to canonical)
        trans_matrix = None
        if self.transition_matrix is not None:
            trans_matrix = {}
            T = self.transition_matrix
            state_map = self.classifier._hmm_state_map or {}
            T_remapped = np.zeros((N_REGIMES, N_REGIMES))
            for i_orig, i_canon in state_map.items():
                for j_orig, j_canon in state_map.items():
                    if i_orig < T.shape[0] and j_orig < T.shape[1]:
                        T_remapped[i_canon, j_canon] = T[i_orig, j_orig]
            # Normalize
            row_sums = T_remapped.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            T_remapped = T_remapped / row_sums

            for i in range(N_REGIMES):
                trans_matrix[REGIME_LABELS[i]] = {
                    REGIME_LABELS[j]: float(T_remapped[i, j]) for j in range(N_REGIMES)
                }

        # Feature importance
        feat_imp = None
        if self.classifier is not None:
            imp = self.classifier.feature_importance()
            feat_imp = imp.head(15).to_dict()

        # Historical regime timeline
        timeline = None
        if self.regime_history is not None:
            timeline = {
                "dates": [d.isoformat() for d in self.regime_history.index],
                "regimes": self.regime_history.values.argmax(axis=1).tolist(),
                "probabilities": self.regime_history.values.tolist(),
            }

        return {
            "manager": self.name,
            "updated_at": self._last_update.isoformat() if self._last_update else None,
            "current_regime": self.current_regime,
            "regime_probabilities": signals.get("regime_probabilities", {}),
            "forecasts": signals.get("forecasts", {}),
            "confidence": confidence,
            "model_agreement": signals.get("model_agreement", 0),
            "individual_models": {
                "hmm": signals.get("hmm_says", ""),
                "gmm": signals.get("gmm_says", ""),
                "xgb": signals.get("xgb_says", ""),
            },
            "asset_class_allocation": ac_allocations,
            "ticker_allocation": allocations,
            "transition_matrix": trans_matrix,
            "feature_importance": feat_imp,
            "regime_history": timeline,
            "ensemble_weights": (
                self.classifier.ensemble_weights if self.classifier else None
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
#  CIO Aggregator (Stub — expands as managers are added)
# ═══════════════════════════════════════════════════════════════════════


class CIOAggregator:
    """
    Chief Investment Officer aggregator.
    Combines signals from all registered managers into a unified view.
    Hybrid arch: each manager reports independently; CIO sees everything.
    """

    def __init__(self):
        self.managers: Dict[str, BaseManager] = {}

    def register_manager(self, manager: BaseManager):
        """Register an investment manager."""
        self.managers[manager.name] = manager
        logger.info(f"CIO: Registered '{manager.name}'")

    def get_unified_view(self) -> Dict[str, Any]:
        """Aggregate reports from all managers."""
        reports = {}
        for name, mgr in self.managers.items():
            try:
                reports[name] = mgr.get_report()
            except Exception as e:
                logger.error(f"CIO: Error getting report from {name}: {e}")
                reports[name] = {"error": str(e)}

        # Combined allocation (for now, just the Economic manager)
        combined_alloc = {}
        total_weight = 0
        for name, mgr in self.managers.items():
            try:
                conf = mgr.get_confidence()
                alloc = mgr.get_allocations()
                for ticker, w in alloc.items():
                    combined_alloc[ticker] = combined_alloc.get(ticker, 0) + conf * w
                total_weight += conf
            except Exception:
                pass

        if total_weight > 0:
            combined_alloc = {k: v / total_weight for k, v in combined_alloc.items()}

        return {
            "timestamp": datetime.now().isoformat(),
            "managers": reports,
            "combined_allocation": combined_alloc,
            "active_managers": list(self.managers.keys()),
        }
