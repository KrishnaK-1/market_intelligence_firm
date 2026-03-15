"""
Alert System — Regime Shift Early Warning
==========================================
Monitors for:
  - Regime transition signals (probability thresholds)
  - Model disagreement spikes
  - Rapid probability shifts
  - Leading indicator deterioration
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import REGIME_LABELS, N_REGIMES

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    timestamp: str
    severity: str
    title: str
    message: str
    category: str  # "regime_shift", "disagreement", "momentum", "indicator"
    data: Dict = field(default_factory=dict)


class AlertEngine:
    """Generates alerts based on regime model outputs."""

    def __init__(self):
        self.alerts: List[Alert] = []
        self._previous_proba: Optional[np.ndarray] = None
        self._previous_regime: Optional[str] = None

    def evaluate(
        self,
        current_proba: np.ndarray,
        current_regime: str,
        forecasts: Dict[int, np.ndarray],
        confidence_data: pd.DataFrame,
        regime_history: pd.DataFrame,
    ) -> List[Alert]:
        """Run all alert checks and return new alerts."""
        self.alerts = []
        now = datetime.now().isoformat()

        # ── 1. Regime transition in progress ──
        self._check_regime_transition(current_proba, current_regime, now)

        # ── 2. Model disagreement ──
        self._check_model_disagreement(confidence_data, now)

        # ── 3. Rapid probability shift ──
        self._check_probability_momentum(regime_history, now)

        # ── 4. Forecast divergence (current vs forward) ──
        self._check_forecast_divergence(current_proba, forecasts, now)

        # ── 5. Approaching crisis threshold ──
        self._check_crisis_warning(current_proba, forecasts, now)

        # Update state for next evaluation
        self._previous_proba = current_proba.copy()
        self._previous_regime = current_regime

        return self.alerts

    def _check_regime_transition(self, proba: np.ndarray, regime: str, ts: str):
        """Alert if the dominant regime is shifting."""
        if self._previous_regime and regime != self._previous_regime:
            self.alerts.append(Alert(
                timestamp=ts,
                severity=AlertSeverity.CRITICAL,
                title="Regime Transition Detected",
                message=f"Regime shifted from {self._previous_regime} → {regime}",
                category="regime_shift",
                data={"from": self._previous_regime, "to": regime,
                      "probabilities": {REGIME_LABELS[i]: float(proba[i]) for i in range(len(proba))}},
            ))

        # Near-transition: no single regime dominates (max < 0.45)
        max_prob = proba.max()
        if max_prob < 0.45:
            sorted_idx = np.argsort(proba)[::-1]
            self.alerts.append(Alert(
                timestamp=ts,
                severity=AlertSeverity.WARNING,
                title="Regime Uncertainty Elevated",
                message=(
                    f"No dominant regime — top two: "
                    f"{REGIME_LABELS[sorted_idx[0]]} ({proba[sorted_idx[0]]:.1%}) "
                    f"vs {REGIME_LABELS[sorted_idx[1]]} ({proba[sorted_idx[1]]:.1%})"
                ),
                category="regime_shift",
                data={"max_prob": float(max_prob)},
            ))

    def _check_model_disagreement(self, confidence_data: pd.DataFrame, ts: str):
        """Alert if ensemble models disagree strongly."""
        if confidence_data is None or len(confidence_data) == 0:
            return
        latest = confidence_data.iloc[-1]
        agreement = latest.get("model_agreement", 1.0)
        if agreement < 0.33:
            self.alerts.append(Alert(
                timestamp=ts,
                severity=AlertSeverity.WARNING,
                title="High Model Disagreement",
                message=(
                    f"Models diverge: HMM={latest.get('hmm_regime', '?')}, "
                    f"GMM={latest.get('gmm_regime', '?')}, "
                    f"XGBoost={latest.get('xgb_regime', '?')}"
                ),
                category="disagreement",
                data={"agreement_score": float(agreement)},
            ))

    def _check_probability_momentum(self, regime_history: pd.DataFrame, ts: str):
        """Alert if regime probabilities are shifting rapidly (3-month delta)."""
        if regime_history is None or len(regime_history) < 4:
            return
        current = regime_history.iloc[-1].values
        three_months_ago = regime_history.iloc[-4].values if len(regime_history) >= 4 else current

        delta = current - three_months_ago
        max_change_idx = np.abs(delta).argmax()
        max_change = delta[max_change_idx]

        if abs(max_change) > 0.20:
            direction = "rising" if max_change > 0 else "falling"
            regime = REGIME_LABELS[max_change_idx]
            self.alerts.append(Alert(
                timestamp=ts,
                severity=AlertSeverity.WARNING,
                title=f"Rapid Probability Shift — {regime}",
                message=f"{regime} probability {direction} by {abs(max_change):.1%} over 3 months",
                category="momentum",
                data={"regime": regime, "delta_3m": float(max_change)},
            ))

    def _check_forecast_divergence(self, current: np.ndarray, forecasts: Dict, ts: str):
        """Alert if forward forecasts differ materially from current."""
        if not forecasts:
            return
        current_top = REGIME_LABELS[current.argmax()]

        for horizon, fcast in forecasts.items():
            fcast_top = REGIME_LABELS[fcast.argmax()]
            if fcast_top != current_top and fcast.max() > 0.35:
                self.alerts.append(Alert(
                    timestamp=ts,
                    severity=AlertSeverity.INFO,
                    title=f"{horizon}M Forecast Differs from Current",
                    message=(
                        f"Current: {current_top} | "
                        f"{horizon}M forecast: {fcast_top} ({fcast.max():.1%})"
                    ),
                    category="regime_shift",
                    data={"horizon": horizon, "current": current_top,
                          "forecast": fcast_top},
                ))

    def _check_crisis_warning(self, current: np.ndarray, forecasts: Dict, ts: str):
        """Special alert if Crisis probability is rising."""
        crisis_idx = 4  # Crisis state
        crisis_now = current[crisis_idx] if crisis_idx < len(current) else 0

        if crisis_now > 0.15:
            self.alerts.append(Alert(
                timestamp=ts,
                severity=AlertSeverity.CRITICAL,
                title="Crisis Probability Elevated",
                message=f"Crisis regime probability at {crisis_now:.1%}",
                category="regime_shift",
                data={"crisis_prob": float(crisis_now)},
            ))

        # Check forward
        for h, fcast in (forecasts or {}).items():
            fcast_crisis = fcast[crisis_idx] if crisis_idx < len(fcast) else 0
            if fcast_crisis > 0.20 and fcast_crisis > crisis_now + 0.05:
                self.alerts.append(Alert(
                    timestamp=ts,
                    severity=AlertSeverity.CRITICAL,
                    title=f"Crisis Risk Rising at {h}M Horizon",
                    message=f"Crisis probability: current {crisis_now:.1%} → {h}M {fcast_crisis:.1%}",
                    category="regime_shift",
                    data={"horizon": h, "current_crisis": float(crisis_now),
                          "forecast_crisis": float(fcast_crisis)},
                ))

    def get_active_alerts(self) -> List[Dict]:
        """Return alerts as dicts for API/dashboard consumption."""
        return [
            {
                "timestamp": a.timestamp,
                "severity": a.severity,
                "title": a.title,
                "message": a.message,
                "category": a.category,
                "data": a.data,
            }
            for a in self.alerts
        ]
