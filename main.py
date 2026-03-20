"""
Market Intelligence Firm -- Main Orchestrator
=============================================
Coordinates data ingestion, model training, agent updates,
backtesting, and alert generation.
FEEDBACK LOOP:
  1. Load persisted optimized params from disk (if they exist)
  2. Train model using those params
  3. Validate against NBER history and market returns
  4. Optimize: try new param combinations, keep the best
  5. Save best params to disk for next retrain cycle
  6. Each retrain gets progressively better
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.economic_manager import CIOAggregator, EconomicRegimeManager
from alerts.engine import AlertEngine
from backtest.engine import BacktestEngine
from config.settings import MODEL_DIR
from data.ingestion import (
    build_feature_matrix,
    clear_cache,
    fetch_all_fred,
    fetch_shiller_cape,
    fetch_yahoo_prices,
)
from models.regime_classifier import EnsembleRegimeClassifier
from models.feedback import FeedbackLoop, ParameterOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/firm.log", mode="a"),
    ],
)
logger = logging.getLogger("orchestrator")


class MarketIntelligenceFirm:
    """
    Top-level orchestrator for the Market Intelligence platform.
    Manages the full pipeline: data -> model -> agents -> alerts -> backtest.
    """

    def __init__(self):
        # Core components
        self.economic_manager = EconomicRegimeManager()
        self.cio = CIOAggregator()
        self.cio.register_manager(self.economic_manager)
        self.alert_engine = AlertEngine()
        self.backtest_engine = BacktestEngine()

        # Data
        self.fred_data: Optional[pd.DataFrame] = None
        self.shiller_data: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[pd.DataFrame] = None
        self.asset_prices: Optional[pd.DataFrame] = None
        self.asset_returns: Optional[pd.DataFrame] = None

        # Model
        self.classifier: Optional[EnsembleRegimeClassifier] = None

        # State
        self._last_run: Optional[datetime] = None
        self._backtest_results: Optional[Dict] = None
        self._alerts: list = []
        self._feedback_report: Optional[Dict] = None

        # Feedback loop with PERSISTENT optimizer
        # ParameterOptimizer loads saved params from disk on init
        self.param_optimizer = ParameterOptimizer()
        self.feedback_loop = FeedbackLoop(param_optimizer=self.param_optimizer)

    # -------------------------------------------------------------------
    #  Full Pipeline
    # -------------------------------------------------------------------

    def run_full_pipeline(self, force_retrain: bool = False, run_backtest: bool = True) -> Dict:
        """
        Execute the full pipeline with feedback loop:
        1. Ingest data
        2. Build feature matrix
        3. Load persisted params -> Train classifier with them
        4. Update Economic Regime Manager
        5. Generate alerts
        5b. Validate + Optimize -> Save better params for next time
        6. Backtest
        """
        logger.info("=" * 60)
        logger.info("MARKET INTELLIGENCE FIRM -- FULL PIPELINE RUN")
        logger.info("=" * 60)

        # -- Step 1: Data Ingestion --
        logger.info("Step 1: Ingesting data...")
        self.fred_data = fetch_all_fred()
        logger.info(f"  FRED: {self.fred_data.shape}, {self.fred_data.index[0]} -> {self.fred_data.index[-1]}")

        self.shiller_data = fetch_shiller_cape()
        logger.info(f"  Shiller: {self.shiller_data.shape}, {self.shiller_data.index[0]} -> {self.shiller_data.index[-1]}")

        self.asset_prices = fetch_yahoo_prices()
        logger.info(f"  Yahoo: {self.asset_prices.shape}")

        # Compute monthly returns
        self.asset_returns = self.asset_prices.pct_change().dropna(how="all")

        # -- Step 2: Feature Matrix --
        logger.info("Step 2: Building feature matrix...")
        self.feature_matrix = build_feature_matrix(self.fred_data, self.shiller_data)
        logger.info(f"  Features: {self.feature_matrix.shape}, cols: {list(self.feature_matrix.columns)}")

        # -- Step 3: Train or Load Classifier --
        model_path = MODEL_DIR / "ensemble_regime_classifier.pkl"
        if not force_retrain and model_path.exists():
            logger.info("Step 3: Loading saved classifier...")
            try:
                self.classifier = EnsembleRegimeClassifier.load(model_path)
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Retraining...")
                force_retrain = True

        if force_retrain or self.classifier is None:
            logger.info("Step 3: Training ensemble classifier...")
            self.classifier = EnsembleRegimeClassifier()

            # LOAD persisted optimized params (from previous feedback runs)
            optimized_params = self.param_optimizer.get_params()
            self.classifier._label_params = optimized_params
            logger.info(f"  Using params from disk: yc_thresh={optimized_params.get('yield_curve_threshold', -0.2):.2f}, "
                        f"indpro={optimized_params.get('indpro_threshold', 0.5):.2f}, "
                        f"signals>={optimized_params.get('slowdown_signal_count', 3)}")

            self.classifier.fit(self.feature_matrix)
            self.classifier.save(model_path)

        # -- Step 4: Update Economic Manager --
        logger.info("Step 4: Updating Economic Regime Manager...")
        self.economic_manager.update(
            feature_matrix=self.feature_matrix,
            classifier=self.classifier,
        )

        # -- Step 5: Generate Alerts --
        logger.info("Step 5: Evaluating alerts...")
        self._alerts = self.alert_engine.evaluate(
            current_proba=self.economic_manager.current_proba,
            current_regime=self.economic_manager.current_regime,
            forecasts=self.economic_manager.forecasts,
            confidence_data=self.economic_manager.confidence_data,
            regime_history=self.economic_manager.regime_history,
        )
        logger.info(f"  Generated {len(self._alerts)} alerts")

        # -- Step 5b: Validate + Optimize + PERSIST --
        logger.info("Step 5b: Running reinforcement feedback validation...")
        try:
            sp500_returns = None
            if self.asset_returns is not None and "SPY" in self.asset_returns.columns:
                sp500_returns = self.asset_returns["SPY"]

            self._feedback_report = self.feedback_loop.run_feedback(
                feature_matrix=self.feature_matrix,
                regime_history=self.economic_manager.regime_history,
                sp500_returns=sp500_returns,
                classifier_class=EnsembleRegimeClassifier,
                optimize=force_retrain,  # only run full optimization on retrain
                n_iterations=30,
            )
            fb = self._feedback_report
            logger.info(f"  Ground truth accuracy: {fb.get('ground_truth_validation', {}).get('accuracy', 0):.1%}")
            logger.info(f"  Return alignment: {fb.get('return_validation', {}).get('overall_alignment', 0):.1%}")
            for rec in fb.get("recommendations", [])[:3]:
                logger.info(f"  Recommendation: {rec}")

            # If optimization found better params, retrain the model NOW
            # so the dashboard serves the improved version
            opt = fb.get("optimization")
            # Score must match the optimizer's formula (35% GT + 30% ret + 35% lead)
            gt_acc = fb.get("ground_truth_validation", {}).get("accuracy", 0)
            ret_align = fb.get("return_validation", {}).get("overall_alignment", 0)
            # Compute transition detection lead for current model
            from models.feedback import NBER_RECESSIONS
            regime_history = self.economic_manager.regime_history
            if regime_history is not None and len(regime_history) > 0:
                r_idx = regime_history.values.argmax(axis=1)
                cur_preds = pd.Series(
                    [{0: "Expansion", 1: "Slowdown", 2: "Contraction",
                      3: "Recovery", 4: "Crisis"}[i] for i in r_idx],
                    index=regime_history.index,
                )
                _lead_months = []
                for _start, _ in NBER_RECESSIONS:
                    _rs = pd.Timestamp(_start)
                    _pre = cur_preds[
                        (cur_preds.index >= _rs - pd.DateOffset(months=18))
                        & (cur_preds.index < _rs)
                    ]
                    _w = _pre[_pre.isin(["Slowdown", "Contraction", "Crisis"])]
                    if len(_w) > 0:
                        _f = _w.index[0]
                        _lead_months.append(
                            (_rs.year - _f.year) * 12 + (_rs.month - _f.month)
                        )
                    else:
                        _lead_months.append(0)
                cur_lead_bonus = min(np.mean(_lead_months) / 12.0, 1.0)
            else:
                cur_lead_bonus = 0.0
            current_score = 0.35 * gt_acc + 0.30 * ret_align + 0.35 * cur_lead_bonus
            best_opt_score = opt.get("best_score", 0) if opt else 0
            improvement = best_opt_score - current_score
            if opt and improvement > 0.02:
                best_params = opt.get("best_params", {})
                logger.info(f"Step 5c: Re-training with optimized params (improvement: {improvement:.3f})...")
                self.classifier = EnsembleRegimeClassifier()
                self.classifier._label_params = best_params
                self.classifier.fit(self.feature_matrix)
                self.classifier.save(model_path)

                # Re-update Economic Manager with improved model
                self.economic_manager.update(
                    feature_matrix=self.feature_matrix,
                    classifier=self.classifier,
                )
                logger.info(f"  Model retrained with optimized params. Current regime: {self.economic_manager.current_regime}")

                # Validate the improved model
                regime_history_new = self.economic_manager.regime_history
                regime_idx = regime_history_new.values.argmax(axis=1)
                new_preds = pd.Series(
                    [dict(enumerate(["Expansion", "Slowdown", "Contraction", "Recovery", "Crisis"]))[i] for i in regime_idx],
                    index=regime_history_new.index,
                )
                # Quick accuracy check on the retrained model
                gt_new = self.feedback_loop._validate_ground_truth(new_preds, self.feature_matrix)
                ret_new = self.feedback_loop._validate_returns(new_preds, self.feature_matrix)
                logger.info(f"  AFTER optimization: GT={gt_new['accuracy']:.1%}, Ret={ret_new['overall_alignment']:.1%}")

                # Re-generate alerts with updated model
                self._alerts = self.alert_engine.evaluate(
                    current_proba=self.economic_manager.current_proba,
                    current_regime=self.economic_manager.current_regime,
                    forecasts=self.economic_manager.forecasts,
                    confidence_data=self.economic_manager.confidence_data,
                    regime_history=self.economic_manager.regime_history,
                )
                # Save improved params as the new baseline for next startup
                self.param_optimizer.update_best(best_params, best_opt_score)
            else:
                if opt:
                    logger.info(f"  Optimizer found score {best_opt_score:.3f} vs current {current_score:.3f} "
                            f"(improvement {improvement:.3f} < 0.02 threshold). Keeping current model.")
                    # Save CURRENT model's params as best so next startup uses them
                    self.param_optimizer.update_best(
                        self.classifier._label_params,
                        current_score,
                    )
        except Exception as e:
            logger.error(f"Feedback loop failed: {e}")
            import traceback
            traceback.print_exc()
            self._feedback_report = {"error": str(e)}

        # -- Step 6: Backtest (optional) --
        if run_backtest:
            logger.info("Step 6: Running walk-forward backtest...")
            try:
                self._backtest_results = self.backtest_engine.run(
                    feature_matrix=self.feature_matrix,
                    asset_returns=self.asset_returns,
                    label_params=self.classifier._label_params if self.classifier else None,
                )
                stats = self._backtest_results.get("strategy_stats", {})
                logger.info(
                    f"  Backtest: Sharpe={stats.get('sharpe_ratio', 'N/A'):.2f}, "
                    f"MaxDD={stats.get('max_drawdown', 'N/A'):.1%}"
                )
            except Exception as e:
                logger.error(f"Backtest failed: {e}")
                self._backtest_results = {"error": str(e)}
        else:
            logger.info("Step 6: Skipping backtest")

        self._last_run = datetime.now()
        logger.info(f"Pipeline complete at {self._last_run}")

        return self.get_full_report()

    def run_inference_only(self) -> Dict:
        """Re-pull data and run inference with existing model (no retraining)."""
        logger.info("Running inference-only refresh...")

        self.fred_data = fetch_all_fred()
        self.shiller_data = fetch_shiller_cape()
        self.asset_prices = fetch_yahoo_prices()
        self.asset_returns = self.asset_prices.pct_change().dropna(how="all")
        self.feature_matrix = build_feature_matrix(self.fred_data, self.shiller_data)

        if self.classifier is None:
            model_path = MODEL_DIR / "ensemble_regime_classifier.pkl"
            if model_path.exists():
                self.classifier = EnsembleRegimeClassifier.load(model_path)
            else:
                raise RuntimeError("No trained model found. Run full pipeline first.")

        self.economic_manager.update(
            feature_matrix=self.feature_matrix,
            classifier=self.classifier,
        )

        self._alerts = self.alert_engine.evaluate(
            current_proba=self.economic_manager.current_proba,
            current_regime=self.economic_manager.current_regime,
            forecasts=self.economic_manager.forecasts,
            confidence_data=self.economic_manager.confidence_data,
            regime_history=self.economic_manager.regime_history,
        )

        self._last_run = datetime.now()
        return self.get_full_report()

    def force_refresh(self) -> Dict:
        """Clear cache and run full pipeline with retraining."""
        clear_cache()
        return self.run_full_pipeline(force_retrain=True, run_backtest=True)

    # -------------------------------------------------------------------
    #  Reporting
    # -------------------------------------------------------------------

    def get_full_report(self) -> Dict:
        """Assemble complete report for dashboard consumption."""
        cio_view = self.cio.get_unified_view()

        return {
            **cio_view,
            "alerts": self.alert_engine.get_active_alerts(),
            "backtest": self._backtest_results,
            "feedback": self._feedback_report,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "data_info": {
                "fred_range": (
                    f"{self.fred_data.index[0].strftime('%Y-%m')} -> "
                    f"{self.fred_data.index[-1].strftime('%Y-%m')}"
                    if self.fred_data is not None
                    else None
                ),
                "n_features": (
                    len(self.feature_matrix.columns) if self.feature_matrix is not None else 0
                ),
                "n_observations": (
                    len(self.feature_matrix) if self.feature_matrix is not None else 0
                ),
            },
        }


# ===================================================================
#  CLI Entry Point
# ===================================================================

if __name__ == "__main__":
    import json

    firm = MarketIntelligenceFirm()
    report = firm.run_full_pipeline(force_retrain=True, run_backtest=True)

    econ = report.get("managers", {}).get("Economic Regime Manager", {})
    print("\n" + "=" * 60)
    print("MARKET INTELLIGENCE FIRM -- SUMMARY")
    print("=" * 60)
    print(f"Current Regime: {econ.get('current_regime', 'N/A')}")
    print(f"Confidence: {econ.get('confidence', 0):.1%}")
    print(f"\nRegime Probabilities:")
    for regime, prob in econ.get("regime_probabilities", {}).items():
        print(f"  {regime}: {prob:.1%}")
    print(f"\nForecasts:")
    for horizon, fcast in econ.get("forecasts", {}).items():
        top = max(fcast, key=fcast.get)
        print(f"  {horizon}: {top} ({fcast[top]:.1%})")

    with open("logs/latest_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to logs/latest_report.json")
