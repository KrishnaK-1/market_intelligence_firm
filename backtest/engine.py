"""
Backtesting Engine
===================
Walk-forward backtest of regime model + allocation strategy.
Measures: accuracy, regime timing, allocation returns, drawdowns.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config.settings import (
    N_REGIMES,
    REGIME_ALLOCATIONS,
    REGIME_LABELS,
    REGIME_SUB_ALLOCATIONS,
)

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Walk-forward backtester for regime-based allocation strategies."""

    def __init__(
        self,
        n_splits: int = 5,
        min_train_months: int = 120,  # 10 years minimum training
        rebalance_freq: str = "M",     # monthly rebalance
    ):
        self.n_splits = n_splits
        self.min_train_months = min_train_months
        self.rebalance_freq = rebalance_freq
        self.results: Optional[Dict] = None

    def run(
            self,
            feature_matrix: pd.DataFrame,
            asset_returns: pd.DataFrame,
            classifier_class=None,
            label_params: Optional[Dict] = None,
    ) -> Dict:
        """
        Run walk-forward backtest.

        Training uses the FULL feature matrix (1952+) so every split sees
        multiple recessions.  Testing/allocation evaluation is restricted to
        the overlap period where both features AND asset returns exist.
        """
        from models.regime_classifier import EnsembleRegimeClassifier

        classifier_class = classifier_class or EnsembleRegimeClassifier

        logger.info("Starting walk-forward backtest...")

        # ── Identify the overlap period (for TEST windows only) ──
        common_idx = feature_matrix.index.intersection(asset_returns.index)
        if len(common_idx) < 36:
            logger.error(f"Only {len(common_idx)} overlapping months — need at least 36")
            return {"error": "Insufficient overlapping data"}

        overlap_start = common_idx[0]
        overlap_end = common_idx[-1]

        returns = asset_returns.loc[common_idx].copy()

        # Full feature matrix for training (includes pre-ETF history)
        full_features = feature_matrix.copy()

        n_overlap = len(common_idx)
        logger.info(
            f"Backtest data: {n_overlap} months overlap, "
            f"{overlap_start.strftime('%Y-%m')} to {overlap_end.strftime('%Y-%m')}, "
            f"full training history: {len(full_features)} months from {full_features.index[0].strftime('%Y-%m')}")

        # ── Define test windows across the overlap period ──
        # Divide the overlap into n_splits roughly equal test windows
        split_size = max(12, n_overlap // self.n_splits)

        regime_predictions = pd.Series(dtype=str, name="predicted_regime")
        regime_actuals = pd.Series(dtype=float, name="actual_recession")
        strategy_returns = pd.Series(dtype=float, name="strategy_return")
        benchmark_returns = pd.Series(dtype=float, name="benchmark_return")
        regime_accuracy_by_split = []

        for split_idx in range(self.n_splits):
            # Test window: a slice of the overlap period
            test_start_pos = split_idx * split_size
            test_end_pos = min(test_start_pos + split_size, n_overlap)

            if test_start_pos >= n_overlap or test_end_pos <= test_start_pos:
                break

            test_start_date = common_idx[test_start_pos]
            test_end_date = common_idx[test_end_pos - 1]

            # Training: ALL feature data BEFORE the test window start
            train_data = full_features[full_features.index < test_start_date]
            test_data = feature_matrix.loc[
                (feature_matrix.index >= test_start_date) &
                (feature_matrix.index <= test_end_date)
            ]

            if len(train_data) < self.min_train_months:
                logger.warning(
                    f"  Split {split_idx}: only {len(train_data)} training months "
                    f"(need {self.min_train_months}), skipping")
                continue

            logger.info(
                f"  Split {split_idx}: train {train_data.index[0].strftime('%Y-%m')} to "
                f"{train_data.index[-1].strftime('%Y-%m')} ({len(train_data)} months), "
                f"test {test_start_date.strftime('%Y-%m')} to "
                f"{test_end_date.strftime('%Y-%m')} ({len(test_data)} months)")

            # Train fresh model on training data
            # Pre-fill NaNs: features like VIX/CAPE start later than 1952.
            # ffill+bfill on the training slice propagates values where possible.
            # Then drop rows where more than half the columns are still NaN.
            train_filled = train_data.ffill().bfill()
            n_cols = len(train_filled.columns)
            train_filled = train_filled.dropna(thresh=int(n_cols * 0.5))

            if len(train_filled) < self.min_train_months:
                logger.warning(
                    f"  Split {split_idx}: only {len(train_filled)} valid rows after fill "
                    f"(need {self.min_train_months}), skipping")
                continue

            model = classifier_class(n_regimes=5)
            if label_params is not None:
                model._label_params = label_params
            try:
                model.fit(train_filled)
            except Exception as e:
                logger.warning(f"  Split {split_idx} training failed: {e}")
                continue

            # Predict on test period
            test_features = test_data.drop(columns=["recession"], errors="ignore")
            preds = model.predict(test_features)
            proba = model.predict_proba(test_features)

            regime_predictions = pd.concat([regime_predictions, preds])

            # Track actual recession labels for accuracy
            if "recession" in test_data.columns:
                actuals = test_data["recession"]
                regime_actuals = pd.concat([regime_actuals, actuals])

                pred_contraction = (preds == "Contraction") | (preds == "Crisis")
                actual_recession = actuals == 1
                if actual_recession.sum() > 0:
                    recall = (pred_contraction & actual_recession).sum() / actual_recession.sum()
                else:
                    recall = np.nan
                regime_accuracy_by_split.append({
                    "split": split_idx,
                    "test_start": test_data.index[0].strftime("%Y-%m"),
                    "test_end": test_data.index[-1].strftime("%Y-%m"),
                    "recession_recall": float(recall) if not np.isnan(recall) else None,
                    "n_months": len(test_data),
                })

            # Compute strategy returns
            from config.settings import REGIME_LABELS, REGIME_SUB_ALLOCATIONS
            for t_idx in range(len(test_data)):
                t_date = test_data.index[t_idx]
                if t_date not in returns.index:
                    continue

                regime_name = preds.iloc[t_idx]
                regime_idx = {v: k for k, v in REGIME_LABELS.items()}.get(regime_name, 0)
                sub_alloc = REGIME_SUB_ALLOCATIONS.get(regime_idx, {})

                port_ret = 0.0
                for ticker, weight in sub_alloc.items():
                    if ticker in returns.columns:
                        ret_val = returns.loc[t_date, ticker]
                        if pd.notna(ret_val):
                            port_ret += weight * ret_val

                strategy_returns.loc[t_date] = port_ret

                available = [t for t in sub_alloc.keys() if t in returns.columns]
                if available:
                    bm_ret = returns.loc[t_date, available].mean()
                    benchmark_returns.loc[t_date] = bm_ret

        # Compute summary statistics
        strategy_returns = strategy_returns.dropna().sort_index()
        benchmark_returns = benchmark_returns.dropna().sort_index()
        strategy_returns = strategy_returns[~strategy_returns.index.duplicated(keep="last")]
        benchmark_returns = benchmark_returns[~benchmark_returns.index.duplicated(keep="last")]

        strat_cum = (1 + strategy_returns).cumprod()
        bm_cum = (1 + benchmark_returns).cumprod()

        self.results = {
            "strategy_cumulative_returns": {
                "dates": [d.isoformat() for d in strat_cum.index],
                "values": strat_cum.values.tolist(),
            },
            "benchmark_cumulative_returns": {
                "dates": [d.isoformat() for d in bm_cum.index],
                "values": bm_cum.values.tolist(),
            },
            "strategy_stats": self._compute_stats(strategy_returns, "Strategy"),
            "benchmark_stats": self._compute_stats(benchmark_returns, "Benchmark"),
            "regime_accuracy": regime_accuracy_by_split,
            "n_months_tested": len(strategy_returns),
            "regime_predictions": {
                "dates": [d.isoformat() for d in regime_predictions.index],
                "regimes": regime_predictions.values.tolist(),
            },
        }

        logger.info(
            f"Backtest complete: {len(strategy_returns)} months across {len(regime_accuracy_by_split)} splits")
        return self.results

    def _compute_stats(self, returns: pd.Series, name: str) -> Dict:
        """Compute standard performance statistics."""
        if len(returns) == 0:
            return {"name": name, "error": "No data"}

        ann_return = (1 + returns.mean()) ** 12 - 1
        ann_vol = returns.std() * np.sqrt(12)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Max drawdown
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        drawdown = (cum - peak) / peak
        max_dd = drawdown.min()

        # Calmar ratio
        calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0

        return {
            "name": name,
            "annualized_return": float(ann_return),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(calmar),
            "total_return": float((1 + returns).prod() - 1),
            "n_months": len(returns),
            "best_month": float(returns.max()),
            "worst_month": float(returns.min()),
            "pct_positive_months": float((returns > 0).mean()),
        }
