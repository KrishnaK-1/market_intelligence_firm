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
    ) -> Dict:
        """
        Run walk-forward backtest.
        Only tests on periods where both features and asset returns exist.
        """
        from models.regime_classifier import EnsembleRegimeClassifier

        classifier_class = classifier_class or EnsembleRegimeClassifier

        logger.info("Starting walk-forward backtest...")

        # Align data to common date range
        common_idx = feature_matrix.index.intersection(asset_returns.index)
        if len(common_idx) < self.min_train_months + 12:
            logger.warning(f"Only {len(common_idx)} overlapping months. Reducing min_train to fit.")
            self.min_train_months = max(60, len(common_idx) // 2)

        features = feature_matrix.loc[common_idx].copy()
        returns = asset_returns.loc[common_idx].copy()

        n_obs = len(features)
        logger.info(
            f"Backtest data: {n_obs} months, {features.index[0].strftime('%Y-%m')} to {features.index[-1].strftime('%Y-%m')}")

        # Walk-forward splits on the OVERLAPPING data only
        test_pool = n_obs - self.min_train_months
        if test_pool <= 0:
            logger.error("Not enough data for backtesting")
            return {"error": "Insufficient overlapping data"}

        split_size = max(12, test_pool // self.n_splits)

        regime_predictions = pd.Series(dtype=str, name="predicted_regime")
        regime_actuals = pd.Series(dtype=float, name="actual_recession")
        strategy_returns = pd.Series(dtype=float, name="strategy_return")
        benchmark_returns = pd.Series(dtype=float, name="benchmark_return")
        regime_accuracy_by_split = []

        for split_idx in range(self.n_splits):
            train_end = self.min_train_months + split_idx * split_size
            test_start = train_end
            test_end = min(train_end + split_size, n_obs)

            if test_start >= n_obs or test_end <= test_start:
                break

            train_data = features.iloc[:train_end]
            test_data = features.iloc[test_start:test_end]

            logger.info(
                f"  Split {split_idx}: train {train_data.index[0].strftime('%Y-%m')} to {train_data.index[-1].strftime('%Y-%m')}, test {test_data.index[0].strftime('%Y-%m')} to {test_data.index[-1].strftime('%Y-%m')}")

            # Train fresh model on training data
            model = classifier_class(n_regimes=5)
            try:
                model.fit(train_data)
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
                    "n_months": test_end - test_start,
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
