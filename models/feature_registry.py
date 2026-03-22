"""
Feature Registry
================
Manages base vs experimental features for the neural regime classifier.
Base features (24) are shared with XGBoost; experimental features can be
added/removed without affecting the existing ensemble.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The 24 base features used by XGBoost and the HMM/GMM ensemble.
# These come from build_feature_matrix(fred_df, shiller_df) in data/ingestion.py
# and include cape + cape_zscore derived from the Shiller ie_data.xls file.
BASE_FEATURES: List[str] = [
    "spread_10y2y", "spread_10y3m", "credit_spread", "credit_spread_chg_3m",
    "unrate", "unrate_chg_3m", "unrate_chg_12m",
    "indpro_yoy", "indpro_mom", "cpi_yoy", "houst_yoy",
    "payems_yoy", "payems_mom", "umcsent", "umcsent_chg",
    "icsa_4wk_chg", "fedfunds", "fedfunds_chg_12m",
    "m2_yoy", "sp500_yoy", "sp500_mom_3m", "vix",
    "cape", "cape_zscore",
]

# Experimental features that the neural net can use but XGBoost does not.
# Each entry defines how to compute the feature from raw FRED data.
EXPERIMENTAL_FEATURES: Dict[str, dict] = {
    "oil_yoy": {
        "fred_series": "DCOILWTICO",
        "transform": "pct_change_12",
        "description": "WTI Crude Oil Year-over-Year % Change",
        "rationale": "Oil price shocks historically precede or accompany recessions",
        "available_from": "1986",
        "yahoo_fallback": "CL=F",
    },
    "permit_yoy": {
        "fred_series": "PERMIT",
        "transform": "pct_change_12",
        "description": "Building Permits Year-over-Year % Change",
        "rationale": "Leading housing indicator, signals future construction/economic activity",
        "available_from": "1960",
    },
    "ahe_yoy": {
        "fred_series": "CES0500000003",
        "transform": "pct_change_12",
        "description": "Average Hourly Earnings Year-over-Year % Change",
        "rationale": "Wage growth signals labor market tightness and inflation pressure",
        "available_from": "1964",
    },
    "lei_yoy": {
        "fred_series": "USSLIND",
        "transform": "pct_change_12",
        "description": "Conference Board Leading Economic Index Year-over-Year % Change",
        "rationale": "Composite of 10 leading indicators; AUC 0.97 for recession prediction at 1-3 month horizon per Chicago Fed research",
        "available_from": "1959",
    },
}


class FeatureRegistry:
    """Manages base and experimental feature sets for the neural regime classifier."""

    def __init__(self):
        self.base_features = list(BASE_FEATURES)
        self.experimental_features = dict(EXPERIMENTAL_FEATURES)

    def get_base_features(self) -> List[str]:
        """Return the 24 base feature names."""
        return list(self.base_features)

    def get_all_features(self, include_experimental: bool = True) -> List[str]:
        """Return feature list, optionally including experimental features."""
        features = list(self.base_features)
        if include_experimental:
            features.extend(self.experimental_features.keys())
        return features

    def compute_experimental_features(
        self,
        fred_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute experimental features from raw FRED data.

        Args:
            fred_df: Raw FRED DataFrame from fetch_all_fred()

        Returns:
            DataFrame with experimental feature columns, same index as fred_df
            (resampled to month-end).
        """
        result = pd.DataFrame(index=fred_df.index)

        for feat_name, spec in self.experimental_features.items():
            series_id = spec["fred_series"]
            transform = spec["transform"]

            if series_id not in fred_df.columns:
                # Try Yahoo Finance fallback if configured
                yahoo_ticker = spec.get("yahoo_fallback")
                if yahoo_ticker:
                    try:
                        import yfinance as yf
                        oil = yf.download(
                            yahoo_ticker, start="1983-01-01",
                            progress=False, auto_adjust=True,
                        )["Close"]
                        oil_monthly = oil.resample("ME").last()
                        if transform == "pct_change_12":
                            result[feat_name] = oil_monthly.pct_change(12) * 100
                        elif transform == "pct_change_1":
                            result[feat_name] = oil_monthly.pct_change(1) * 100
                        else:
                            result[feat_name] = oil_monthly
                        result[feat_name] = result[feat_name].reindex(
                            fred_df.index, method="ffill"
                        )
                        non_null = result[feat_name].notna().sum()
                        logger.info(
                            f"  Experimental feature '{feat_name}': {non_null} obs "
                            f"(Yahoo fallback: {yahoo_ticker})"
                        )
                        continue
                    except Exception as e:
                        logger.warning(
                            f"Yahoo fallback for '{feat_name}' also failed: {e}"
                        )
                logger.warning(
                    f"FRED series '{series_id}' not found for experimental "
                    f"feature '{feat_name}'. Add it to config/settings.py FRED_SERIES."
                )
                continue

            raw = fred_df[series_id].copy()

            if transform == "pct_change_12":
                result[feat_name] = raw.pct_change(12) * 100
            elif transform == "pct_change_1":
                result[feat_name] = raw.pct_change(1) * 100
            elif transform == "diff_12":
                result[feat_name] = raw.diff(12)
            elif transform == "diff_1":
                result[feat_name] = raw.diff(1)
            elif transform == "level":
                result[feat_name] = raw
            else:
                logger.warning(f"Unknown transform '{transform}' for '{feat_name}'")
                continue

            non_null = result[feat_name].notna().sum()
            logger.info(
                f"Experimental feature '{feat_name}': {non_null} valid observations "
                f"(from {spec.get('available_from', '?')})"
            )

        return result

    def describe(self) -> Dict[str, dict]:
        """Return metadata about all features (base + experimental)."""
        info = {}
        for feat in self.base_features:
            info[feat] = {"type": "base", "description": f"Base feature: {feat}"}
        for feat_name, spec in self.experimental_features.items():
            info[feat_name] = {
                "type": "experimental",
                "description": spec.get("description", ""),
                "rationale": spec.get("rationale", ""),
                "fred_series": spec.get("fred_series", ""),
            }
        return info
