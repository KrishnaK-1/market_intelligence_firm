"""
Environment 1 — Energy Desk
Assets: WTI Crude, Brent Crude, Natural Gas
Extra state: OVX (oil volatility index), DXY (dollar index), TIPS (real yield /
             inflation-expectations proxy), VIX (crisis trigger), CREDIT (HYG, credit stress)
"""
import numpy as np
import pandas as pd
from commodities.envs.base_env import CommodityTradingEnv
from commodities.data.fetcher import get_close_prices, fetch_macro


class EnergyTradingEnv(CommodityTradingEnv):
    """
    Extends the base env with energy-specific state variables:
    oil volatility (OVX) and dollar strength (DXY), both of which
    are leading indicators of crude price moves, plus the broad
    crisis indicators (VIX, credit stress) used by the base env's
    crisis override.
    """

    GROUP = "energy"

    def __init__(self, start: str = "2015-01-01", end: str = None, **kwargs):
        prices = get_close_prices(self.GROUP, start=start, end=end)
        prices = prices.dropna()

        macro_df = None
        macro = fetch_macro(start=start, end=end)
        if "Close" in macro:
            macro_close = macro["Close"][["UUP", "^OVX", "TIP", "^VIX", "HYG"]].rename(
                columns={"UUP": "DXY", "^OVX": "OVX", "TIP": "TIPS", "^VIX": "VIX", "HYG": "CREDIT"}
            )
            prices, macro_df = prices.align(macro_close, join="inner", axis=0)
            prices = prices.dropna()
            macro_df = macro_df.reindex(prices.index)

        super().__init__(prices, macro=macro_df, **kwargs)
