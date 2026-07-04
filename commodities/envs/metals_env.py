"""
Environment 3 — Metals Desk
Assets: Gold, Silver, Copper, Platinum
Extra state: Nominal 10Y yield, TIPS (real-yield component), DXY (inverse
             relationship with gold), GVZ (gold volatility index),
             VIX (crisis trigger), CREDIT (HYG, credit stress)
"""
import pandas as pd
from commodities.envs.base_env import CommodityTradingEnv
from commodities.data.fetcher import get_close_prices, fetch_macro


class MetalsTradingEnv(CommodityTradingEnv):
    """
    Metals environment. Real yields and dollar strength are the dominant
    drivers of gold/silver. Copper is an industrial demand proxy (Dr. Copper).

    Real yield approximation: 10Y nominal yield minus TIPS breakeven
    (both available in macro tickers).
    """

    GROUP = "metals"

    def __init__(self, start: str = "2015-01-01", end: str = None, **kwargs):
        prices = get_close_prices(self.GROUP, start=start, end=end)
        prices = prices.dropna()

        macro_df = None
        macro = fetch_macro(start=start, end=end)
        if "Close" in macro:
            macro_close = macro["Close"][["UUP", "^TNX", "TIP", "^GVZ", "^VIX", "HYG"]].rename(
                columns={"UUP": "DXY", "^TNX": "YIELD_10Y", "TIP": "TIPS", "^GVZ": "GVZ", "^VIX": "VIX", "HYG": "CREDIT"}
            )
            prices, macro_df = prices.align(macro_close, join="inner", axis=0)
            prices = prices.dropna()
            macro_df = macro_df.reindex(prices.index)

        super().__init__(prices, macro=macro_df, **kwargs)
