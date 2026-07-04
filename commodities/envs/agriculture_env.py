"""
Environment 2 — Agriculture Desk
Assets: Corn, Wheat, Soybeans
Extra state: DXY (USD strength affects export demand), TIPS (inflation proxy for food prices),
             BDRY (Baltic Dry shipping ETF — freight cost/export disruption proxy,
             relevant to grain export shocks like the 2022 Black Sea blockade),
             VIX (crisis trigger), CREDIT (HYG, credit stress)
"""
import pandas as pd
from commodities.envs.base_env import CommodityTradingEnv
from commodities.data.fetcher import get_close_prices, fetch_macro


class AgricultureTradingEnv(CommodityTradingEnv):
    """
    Agriculture commodity environment. USD strength and inflation expectations
    both drive demand shifts in grain markets. Agriculture has no dedicated
    volatility index (unlike OVX for oil or GVZ for gold), so VIX is the
    primary crisis-detection signal here.

    Note: Climate/yield data (e.g. NOAA seasonal forecasts) can be added
    as additional columns to prices before passing to super().__init__.
    """

    GROUP = "agriculture"

    def __init__(self, start: str = "2015-01-01", end: str = None, **kwargs):
        prices = get_close_prices(self.GROUP, start=start, end=end)
        prices = prices.dropna()

        macro_df = None
        macro = fetch_macro(start=start, end=end)
        if "Close" in macro:
            macro_close = macro["Close"][["UUP", "TIP", "BDRY", "^VIX", "HYG"]].rename(
                columns={"UUP": "DXY", "TIP": "TIPS", "BDRY": "SHIPPING", "^VIX": "VIX", "HYG": "CREDIT"}
            )
            prices, macro_df = prices.align(macro_close, join="inner", axis=0)
            prices = prices.dropna()
            macro_df = macro_df.reindex(prices.index)

        super().__init__(prices, macro=macro_df, **kwargs)
