"""
Build the meta-allocator's training dataset by stitching together each
sector's GENUINE out-of-sample daily returns from the walk-forward folds
(2020-2025). Each fold's model only ever traded a year it never saw during
training, so the resulting return series has no look-ahead leakage —
the meta-allocator only ever learns from real unseen-data performance.

Output: commodities/walkforward_results/meta_dataset.pkl
        {"returns": DataFrame[date, energy, agriculture, metals],
         "macro": DataFrame[date, ...]}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from commodities.agents.sub_agent import CommoditySubAgent
from commodities.data.fetcher import fetch_macro

GROUPS = ["energy", "agriculture", "metals"]
YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"]
OUT_PATH = Path(__file__).resolve().parent / "walkforward_results" / "meta_dataset.pkl"


def get_env_class(group: str):
    if group == "energy":
        from commodities.envs.energy_env import EnergyTradingEnv as EnvCls
    elif group == "agriculture":
        from commodities.envs.agriculture_env import AgricultureTradingEnv as EnvCls
    elif group == "metals":
        from commodities.envs.metals_env import MetalsTradingEnv as EnvCls
    return EnvCls


def stitch_group_returns(group: str) -> pd.Series:
    EnvCls = get_env_class(group)
    all_dates, all_rets = [], []

    for year in YEARS:
        agent = CommoditySubAgent(EnvCls, algo="ppo", name=f"{group}_wf_{year}")
        agent.load()
        oos = agent.run_episode(env_kwargs={"start": f"{year}-01-01", "end": f"{year}-12-31"})
        # portfolio_values[0] = 1.0 (start of episode, before lookback offset); dates align with values[1:]
        values = np.array(oos["portfolio_values"])
        daily_rets = np.diff(values) / values[:-1]
        all_dates.extend(oos["dates"])
        all_rets.extend(daily_rets.tolist())
        print(f"  {group} {year}: {len(oos['dates'])} trading days stitched")

    series = pd.Series(all_rets, index=pd.to_datetime(all_dates), name=group)
    return series[~series.index.duplicated(keep="first")].sort_index()


if __name__ == "__main__":
    print("Stitching out-of-sample daily returns for each sector...")
    series_by_group = {g: stitch_group_returns(g) for g in GROUPS}
    returns_df = pd.DataFrame(series_by_group).dropna()
    print(f"\nCombined returns shape: {returns_df.shape}")
    print(returns_df.describe())

    macro = fetch_macro(start=str(returns_df.index.min().date()), end=str(returns_df.index.max().date()))
    macro_close = macro["Close"] if "Close" in macro else macro
    macro_close = macro_close.reindex(returns_df.index).ffill().bfill()

    OUT_PATH.parent.mkdir(exist_ok=True)
    pd.to_pickle({"returns": returns_df, "macro": macro_close}, OUT_PATH)
    print(f"\nSaved meta-allocator training dataset to {OUT_PATH}")
