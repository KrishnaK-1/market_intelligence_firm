"""
Commodities Desk — Multi-Agent Deep Reinforcement Learning
==========================================================
Architecture:
  Layer 1 (Sub-Agents): One RL agent per commodity group
      - EnergyAgent    : WTI, Brent, Natural Gas, Heating Oil
      - AgricultureAgent: Corn, Wheat, Soybeans, Live Cattle
      - MetalsAgent    : Gold, Silver, Copper, Platinum

  Layer 2 (Meta-Agent): Allocates capital across sub-agents
      - Inputs: macro/geopolitical indicators + sub-agent signals
      - Output: allocation weights per commodity group
"""
from commodities.envs.energy_env import EnergyTradingEnv
from commodities.envs.agriculture_env import AgricultureTradingEnv
from commodities.envs.metals_env import MetalsTradingEnv
from commodities.agents.sub_agent import CommoditySubAgent
from commodities.meta.allocator import MetaAllocator

__all__ = [
    "EnergyTradingEnv",
    "AgricultureTradingEnv",
    "MetalsTradingEnv",
    "CommoditySubAgent",
    "MetaAllocator",
]
