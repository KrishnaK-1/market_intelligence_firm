"""
Meta-Allocator — Layer 2 of the commodities hierarchy.

Inputs : signals from each sub-agent + macro/geopolitical indicators
Output : allocation weights across Energy, Agriculture, Metals (sums to 1)

At training time, each sub-agent returns a "signal" (e.g. its portfolio return
or a confidence score). The meta-agent observes these signals plus macro state
and decides how to distribute capital across the three groups.

This can be trained with PPO as well, treating it as a higher-level portfolio
problem where each "asset" is a commodity sub-strategy.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from commodities.data.fetcher import fetch_macro

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "saved" / "commodities"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = ["energy", "agriculture", "metals"]
N_GROUPS = len(GROUPS)


class MetaAllocationEnv(gym.Env):
    """
    Environment for the meta-agent. At each step it receives:
      - Macro state (DXY, VIX, 10Y yield, TIPS, OVX, GVZ) — 6 features
      - Sub-agent signals: simulated returns for each group — N_GROUPS features
    Action: allocation weight to each group (sums to 1 after softmax)
    Reward: weighted sum of sub-agent returns minus turnover cost
    """

    def __init__(
        self,
        macro_df: pd.DataFrame,
        group_returns: dict[str, pd.Series],
        start: str = "2015-01-01",
    ):
        super().__init__()
        self.macro = macro_df.values.astype(np.float32)
        self.group_rets = np.column_stack([
            group_returns[g].values for g in GROUPS
        ]).astype(np.float32)  # (T, 3)

        self.T = min(len(self.macro), len(self.group_rets))
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(N_GROUPS,), dtype=np.float32)
        obs_dim = self.macro.shape[1] + N_GROUPS
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.weights = np.ones(N_GROUPS, dtype=np.float32) / N_GROUPS
        return self._obs(), {}

    def step(self, action: np.ndarray):
        # Softmax normalization so weights sum to 1
        exp_a = np.exp(action - action.max())
        new_weights = exp_a / exp_a.sum()

        turnover = np.sum(np.abs(new_weights - self.weights))
        port_ret = float(np.dot(new_weights, self.group_rets[self.t])) - 0.001 * turnover

        self.weights = new_weights
        self.t += 1

        terminated = self.t >= self.T - 1
        return self._obs(), np.log1p(port_ret), terminated, False, {"step_return": port_ret}

    def _obs(self):
        t = min(self.t, self.T - 1)
        return np.concatenate([self.macro[t], self.group_rets[t]]).astype(np.float32)


class MetaAllocator:
    """
    Trains and runs the meta-agent that allocates capital across commodity groups.

    Usage:
        allocator = MetaAllocator()
        allocator.train(macro_df, group_returns, total_timesteps=50_000)
        weights = allocator.allocate(current_macro_obs, current_group_signals)
    """

    def __init__(self):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 not installed.")
        self.model = None

    def train(
        self,
        macro_df: pd.DataFrame,
        group_returns: dict[str, pd.Series],
        total_timesteps: int = 50_000,
        verbose: int = 1,
    ):
        env = DummyVecEnv([lambda: MetaAllocationEnv(macro_df, group_returns)])
        self.model = PPO("MlpPolicy", env, verbose=verbose, policy_kwargs={"net_arch": [64, 64]})
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(str(MODEL_DIR / "meta_allocator"))

    def allocate(self, obs: np.ndarray) -> dict[str, float]:
        """Return allocation weights for each commodity group."""
        if self.model is None:
            raise RuntimeError("MetaAllocator not trained. Call .train() first.")
        action, _ = self.model.predict(obs, deterministic=True)
        exp_a = np.exp(action - action.max())
        weights = exp_a / exp_a.sum()
        return dict(zip(GROUPS, weights.tolist()))

    def load(self):
        path = MODEL_DIR / "meta_allocator.zip"
        if not path.exists():
            raise FileNotFoundError(f"No saved meta-allocator at {path}")
        self.model = PPO.load(str(path))
