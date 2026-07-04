"""
Sub-agent wrapper around stable-baselines3.
Handles train / predict / save / load for one commodity group environment.

Default algorithm: PPO (on-policy, stable, works well on financial data).
Can swap to SAC or TD3 for off-policy continuous-action control.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import gymnasium as gym

try:
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "saved" / "commodities"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ALGORITHMS = {"ppo": PPO, "sac": SAC, "td3": TD3} if SB3_AVAILABLE else {}


class CommoditySubAgent:
    """
    Wraps a stable-baselines3 agent for one commodity group.

    Args:
        env_class: one of EnergyTradingEnv, AgricultureTradingEnv, MetalsTradingEnv
        algo: "ppo" (default), "sac", or "td3"
        name: used for model save/load path
    """

    def __init__(self, env_class, algo: str = "ppo", name: str = "agent"):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is not installed. Run: pip install stable-baselines3")

        self.name = name
        self.env_class = env_class
        self.algo_name = algo.lower()
        self.algo_cls = ALGORITHMS[self.algo_name]
        self.model = None
        self._env = None

    def train(
        self,
        total_timesteps: int = 100_000,
        env_kwargs: dict = None,
        policy_kwargs: dict = None,
        verbose: int = 1,
    ):
        """Train the agent from scratch. Call this first."""
        env_kwargs = env_kwargs or {}
        policy_kwargs = policy_kwargs or {"net_arch": [128, 128]}

        self._env = DummyVecEnv([lambda: self.env_class(**env_kwargs)])

        self.model = self.algo_cls(
            policy="MlpPolicy",
            env=self._env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            tensorboard_log=str(MODEL_DIR / "logs" / self.name),
        )
        self.model.learn(total_timesteps=total_timesteps)
        self.save()

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return action given an observation vector."""
        if self.model is None:
            raise RuntimeError("Agent not trained or loaded. Call .train() or .load() first.")
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def run_episode(self, env_kwargs: dict = None) -> dict:
        """
        Run one full episode (entire price history) and return performance metrics.
        Use this for backtesting the trained agent.
        """
        env_kwargs = env_kwargs or {}
        env = self.env_class(**env_kwargs)
        obs, _ = env.reset()
        done = False
        portfolio_values = [1.0]
        dates = []

        while not done:
            action = self.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            portfolio_values.append(info["portfolio_value"] / 1_000_000)
            dates.append(info["date"])

        total_return = portfolio_values[-1] - 1.0
        returns = np.diff(portfolio_values)
        sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252)

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "portfolio_values": portfolio_values,
            "dates": dates,
        }

    def save(self):
        path = MODEL_DIR / f"{self.name}_{self.algo_name}"
        self.model.save(str(path))

    def load(self):
        path = MODEL_DIR / f"{self.name}_{self.algo_name}.zip"
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}. Train first.")
        algo_cls = ALGORITHMS[self.algo_name]
        self.model = algo_cls.load(str(path))
