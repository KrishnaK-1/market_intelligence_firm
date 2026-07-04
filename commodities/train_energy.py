"""
Train the energy sub-agent on an in-sample window and evaluate on an unseen
out-of-sample window, so the reported performance reflects generalization
rather than memorization of the training data.

Train window:      2015-01-01 to 2022-12-31  (8 years)
Test window (OOS):  2023-01-01 to present     (~3.5 years, never seen in training)

Run: python commodities/train_energy.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commodities.envs.energy_env import EnergyTradingEnv
from commodities.agents.sub_agent import CommoditySubAgent

TRAIN_START, TRAIN_END = "2015-01-01", "2022-12-31"
TEST_START = "2023-01-01"

if __name__ == "__main__":
    agent = CommoditySubAgent(EnergyTradingEnv, algo="ppo", name="energy")
    print(f"Training energy agent on {TRAIN_START} to {TRAIN_END} (200,000 timesteps)...")
    agent.train(
        total_timesteps=200_000,
        env_kwargs={"start": TRAIN_START, "end": TRAIN_END},
    )

    print("\n--- IN-SAMPLE (training window) ---")
    in_sample = agent.run_episode(env_kwargs={"start": TRAIN_START, "end": TRAIN_END})
    print(f"Total return: {in_sample['total_return']:.1%}")
    print(f"Sharpe ratio:  {in_sample['sharpe']:.2f}")

    print(f"\n--- OUT-OF-SAMPLE ({TEST_START} to present, never seen during training) ---")
    out_sample = agent.run_episode(env_kwargs={"start": TEST_START})
    print(f"Total return: {out_sample['total_return']:.1%}")
    print(f"Sharpe ratio:  {out_sample['sharpe']:.2f}")
    print(f"Final portfolio value (x initial): {out_sample['portfolio_values'][-1]:.3f}")
