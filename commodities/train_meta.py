"""
Train the meta-allocator on the stitched out-of-sample sub-agent returns,
with its own train/test split (2020-2023 train, 2024-2025 test) so the
meta-layer's evaluation is held to the same standard as the sub-agents.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from commodities.meta.allocator import MetaAllocator, MetaAllocationEnv, GROUPS

DATA_PATH = Path(__file__).resolve().parent / "walkforward_results" / "meta_dataset.pkl"
SPLIT_DATE = "2024-01-01"


def evaluate(allocator: MetaAllocator, macro_df: pd.DataFrame, returns_df: pd.DataFrame, label: str):
    env = MetaAllocationEnv(macro_df, {g: returns_df[g] for g in GROUPS})
    obs, _ = env.reset()
    done = False
    port_values = [1.0]
    equal_wt_values = [1.0]

    while not done:
        action = allocator.allocate(obs)
        action_arr = np.array([action[g] for g in GROUPS])
        obs, reward, terminated, truncated, info = env.step(action_arr)
        done = terminated or truncated
        port_values.append(port_values[-1] * (1 + info["step_return"]))
        equal_wt_values.append(equal_wt_values[-1] * (1 + returns_df.iloc[env.t - 1].mean()))

    total_ret = port_values[-1] - 1.0
    rets = np.diff(port_values)
    sharpe = (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252)
    ew_ret = equal_wt_values[-1] - 1.0

    print(f"\n--- {label} ---")
    print(f"Meta-allocator: return {total_ret:+.1%}  sharpe {sharpe:+.2f}")
    print(f"Equal-weight (1/3 each) baseline: return {ew_ret:+.1%}")


if __name__ == "__main__":
    data = pd.read_pickle(DATA_PATH)
    returns_df, macro_df = data["returns"], data["macro"]

    train_mask = returns_df.index < SPLIT_DATE
    train_returns, test_returns = returns_df[train_mask], returns_df[~train_mask]
    train_macro, test_macro = macro_df[train_mask], macro_df[~train_mask]
    print(f"Train: {train_returns.index.min().date()} to {train_returns.index.max().date()} ({len(train_returns)} days)")
    print(f"Test:  {test_returns.index.min().date()} to {test_returns.index.max().date()} ({len(test_returns)} days)")

    allocator = MetaAllocator()
    print("\nTraining meta-allocator...")
    allocator.train(train_macro, {g: train_returns[g] for g in GROUPS}, total_timesteps=50_000, verbose=0)

    evaluate(allocator, train_macro, train_returns, "IN-SAMPLE (2020-2023)")
    evaluate(allocator, test_macro, test_returns, "OUT-OF-SAMPLE (2024-2025)")
