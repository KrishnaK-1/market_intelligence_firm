"""
Walk-forward validation for a commodity sub-agent — runs ONE fold per invocation
(so each run stays well under any execution time limit) and appends the result
to a JSON file. Call repeatedly with increasing --fold to build up the full
walk-forward report, then use summarize() to view it.

Folds (expanding window, one unseen test year each):
  0: train <=2019-12-31 -> test 2020   (includes COVID crash — useful stress test)
  1: train <=2020-12-31 -> test 2021
  2: train <=2021-12-31 -> test 2022
  3: train <=2022-12-31 -> test 2023
  4: train <=2023-12-31 -> test 2024
  5: train <=2024-12-31 -> test 2025

Usage: python commodities/walkforward.py energy 0
       python commodities/walkforward.py energy 1
       ... etc, then:
       python commodities/walkforward.py energy summarize
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from commodities.agents.sub_agent import CommoditySubAgent

TRAIN_START = "2015-01-01"
TIMESTEPS_PER_FOLD = 150_000
RESULTS_DIR = Path(__file__).resolve().parent / "walkforward_results"
RESULTS_DIR.mkdir(exist_ok=True)

FOLDS = [
    ("2019-12-31", "2020-01-01", "2020-12-31"),
    ("2020-12-31", "2021-01-01", "2021-12-31"),
    ("2021-12-31", "2022-01-01", "2022-12-31"),
    ("2022-12-31", "2023-01-01", "2023-12-31"),
    ("2023-12-31", "2024-01-01", "2024-12-31"),
    ("2024-12-31", "2025-01-01", "2025-12-31"),
]


def get_env_class(group: str):
    if group == "energy":
        from commodities.envs.energy_env import EnergyTradingEnv as EnvCls
    elif group == "agriculture":
        from commodities.envs.agriculture_env import AgricultureTradingEnv as EnvCls
    elif group == "metals":
        from commodities.envs.metals_env import MetalsTradingEnv as EnvCls
    else:
        raise ValueError(f"Unknown group: {group}")
    return EnvCls


def results_path(group: str) -> Path:
    return RESULTS_DIR / f"{group}.json"


def run_fold(group: str, fold_idx: int):
    EnvCls = get_env_class(group)
    train_end, test_start, test_end = FOLDS[fold_idx]

    print(f"=== {group} fold {fold_idx}: train {TRAIN_START}->{train_end} | test {test_start}->{test_end} ===")
    agent = CommoditySubAgent(EnvCls, algo="ppo", name=f"{group}_wf_{test_start[:4]}")
    agent.train(
        total_timesteps=TIMESTEPS_PER_FOLD,
        env_kwargs={"start": TRAIN_START, "end": train_end},
        verbose=0,
    )
    oos = agent.run_episode(env_kwargs={"start": test_start, "end": test_end})
    result = {
        "fold": fold_idx,
        "year": test_start[:4],
        "return": float(oos["total_return"]),
        "sharpe": float(oos["sharpe"]),
    }
    print(f"  OOS {result['year']} -> return: {result['return']:+.1%}  sharpe: {result['sharpe']:+.2f}")

    path = results_path(group)
    existing = json.loads(path.read_text()) if path.exists() else []
    existing = [r for r in existing if r["fold"] != fold_idx]  # replace if rerun
    existing.append(result)
    existing.sort(key=lambda r: r["fold"])
    path.write_text(json.dumps(existing, indent=2))


def summarize(group: str):
    path = results_path(group)
    if not path.exists():
        print(f"No results yet for {group}")
        return
    results = json.loads(path.read_text())
    returns = [r["return"] for r in results]
    sharpes = [r["sharpe"] for r in results]

    print(f"\n========== WALK-FORWARD SUMMARY: {group.upper()} ({len(results)}/{len(FOLDS)} folds) ==========")
    for r in results:
        print(f"  {r['year']}: return {r['return']:+7.1%}   sharpe {r['sharpe']:+.2f}")
    print("-" * 60)
    if returns:
        print(f"  Mean OOS return: {np.mean(returns):+.1%}   (std: {np.std(returns):.1%})")
        print(f"  Mean OOS Sharpe: {np.mean(sharpes):+.2f}   (std: {np.std(sharpes):.2f})")
        print(f"  Years profitable: {sum(1 for r in returns if r > 0)}/{len(returns)}")


if __name__ == "__main__":
    group = sys.argv[1] if len(sys.argv) > 1 else "energy"
    arg2 = sys.argv[2] if len(sys.argv) > 2 else "0"

    if arg2 == "summarize":
        summarize(group)
    else:
        run_fold(group, int(arg2))
