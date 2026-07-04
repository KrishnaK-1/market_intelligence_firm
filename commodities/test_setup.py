"""
Quick sanity check — run this after pip install gymnasium stable-baselines3 shimmy
to verify the commodities module loads correctly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("Checking imports...")

try:
    import gymnasium
    print(f"  gymnasium {gymnasium.__version__}")
except ImportError:
    print("  gymnasium NOT installed — run: pip install gymnasium")

try:
    import stable_baselines3
    print(f"  stable-baselines3 {stable_baselines3.__version__}")
except ImportError:
    print("  stable-baselines3 NOT installed — run: pip install stable-baselines3")

try:
    from commodities.envs.base_env import CommodityTradingEnv
    from commodities.agents.sub_agent import CommoditySubAgent
    from commodities.meta.allocator import MetaAllocator
    print("  commodities module: OK")
except Exception as e:
    print(f"  commodities module ERROR: {e}")

print("\nModule structure:")
import os
for root, dirs, files in os.walk(os.path.dirname(__file__)):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    level = root.replace(os.path.dirname(__file__), "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in files:
        if not f.startswith("__"):
            print(f"{indent}  {f}")
