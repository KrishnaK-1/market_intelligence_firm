"""
Base gymnasium environment for commodity trading.
Each sub-agent environment inherits this and specifies its own tickers.

State space  : [normalized prices (n), returns (n), volatility (n), trend signal (n),
                 position (n), portfolio_value, macro (m)]
Action space : continuous weights in [-1, 1] per TRADEABLE asset (negative = short)
Reward       : log return of portfolio, penalized by turnover cost

Note: macro/context columns (e.g. DXY, OVX, 10Y yield) are observed but NOT
tradeable — they inform the agent's decisions without being assets it can
take a position in. Keeping them out of the action space prevents the agent
from "trading" things like a volatility index that has no direct ETF.

Trend signal: price relative to its trend_window-day moving average. Commodity
markets (unlike equities) have decades of evidence that trend-following is a
persistent, harvestable style (this is the core signal behind managed futures
/ CTA funds), so it's included explicitly rather than left for the agent to
rediscover from raw prices alone.

Crisis override: a deterministic (non-learned) safety rule that forces full
liquidation to cash when either (a) broad market stress (VIX) is elevated,
or (b) the asset basket's OWN realized volatility spikes well above its
trailing norm. (b) exists because VIX is a equity-market proxy and misses
sector-idiosyncratic shocks — e.g. the 2022 wheat/Ukraine-war spike in
agriculture prices happened without VIX ever reaching crisis territory.
A self-referential vol trigger catches "this basket itself is in turmoil"
regardless of whether the broader market agrees. This exists because RL
cannot reliably learn rare-event behavior from a handful of historical
crises — each walk-forward fold trains a fresh agent that may have zero
exposure to anything resembling a crash in its training window. Rather than
hoping the policy discovers "high VIX -> sell," the rule is hard-coded, the
same way the existing momentum strategy's 200-day MA filter is hard-coded
rather than learned.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


TRANSACTION_COST  = 0.001  # 10 bps per trade (round-trip)
INITIAL_CAPITAL   = 1_000_000.0
LOOKBACK          = 20    # days of short-term history in state
TREND_WINDOW      = 100   # days for the trend-following signal
MACRO_NORM_WINDOW = 252   # days for causal z-score normalization of macro features
VIX_LEVEL_TRIGGER = 35.0  # VIX above this = true crisis territory (30-35 is just elevated nervousness,
                          # too common to justify forced liquidation — only fire on genuine tail events)
BASKET_VOL_SHORT_WINDOW = 10   # days for short-term realized vol of the basket
BASKET_VOL_LONG_WINDOW  = 252  # days for the trailing distribution the short vol is z-scored against
BASKET_VOL_Z_TRIGGER    = 3.0  # basket vol must be ~3 std devs above its own trailing norm to fire


class CommodityTradingEnv(gym.Env):
    """
    Generic commodity trading environment.

    Args:
        prices: DataFrame of daily close prices for TRADEABLE assets, columns = asset names
        macro: optional DataFrame of context-only features (same index as prices),
               observed by the agent but excluded from the action space and
               portfolio return calculation
        lookback: number of past days included in the observation
        allow_short: whether negative positions are permitted
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: pd.DataFrame,
        macro: pd.DataFrame = None,
        lookback: int = LOOKBACK,
        trend_window: int = TREND_WINDOW,
        allow_short: bool = False,
        crisis_override: bool = True,
        vix_level_trigger: float = VIX_LEVEL_TRIGGER,
        basket_vol_z_trigger: float = BASKET_VOL_Z_TRIGGER,
    ):
        super().__init__()
        self.prices = prices.values.astype(np.float32)
        self.dates  = prices.index
        self.assets = list(prices.columns)
        self.n      = len(self.assets)
        self.lookback = lookback
        self.trend_window = trend_window
        self.allow_short = allow_short

        # Basket's own realized-vol z-score: causal (rolling windows only look
        # backward), precomputed once since it's the same regardless of policy.
        basket_rets = np.diff(self.prices, axis=0, prepend=self.prices[:1]) / (self.prices + 1e-8)
        basket_rets = basket_rets.mean(axis=1)  # equal-weight combined return series
        basket_series = pd.Series(basket_rets)
        realized_vol  = basket_series.rolling(BASKET_VOL_SHORT_WINDOW, min_periods=5).std()
        vol_mean      = realized_vol.rolling(BASKET_VOL_LONG_WINDOW, min_periods=30).mean()
        vol_std       = realized_vol.rolling(BASKET_VOL_LONG_WINDOW, min_periods=30).std()
        vol_z         = (realized_vol - vol_mean) / (vol_std + 1e-8)
        self.basket_vol_z = vol_z.fillna(0.0).values.astype(np.float32)
        self.basket_vol_z_trigger = basket_vol_z_trigger

        if macro is not None:
            macro = macro.reindex(prices.index).ffill().bfill()
            self.macro = macro.values.astype(np.float32)
            self.macro_names = list(macro.columns)
        else:
            self.macro = np.zeros((len(self.prices), 0), dtype=np.float32)
            self.macro_names = []
        self.m = self.macro.shape[1]

        # Crisis override: hard rule, not learned. Only active if VIX is available in macro.
        self.crisis_override = crisis_override
        self.vix_level_trigger = vix_level_trigger
        self.vix_idx = self.macro_names.index("VIX") if "VIX" in self.macro_names else None

        # Action: portfolio weight for each TRADEABLE asset only [-1, 1] (or [0,1] if no shorting)
        low  = -1.0 if allow_short else 0.0
        self.action_space = spaces.Box(low=low, high=1.0, shape=(self.n,), dtype=np.float32)

        # Observation: lookback x n price features + trend signal (n) + positions
        # + portfolio value + basket vol z-score (1) + macro
        obs_dim = lookback * self.n * 3 + self.n + self.n + 1 + 1 + self.m
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    # ── Gymnasium API ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t           = self.lookback
        self.portfolio   = INITIAL_CAPITAL
        self.positions   = np.zeros(self.n, dtype=np.float32)  # fraction of portfolio per asset
        self.prev_prices = self.prices[self.t - 1]
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        crisis = self._crisis_triggered()
        if crisis:
            action = np.zeros_like(action)  # hard liquidation to cash, overrides the policy

        # Normalize so weights sum to ≤ 1 (remaining goes to cash)
        total = np.sum(np.abs(action))
        if total > 1.0:
            action = action / total

        # Transaction cost on weight changes
        turnover   = np.sum(np.abs(action - self.positions))
        cost       = turnover * TRANSACTION_COST

        # Portfolio return
        curr_prices = self.prices[self.t]
        rets        = (curr_prices - self.prev_prices) / (self.prev_prices + 1e-8)
        port_ret    = float(np.dot(self.positions, rets)) - cost

        self.portfolio   *= (1 + port_ret)
        self.positions    = action.copy()
        self.prev_prices  = curr_prices
        self.t           += 1

        reward      = np.log1p(port_ret)  # log return as reward signal
        terminated  = self.t >= len(self.prices) - 1
        truncated   = False
        info        = {
            "portfolio_value": self.portfolio,
            "step_return": port_ret,
            "date": str(self.dates[self.t - 1]),
            "crisis_override": crisis,
        }
        return self._obs(), reward, terminated, truncated, info

    def _crisis_triggered(self) -> bool:
        """Deterministic safety rule: force liquidation when either broad market
        stress (VIX) or the basket's own realized vol is at crisis levels."""
        if not self.crisis_override:
            return False
        vix_trigger = False
        if self.vix_idx is not None:
            vix_now = float(self.macro[self.t, self.vix_idx])
            vix_trigger = vix_now > self.vix_level_trigger
        vol_trigger = bool(self.basket_vol_z[min(self.t, len(self.basket_vol_z) - 1)] > self.basket_vol_z_trigger)
        return vix_trigger or vol_trigger

    def render(self):
        print(f"t={self.t} | portfolio=${self.portfolio:,.0f} | positions={dict(zip(self.assets, self.positions.round(3)))}")

    # ── Internal ────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        window = self.prices[self.t - self.lookback : self.t]  # (lookback, n)

        # Normalize prices relative to first day of window
        norm_prices = window / (window[0] + 1e-8) - 1.0

        # Rolling returns
        rets = np.diff(window, axis=0, prepend=window[:1]) / (window + 1e-8)

        # Rolling std (20-day vol)
        vols = np.std(rets, axis=0, keepdims=True).repeat(self.lookback, axis=0)

        # Trend signal: current price vs trend_window-day moving average (ramps up early on)
        trend_start = max(0, self.t - self.trend_window)
        trend_hist  = self.prices[trend_start : self.t]
        trend_sma   = trend_hist.mean(axis=0)
        trend_signal = self.prices[self.t - 1] / (trend_sma + 1e-8) - 1.0

        vol_z = self.basket_vol_z[min(self.t - 1, len(self.basket_vol_z) - 1)]

        parts = [
            norm_prices.flatten(),
            rets.flatten(),
            vols.flatten(),
            trend_signal,
            self.positions,
            [self.portfolio / INITIAL_CAPITAL],
            [vol_z],
        ]
        if self.m > 0:
            parts.append(self._normalized_macro())

        return np.concatenate(parts).astype(np.float32)

    def _normalized_macro(self) -> np.ndarray:
        """
        Causal rolling z-score of macro features (uses only data up to and including
        the current step, never future data, so there's no look-ahead leakage).
        Raw macro features span wildly different scales (VIX ~10-80, 10Y yield ~0.5-5,
        DXY ETF price ~20-30), which without normalization can cause gradient-based
        training to effectively underweight some features relative to others.
        """
        t = min(self.t, len(self.macro) - 1)
        start = max(0, t - MACRO_NORM_WINDOW)
        hist = self.macro[start : t + 1]
        mean = hist.mean(axis=0)
        std  = hist.std(axis=0) + 1e-8
        return (self.macro[t] - mean) / std
