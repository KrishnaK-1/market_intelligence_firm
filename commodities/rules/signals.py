"""
Rules-based signal library for the commodity desk.

Every function here is causal: a value at date t uses only closes up to and
including t. The strategy layer shifts weights by one day before applying
them to returns, so the signal computed at t earns the return from t to t+1.

Parameter choices are literature-standard, fixed a priori, and NOT tuned on
the 2020-2025 evaluation window:
  - Trend windows 50/100/200d: standard CTA multi-horizon blend; 100d matches
    the existing RL env's TREND_WINDOW.
  - EWMA vol half-life 20d: Harvey et al. (2018), "The Impact of Volatility
    Targeting" (JPM 45(1)).
  - Carry lookback 63d (~3 months): trailing front-vs-deferred ETF return
    spread as a curve-slope proxy (cf. Koijen et al. 2018, "Carry", JFE).
"""
import numpy as np
import pandas as pd

TREND_WINDOWS = (50, 100, 200)
GATE_SMOOTH_HALFLIFE = 5
VOL_HALFLIFE = 20
CARRY_LOOKBACK = 63
TRADING_DAYS = 252


def multi_horizon_trend(prices: pd.DataFrame, windows=TREND_WINDOWS,
                        smooth_halflife: int = GATE_SMOOTH_HALFLIFE) -> pd.DataFrame:
    """
    Long/flat trend gate in [0, 1]: the fraction of SMA horizons the price is
    currently above. 1.0 = above all three SMAs (full conviction long),
    0.0 = below all three (flat). Blending horizons avoids the whipsaw of a
    single crossover and is standard CTA construction.

    The raw gate is a step function that flips on ~40% of days, and every flip
    forces a re-normalization of the whole sleeve — measured at ~15%/day of
    weight churn, ~2.5%/yr in costs at 10 bps. A short EWMA turns flips into
    multi-day ramps (cf. Koijen et al.'s carry1-12: smoothing the signal to cut
    turnover, not to change its information content).
    """
    gates = [(prices > prices.rolling(w).mean()).astype(float) for w in windows]
    gate = sum(gates) / len(gates)
    # No signal until the longest SMA has data — stay flat, don't fake conviction
    gate[prices.rolling(max(windows)).mean().isna()] = 0.0
    if smooth_halflife:
        gate = gate.ewm(halflife=smooth_halflife).mean()
    return gate


def ewma_vol(returns: pd.DataFrame, halflife: int = VOL_HALFLIFE) -> pd.DataFrame:
    """Annualized EWMA volatility per asset."""
    return returns.ewm(halflife=halflife, min_periods=10).std() * np.sqrt(TRADING_DAYS)


def curve_spread(front: pd.Series, deferred: pd.Series, lookback: int = CARRY_LOOKBACK) -> pd.Series:
    """
    Trailing total-return spread of front-month ETF over 12-month ETF.
    > 0: front outperforming deferred = backwardation = positive roll yield.
    < 0: contango = roll drag on long positions.
    """
    return front.pct_change(lookback) - deferred.pct_change(lookback)


def carry_multiplier(spread: pd.Series, contango_scale: float = 0.5) -> pd.Series:
    """
    Long/flat carry tilt: full weight in backwardation, scaled-down weight in
    contango (we can't short it, so we underweight it). Neutral (1.0) while
    the spread has insufficient history.
    """
    mult = pd.Series(1.0, index=spread.index)
    mult[spread < 0] = contango_scale
    mult[spread.isna()] = 1.0
    return mult


def xs_momentum_multiplier(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    low: float = 0.5,
    high: float = 1.5,
    smooth_halflife: int = GATE_SMOOTH_HALFLIFE,
) -> pd.DataFrame:
    """
    Cross-sectional momentum tilt: rank every asset by its 12-1 month return
    (12-month lookback, skipping the most recent month to avoid short-term
    reversal — the standard construction, cf. Erb & Harvey 2006) and scale
    weights linearly from `low` (weakest) to `high` (strongest). Long/flat
    version of the classic long-short sort: the trend gate still decides
    in-or-out, this decides relative size among what's in. Assets with
    insufficient history get a neutral 1.0.
    """
    mom = prices.shift(skip).pct_change(lookback - skip)
    rank = mom.rank(axis=1)
    n = mom.notna().sum(axis=1)
    pct = rank.sub(1, axis=0).div((n - 1).where(n > 1), axis=0)
    mult = low + (high - low) * pct
    mult = mult.fillna(1.0)
    if smooth_halflife:
        mult = mult.ewm(halflife=smooth_halflife).mean()
    return mult


def crisis_flag(
    prices: pd.DataFrame,
    vix: pd.Series,
    vix_trigger: float = 35.0,
    vol_z_trigger: float = 3.0,
    short_window: int = 10,
    long_window: int = 252,
) -> pd.Series:
    """
    Deterministic crisis override ported unchanged from the RL env
    (base_env.py): True on days when either broad equity stress (VIX above
    35) or the sleeve's own 10-day realized vol is 3+ standard deviations
    above its trailing 252-day norm. The strategy layer liquidates the
    sleeve to cash while the flag is up.
    """
    basket = prices.pct_change().mean(axis=1)
    realized = basket.rolling(short_window, min_periods=5).std()
    z = (realized - realized.rolling(long_window, min_periods=30).mean()) / (
        realized.rolling(long_window, min_periods=30).std() + 1e-8
    )
    vix_hit = vix.reindex(prices.index).ffill() > vix_trigger
    return (z > vol_z_trigger) | vix_hit.fillna(False)


def ewma_cov(returns: pd.DataFrame, halflife: int = VOL_HALFLIFE) -> pd.DataFrame:
    """EWMA covariance matrices (annualized), MultiIndex (date, asset)."""
    return returns.ewm(halflife=halflife, min_periods=10).cov() * TRADING_DAYS


def portfolio_vol(weights: pd.DataFrame, cov: pd.DataFrame) -> pd.Series:
    """
    Annualized portfolio vol sqrt(w' Σ w) per date, given a weight matrix
    (dates x assets) and the MultiIndex covariance from ewma_cov.

    Only assets with nonzero weight enter the calculation, so an asset whose
    ETF doesn't exist yet (NaN covariance, zero weight) never poisons the
    sleeve's vol estimate — required for backtests that start before every
    ticker's inception.
    """
    vols = pd.Series(np.nan, index=weights.index)
    for dt in weights.index:
        w_row = weights.loc[dt]
        active = w_row[w_row != 0.0].index
        if len(active) == 0:
            vols.loc[dt] = 0.0
            continue
        try:
            sigma = cov.loc[dt].reindex(index=active, columns=active)
        except KeyError:
            continue
        if np.isnan(sigma.values).any():
            continue
        w = w_row[active].values
        vols.loc[dt] = np.sqrt(max(0.0, w @ sigma.values @ w))
    return vols
