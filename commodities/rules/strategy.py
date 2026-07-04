"""
Rules-based commodity desk: vol-targeted trend/carry sleeves combined with
risk parity. No learning, no fitting — every parameter is fixed a priori
(see signals.py), so every day of the evaluation window is out-of-sample
in the sense that nothing was trained on it.

Construction, per sleeve (long/flat, no leverage):
  1. raw_i    = trend_gate_i x carry_mult_i x (1 / ewma_vol_i)
                (inverse-vol sizing so each active asset contributes
                 comparable risk; carry tilt only where a curve pair exists)
  2. w_norm   = raw / sum(raw)          (fully-invested allocation)
  3. scale    = min(1, target_vol / sqrt(w' Sigma w))   (vol targeting —
                can only de-risk, never lever, since this is a cash account)
  4. weights  = w_norm x scale, remainder in cash

Sleeves are combined by inverse trailing vol (floored so an all-cash sleeve
doesn't hog capital), which approximates equal risk contribution given the
low cross-sector correlations of energy/ags/metals.

Accounting matches the RL env exactly: weights decided at close of day t
earn day t+1's return; turnover is charged at 10 bps (TRANSACTION_COST).
"""
import numpy as np
import pandas as pd

from commodities.rules.signals import (
    multi_horizon_trend, ewma_vol, ewma_cov, portfolio_vol,
    curve_spread, carry_multiplier, TRADING_DAYS,
)

TRANSACTION_COST = 0.001   # 10 bps per unit turnover, same as the RL env
SLEEVE_TARGET_VOL = 0.12   # 12% annualized per sleeve (Harvey et al. 10-15% range)
SLEEVE_VOL_WINDOW = 63     # trailing window for risk-parity allocation across sleeves
SLEEVE_VOL_FLOOR = 0.02    # floor so a becalmed/all-cash sleeve doesn't absorb capital
NO_TRADE_BAND = 0.05       # skip rebalances smaller than this L1 weight change


def apply_trade_band(target_w: pd.DataFrame, band: float = NO_TRADE_BAND) -> pd.DataFrame:
    """
    Hold current weights until the target has drifted more than `band` in L1
    distance, then trade fully to target. Kills the daily sub-1% rebalancing
    wiggle from EWMA vol estimates without delaying real signal changes
    (a full gate flip on one asset moves the target far past the band).
    """
    out = np.zeros_like(target_w.values)
    held = target_w.values[0].copy()
    for i, row in enumerate(target_w.values):
        if np.abs(row - held).sum() > band:
            held = row.copy()
        out[i] = held
    return pd.DataFrame(out, index=target_w.index, columns=target_w.columns)


def build_sleeve_weights(
    prices: pd.DataFrame,
    carry_mults: dict = None,
    tilt_mults: pd.DataFrame = None,
    target_vol: float = SLEEVE_TARGET_VOL,
) -> pd.DataFrame:
    """
    Daily target weights for one sleeve. `carry_mults` maps asset name ->
    multiplier Series for assets that have a futures-curve proxy pair.
    `tilt_mults` is a per-asset multiplier frame (e.g. the cross-sectional
    momentum tilt, ranked across the whole desk rather than within-sleeve).
    """
    rets = prices.pct_change()
    trend = multi_horizon_trend(prices)
    vol = ewma_vol(rets)

    raw = trend / vol.replace(0.0, np.nan)
    if carry_mults:
        for asset, mult in carry_mults.items():
            if asset in raw.columns:
                raw[asset] = raw[asset] * mult.reindex(raw.index).ffill().fillna(1.0)
    if tilt_mults is not None:
        raw = raw * tilt_mults.reindex(index=raw.index, columns=raw.columns).fillna(1.0)
    raw = raw.fillna(0.0)

    total = raw.sum(axis=1)
    w_norm = raw.div(total.where(total > 0), axis=0).fillna(0.0)

    cov = ewma_cov(rets)
    vol_est = portfolio_vol(w_norm, cov)
    scale = (target_vol / vol_est.replace(0.0, np.nan)).clip(upper=1.0).fillna(0.0)
    return apply_trade_band(w_norm.mul(scale, axis=0))


def sleeve_gross_returns(weights: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """Pre-cost daily returns of a sleeve (weights lagged one day — the weight
    decided at close of t earns t+1's return)."""
    return (weights.shift(1) * rets[weights.columns]).sum(axis=1)


def risk_parity_alloc(sleeve_rets: pd.DataFrame) -> pd.DataFrame:
    """Inverse trailing-vol allocation across sleeves, normalized to sum to 1."""
    vol = sleeve_rets.rolling(SLEEVE_VOL_WINDOW, min_periods=20).std() * np.sqrt(TRADING_DAYS)
    inv = 1.0 / vol.clip(lower=SLEEVE_VOL_FLOOR)
    alloc = inv.div(inv.sum(axis=1), axis=0)
    # Equal weight until every sleeve has enough history for a vol estimate
    return alloc.fillna(1.0 / sleeve_rets.shape[1])


def combine(
    sleeve_weights: dict,
    all_prices: pd.DataFrame,
    alloc: pd.DataFrame = None,
    cash_returns: pd.Series = None,
) -> dict:
    """
    Aggregate sleeves into one portfolio. If `alloc` is None, sleeves are
    equal-weighted. If `cash_returns` is given (e.g. BIL daily returns), the
    uninvested remainder earns it — the RL env leaves cash at zero yield,
    which materially understates 2023-2025 performance when T-bills paid ~5%
    and the desk was often half in cash. Returns dict with daily net returns,
    final asset weights, and daily turnover.
    """
    all_rets = all_prices.pct_change()
    names = list(sleeve_weights.keys())

    if alloc is None:
        alloc = pd.DataFrame(
            1.0 / len(names), index=all_prices.index, columns=names
        )

    final_w = pd.DataFrame(0.0, index=all_prices.index, columns=all_prices.columns)
    for name in names:
        w = sleeve_weights[name].reindex(all_prices.index).fillna(0.0)
        final_w[w.columns] = final_w[w.columns] + w.mul(alloc[name], axis=0)
    # Band again at the portfolio level: the daily-drifting sleeve allocation
    # would otherwise reintroduce the wiggle the sleeve-level band removed.
    final_w = apply_trade_band(final_w)

    held = final_w.shift(1).fillna(0.0)
    gross = (held * all_rets).sum(axis=1)
    if cash_returns is not None:
        held_cash = (1.0 - held.sum(axis=1)).clip(lower=0.0)
        gross = gross + held_cash * cash_returns.reindex(gross.index).fillna(0.0)
    turnover = (final_w - held).abs().sum(axis=1)
    net = gross - turnover * TRANSACTION_COST

    return {"returns": net, "weights": final_w, "turnover": turnover}


def perf_stats(returns: pd.Series) -> dict:
    """Return/Sharpe/vol/max-drawdown. Sharpe matches sub_agent.run_episode:
    daily simple returns, mean/std x sqrt(252), no risk-free adjustment."""
    r = returns.dropna()
    if len(r) == 0:
        return {"return": 0.0, "sharpe": 0.0, "vol": 0.0, "max_dd": 0.0}
    equity = (1 + r).cumprod()
    total_return = equity.iloc[-1] - 1.0
    sharpe = (r.mean() / (r.std() + 1e-8)) * np.sqrt(TRADING_DAYS)
    vol = r.std() * np.sqrt(TRADING_DAYS)
    max_dd = (equity / equity.cummax() - 1.0).min()
    return {
        "return": float(total_return),
        "sharpe": float(sharpe),
        "vol": float(vol),
        "max_dd": float(max_dd),
    }
