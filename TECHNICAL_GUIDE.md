# Market Intelligence Firm -- Technical Architecture & Maintenance Guide

## System Overview

A multi-manager investment decision platform that uses an ensemble of machine learning models (HMM + GMM + XGBoost) to classify economic regimes and recommend asset allocations. The system includes a self-optimizing feedback loop that validates regime classifications against historical data and adjusts model parameters automatically.

---

## How It Runs

```
You click Run in PyCharm
    |
    v
uvicorn (web server) starts
    |
    v
dashboard/server.py loads
    |
    v
main.py -> MarketIntelligenceFirm.run_full_pipeline()
    |
    |-- Step 1: data/ingestion.py pulls FRED, Shiller, Yahoo data
    |-- Step 2: data/ingestion.py builds feature matrix (24 features)
    |-- Step 3: models/regime_classifier.py trains HMM + GMM + XGBoost
    |-- Step 4: agents/economic_manager.py produces regime signals + allocations
    |-- Step 5: alerts/engine.py checks for regime shift warnings
    |-- Step 5b: models/feedback.py validates against history + optimizes params
    |-- Step 5c: Re-trains with optimized params if improvement found
    |-- Step 6: backtest/engine.py runs walk-forward backtest
    |
    v
Dashboard serves at localhost:8000
    |
    |-- Browser loads dashboard/index.html
    |-- JavaScript calls /api/report endpoint
    |-- FastAPI returns JSON, charts render
```

---

## Module Reference

### main.py (Orchestrator)

**Purpose:** Coordinates the entire pipeline. This is the brain that calls every other module in sequence.

**What it does:**
- Initializes all components (managers, alert engine, backtest engine, feedback loop)
- Runs the 6-step pipeline on startup and when Retrain is clicked
- Loads optimized parameters from disk before training
- After optimization, re-trains the model with better params
- Assembles the final report for the dashboard

**Key class:** `MarketIntelligenceFirm`
- `run_full_pipeline()` -- Full run with training, validation, optimization, backtest
- `run_inference_only()` -- Quick daily refresh, no retraining
- `force_refresh()` -- Clears cache and does full retrain

**Interacts with:** Every other module. This is the central hub.

---

### config/settings.py (Configuration)

**Purpose:** All configuration in one place. No magic numbers scattered across code.

**What it contains:**
- `FRED_API_KEY` -- loaded from .env file
- `N_REGIMES = 5` -- number of economic states (Expansion, Slowdown, Contraction, Recovery, Crisis)
- `REGIME_LABELS` -- mapping of state index to name
- `REGIME_COLORS` -- colors for dashboard charts
- `FRED_SERIES` -- dictionary of 19 FRED economic series to download
- `EQUITY_SECTORS`, `FIXED_INCOME`, `COMMODITIES`, `INTERNATIONAL` -- ETF tickers for investment universe
- `REGIME_ALLOCATIONS` -- asset class weights per regime (high level)
- `REGIME_SUB_ALLOCATIONS` -- specific ETF weights per regime (granular)
- `SERVER_PORT`, `REFRESH_HOUR`, `TIMEZONE` -- server settings

**Maintenance:** When you want to add a new FRED series, ETF ticker, or change allocation templates, edit this file.

---

### data/ingestion.py (Data Layer)

**Purpose:** Pulls raw data from three sources and engineers features for the model.

**Data sources:**
1. **FRED API** -- 19 macroeconomic time series (GDP, unemployment, CPI, yield curves, VIX, etc.) going back to 1920
2. **Shiller CAPE** -- Read from local file `data/ie_data.xls` (1871-present). Monthly S&P valuations and CAPE ratio
3. **Yahoo Finance** -- Daily prices for 26 ETFs (S&P sectors, bonds, gold, international), converted to monthly

**Key functions:**
- `fetch_all_fred()` -- Downloads all FRED series, resamples to monthly, caches to `data/cache/fred_raw.pkl`
- `fetch_shiller_cape()` -- Reads local Excel file, parses fractional dates, caches to `data/cache/shiller_cape.pkl`
- `fetch_yahoo_prices()` -- Downloads ETF prices via yfinance, caches to `data/cache/yahoo_prices.pkl`
- `build_feature_matrix()` -- Combines FRED + Shiller into 24+ engineered features (yield spreads, growth rates, momentum, etc.)
- `clear_cache()` -- Deletes all .pkl cache files to force fresh download

**Cache behavior:** Each data source caches for ~20 hours. If you run the server twice in a day, the second run loads from cache. `clear_cache()` forces a fresh pull.

**Feature list (24 columns):**
- `spread_10y2y`, `spread_10y3m` -- Yield curve spreads
- `credit_spread`, `credit_spread_chg_3m` -- Baa-10Y spread and its 3-month change
- `unrate`, `unrate_chg_3m`, `unrate_chg_12m` -- Unemployment rate and acceleration
- `indpro_yoy`, `indpro_mom` -- Industrial production growth
- `cpi_yoy` -- Inflation
- `houst_yoy` -- Housing starts momentum
- `payems_yoy`, `payems_mom` -- Payroll growth
- `umcsent`, `umcsent_chg` -- Consumer sentiment level and change
- `icsa_4wk_chg` -- Initial claims trend
- `fedfunds`, `fedfunds_chg_12m` -- Fed funds rate and trajectory
- `m2_yoy` -- Money supply growth
- `sp500_yoy`, `sp500_mom_3m` -- Equity market momentum
- `vix` -- Volatility index
- `cape`, `cape_zscore` -- Shiller CAPE ratio and z-score
- `recession` -- NBER recession indicator (training target)

**Maintenance:** To add a new feature, add the FRED series ID to `FRED_SERIES` in settings.py, then add the transformation logic in `build_feature_matrix()`.

---

### models/regime_classifier.py (ML Engine)

**Purpose:** The core ML model. Trains three models and combines them into an ensemble.

**Class:** `EnsembleRegimeClassifier`

**Three models:**
1. **HMM (Gaussian Hidden Markov Model, 5 states)** -- Captures sequential regime dynamics. Learns that expansions tend to follow recoveries, contractions follow slowdowns, etc. Provides the transition probability matrix used for forecasting.
2. **GMM (Gaussian Mixture Model, 5 components)** -- Captures distributional clustering. Groups months by their feature similarity without considering time sequence.
3. **XGBoost (multi-class classifier, 5 targets)** -- Supervised learning. Trained on 5-class labels derived from NBER recession data + economic logic. Dominates the ensemble (~50% weight).

**How labels are created (`_create_5class_labels`):**
- State 0 (Expansion): Not in recession, no deteriorating signals
- State 1 (Slowdown): Not in recession, but 3+ of 10 signals are deteriorating (yield curve inverted, unemployment rising, sentiment falling, markets declining, credit widening, etc.)
- State 2 (Contraction): NBER recession indicator = 1
- State 3 (Recovery): First 6 months after a recession ends
- State 4 (Crisis): Recession + extreme VIX or credit spread

**Critical design: Labels use UNSCALED features.** The thresholds (e.g., yield curve < -0.2%) compare against real economic values, not z-scores. The HMM/GMM/XGBoost train on scaled features.

**Parameterized labels:** All thresholds in `_create_5class_labels` are read from `self._label_params`, which is injected by the optimizer. This allows the feedback loop to tune sensitivity.

**10 slowdown signals (any 3+ triggers Slowdown):**
1. Yield curve deeply inverted (spread < threshold)
2. Industrial production declining
3. Consumer sentiment deteriorating sharply
4. Payrolls declining
5. S&P 500 YoY negative
6. Credit spreads elevated
7. Unemployment rate accelerating (NEW)
8. S&P 500 3-month momentum negative (NEW)
9. Credit spreads widening rapidly (NEW)
10. Consumer sentiment below historical average (NEW)

**Key methods:**
- `fit(features)` -- Trains all three models
- `predict_proba(X)` -- Returns regime probabilities (weighted ensemble)
- `predict(X)` -- Returns most likely regime name
- `forecast_regimes(current_proba, horizons)` -- Projects regime probabilities 1/3/6 months ahead using HMM transition matrix
- `compute_confidence(X)` -- Measures model agreement and prediction entropy
- `feature_importance()` -- Returns XGBoost feature rankings
- `save() / load()` -- Persists to/from `models/saved/ensemble_regime_classifier.pkl`

**State alignment:** HMM and GMM produce arbitrary state numbers (0-4) that don't correspond to our regime labels. The Hungarian algorithm (`_compute_state_mapping`) finds the optimal mapping from unsupervised states to canonical labels.

---

### models/feedback.py (Self-Optimization)

**Purpose:** Validates regime classifications against known economic history and optimizes model parameters. Persists improvements to disk so each retrain starts from a better baseline.

**Two classes:**

**`ParameterOptimizer`** -- Manages the parameter values
- Loads saved params from `models/saved/optimized_params.json` on init
- Saves improved params to disk when a better score is found
- Ensures new parameters are backward-compatible with old saved files

**`FeedbackLoop`** -- Runs the validation and optimization cycle
- `_validate_ground_truth()` -- Checks if NBER recession months are called Contraction/Crisis (should be >80%) and known expansion months are called Expansion/Recovery (should be >80%)
- `_validate_returns()` -- Checks if S&P 500 YoY returns match what you'd expect for each regime (e.g., Expansion should have positive returns)
- `_run_optimization()` -- Tries 30-40 random parameter combinations, trains a model with each, validates, keeps the best. **Rejects any parameter set that eliminates a regime class** (minimum label counts enforced)
- `_check_label_distribution()` -- Enforces minimum counts: Expansion >= 350, Slowdown >= 40, Contraction >= 60, Recovery >= 25, Crisis >= 8

**Search space (parameters the optimizer tunes):**
- `yield_curve_threshold`: How inverted must the curve be (-1.0 to 0.5)
- `indpro_threshold`: How weak must industrial production be (-0.5 to 2.0)
- `slowdown_signal_count`: How many signals needed (2 to 5)
- `sentiment_threshold`: How much must sentiment drop (-12 to -3)
- `payems_threshold`: Payroll decline threshold (-0.1 to 0.0)
- `unrate_accel_threshold`: Unemployment acceleration trigger (0.1 to 0.5)
- `sp500_mom3m_threshold`: 3-month market decline trigger (-10 to 0)
- `credit_accel_threshold`: Credit spread widening speed (0.1 to 0.5)
- `umcsent_level_threshold`: Absolute sentiment level (55 to 75)
- `recovery_months`: How long recovery lasts after recession (4 to 8)

**Persistence files:**
- `models/saved/optimized_params.json` -- Current best parameters
- `models/saved/optimization_history.json` -- History of all optimization runs

---

### agents/economic_manager.py (Agent Layer)

**Purpose:** Translates model outputs into investment decisions. Implements the multi-manager architecture.

**Three classes:**

**`BaseManager`** (abstract) -- Standard interface all managers must implement:
- `get_signals()` -- Named signals with values
- `get_allocations()` -- Ticker weights summing to 1.0
- `get_confidence()` -- Score from 0.0 to 1.0
- `get_report()` -- Full analysis for dashboard/CIO
- `update()` -- Refresh with new data

**`EconomicRegimeManager`** -- First (and currently only) manager:
- Runs the regime classifier on the feature matrix
- Produces probability-weighted allocations across all 5 regimes
- Blends current + 1-month + 3-month forecast probabilities (50/30/20 weighting)
- Provides asset class and individual ticker allocations
- Computes transition probability matrix

**`CIOAggregator`** -- Combines signals from all registered managers:
- Currently only has Economic Regime Manager
- Designed for future: register Equity Manager, Risk Manager, etc.
- Combines allocations weighted by each manager's confidence score

**Hybrid architecture:** Managers can read Economic signals (`economic_manager.get_signals()`) but each reports independently to the CIO. No manager can override another.

---

### alerts/engine.py (Early Warning System)

**Purpose:** Monitors for regime transition signals and generates alerts.

**Alert types:**
- **Regime Transition Detected** (CRITICAL) -- Regime changed since last evaluation
- **Regime Uncertainty Elevated** (WARNING) -- No single regime has >45% probability
- **High Model Disagreement** (WARNING) -- HMM, GMM, XGBoost disagree on top regime
- **Rapid Probability Shift** (WARNING) -- Any regime probability changed >20% in 3 months
- **Forecast Divergence** (INFO) -- Forward forecast differs from current regime
- **Crisis Probability Elevated** (CRITICAL) -- Crisis probability >15%
- **Crisis Risk Rising** (CRITICAL) -- Crisis probability increasing at forward horizons

---

### backtest/engine.py (Performance Validation)

**Purpose:** Walk-forward backtesting of the regime strategy against actual ETF returns.

**How it works:**
1. Aligns feature matrix with ETF return data (overlap starts ~1993)
2. Splits into 5 walk-forward windows
3. For each split: trains model on past data, predicts regimes on future data, computes portfolio returns using regime-based allocations
4. Computes: annualized return, volatility, Sharpe ratio, max drawdown, Calmar ratio, win rate

**Current limitation:** Splits 0-3 fail because the feature matrix has NaN rows that the scaler can't handle. Only split 4 (2021-2025) runs successfully. This is a known issue for future improvement.

---

### dashboard/server.py (Web Server)

**Purpose:** FastAPI application that serves the dashboard and API endpoints.

**Startup behavior:**
1. Creates `MarketIntelligenceFirm` instance
2. Runs full pipeline (data + train + validate + backtest)
3. Schedules daily auto-refresh at 7 AM Eastern
4. Starts serving HTTP on port 8000

**API endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| GET | / | Dashboard HTML page |
| GET | /api/health | Server status |
| GET | /api/report | Complete JSON report |
| GET | /api/regime | Current regime + probabilities |
| GET | /api/allocations | Recommended allocations |
| GET | /api/forecasts | 1/3/6 month forecasts |
| GET | /api/alerts | Active alerts |
| GET | /api/backtest | Backtest results |
| GET | /api/transition | Transition matrix |
| GET | /api/confidence | Model confidence + feature importance |
| POST | /api/refresh | Manual data refresh (inference only) |
| POST | /api/retrain | Full retrain with optimization |
| GET | /docs | Swagger API documentation (auto-generated) |

**Auto-generated docs:** FastAPI automatically creates interactive API documentation at `localhost:8000/docs`.

---

### dashboard/index.html (Frontend)

**Purpose:** Single-page interactive dashboard using Chart.js for visualizations.

**Panels:**
- Current Regime with probability bars
- Regime Forecasts table + line chart (1/3/6 month)
- Alerts & Warnings
- Recommended Allocation with doughnut chart + ticker weights
- Model Confidence meter + individual model votes + ensemble weights + feature importance bars
- Historical Regime Timeline (stacked bar chart, 1952-present)
- Regime Transition Probability Matrix (heatmap table)
- Walk-Forward Backtest Results with cumulative return chart

**Auto-refresh:** Polls `/api/report` every 5 minutes. Manual refresh and retrain buttons call POST endpoints.

---

## Data Flow Diagram

```
FRED API (19 series, 1920-present)
Shiller Excel (CAPE, 1871-present)      --> data/ingestion.py
Yahoo Finance (26 ETFs, 1993-present)        |
                                             v
                                    Feature Matrix (881 months x 24 features)
                                             |
                    +------------------------+------------------------+
                    |                        |                        |
                    v                        v                        v
              HMM (5 states)          GMM (5 components)      XGBoost (5 class)
                    |                        |                        |
                    +------------------------+------------------------+
                                             |
                                    Ensemble Probabilities
                                             |
                    +------------------------+------------------------+
                    |                        |                        |
                    v                        v                        v
           Regime Forecasts          Asset Allocations          Alerts
           (HMM transitions)         (probability-weighted)     (threshold checks)
                    |                        |                        |
                    +------------------------+------------------------+
                                             |
                                    Feedback Validation
                                    (NBER + returns check)
                                             |
                                    Parameter Optimization
                                    (30 iterations, saves to disk)
                                             |
                                    Re-train with best params
                                             |
                                    Dashboard + API
```

---

## File Locations

| Path | Contents | Persistent? |
|------|----------|------------|
| `data/cache/` | Cached FRED, Shiller, Yahoo data (.pkl) | No - regenerated on each pull |
| `data/ie_data.xls` | Shiller CAPE Excel file | Yes - update monthly from Shiller's website |
| `models/saved/ensemble_regime_classifier.pkl` | Trained model | Yes - regenerated on retrain |
| `models/saved/optimized_params.json` | Best optimization parameters | Yes - accumulates improvements |
| `models/saved/optimization_history.json` | History of optimization runs | Yes - for tracking convergence |
| `logs/firm.log` | Runtime logs | Yes - grows over time |
| `.env` | FRED API key | Yes - never commit to git |

---

## Maintenance Checklist

**Daily:** Nothing needed. Auto-refresh at 7 AM pulls fresh data and re-scores.

**Monthly:**
- Download latest `ie_data.xls` from Shiller's website and replace in `data/` folder
- Click "Retrain" on dashboard to trigger full optimization cycle

**When adding a new FRED series:**
1. Add series ID to `FRED_SERIES` in `config/settings.py`
2. Add feature transformation in `build_feature_matrix()` in `data/ingestion.py`
3. Optionally add as a slowdown signal in `_create_5class_labels()` in `models/regime_classifier.py`
4. Clear cache and retrain

**When adding a new ETF:**
1. Add ticker to the appropriate dict in `config/settings.py` (EQUITY_SECTORS, FIXED_INCOME, etc.)
2. Add weight to `REGIME_SUB_ALLOCATIONS` for each regime
3. Clear cache and retrain

**When adding a new Manager:**
1. Create new class inheriting from `BaseManager` in `agents/`
2. Implement `get_signals()`, `get_allocations()`, `get_confidence()`, `get_report()`, `update()`
3. Register with CIO in `main.py`: `self.cio.register_manager(new_manager)`
4. Can read economic signals via `self.economic_manager.get_signals()` (hybrid architecture)
