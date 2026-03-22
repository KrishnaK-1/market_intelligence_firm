# Market Intelligence Firm
### Multi-Manager Investment Decision Platform

A fully ML-driven economic regime classification and asset allocation system built on an ensemble of HMM + GMM + XGBoost, with an independent LSTM neural network classifier and a FastAPI dashboard for real-time monitoring.

---

## Architecture

```
market_intelligence_firm/
├── config/
│   └── settings.py           # All configuration, tickers, regime templates
├── data/
│   ├── ingestion.py           # FRED API, Yahoo Finance, Shiller CAPE
│   ├── nowcast.py             # Real-time nowcast (BLS + Yahoo + FRED daily)
│   ├── ie_data.xls            # Shiller CAPE local data file
│   └── cache/                 # Cached data files (auto-generated)
├── models/
│   ├── regime_classifier.py   # Ensemble HMM + GMM + XGBoost
│   ├── neural_regime_classifier.py  # LSTM neural net (display-only)
│   ├── feature_registry.py    # Base (24) + experimental (4) feature management
│   ├── feedback.py            # Self-optimizing feedback loop
│   └── saved/                 # Persisted trained models + params
├── agents/
│   └── economic_manager.py    # Economic Regime Manager + CIO Aggregator
├── alerts/
│   └── engine.py              # Regime shift early warning system
├── backtest/
│   └── engine.py              # Walk-forward backtester
├── dashboard/
│   ├── server.py              # FastAPI server + scheduler + nowcast endpoint
│   └── index.html             # Interactive 4-tab dashboard
├── main.py                    # Orchestrator (full pipeline + retrain stability)
├── requirements.txt
├── .env                       # FRED_API_KEY + BLS_API_KEY
└── README.md
```

## Quick Start

### 1. Clone and Install

```bash
cd market_intelligence_firm
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your API keys:
# FRED_API_KEY=your_fred_key_here
# BLS_API_KEY=your_bls_key_here
```

- Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html
- Get a free BLS API key at: https://data.bls.gov/registrationEngine/ (required for nowcast module)

### 3. Run the Server

```bash
# From the project root:
python -m dashboard.server
```

Or with uvicorn directly:

```bash
uvicorn dashboard.server:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

### 4. CLI Mode (no server)

```bash
python main.py
```

---

## PyCharm Setup

### Run Configuration

1. **Run → Edit Configurations → Add → Python**
2. **Script path**: `dashboard/server.py`
3. **Working directory**: (project root)
4. **Python interpreter**: Your venv with requirements installed
5. **Environment variables**: `FRED_API_KEY=your_key`

### Alternate: Module Configuration

1. **Module name**: `uvicorn`
2. **Parameters**: `dashboard.server:app --host 0.0.0.0 --port 8000 --reload`
3. **Working directory**: (project root)

---

## System Design

### Ensemble Regime Classifier (5 States)

| State | Description | Identification |
|-------|-------------|---------------|
| **Expansion** | Broad growth, positive momentum | Non-recession + positive indicators |
| **Slowdown** | Growth decelerating, risks rising | Non-recession + deteriorating leads |
| **Contraction** | NBER recession | NBER label = 1 |
| **Recovery** | Early-cycle rebound | First 6 months post-recession |
| **Crisis** | Extreme financial stress | Recession + VIX/credit spike |

**Three models, zero hard-coded rules:**

- **HMM (Gaussian, 5 states)** — Captures sequential regime dynamics and transition probabilities
- **GMM (Gaussian Mixture, 5 components)** — Captures distributional regime clustering
- **XGBoost (multi-class, 5 targets)** — Supervised classifier trained on derived 5-class labels from NBER + feature logic

Weights are calibrated via time-series cross-validation. States from unsupervised models are aligned to canonical labels using the Hungarian algorithm.

### Multi-Agent Architecture (Hybrid)

```
                    ┌──────────────┐
                    │  CIO         │
                    │  Aggregator  │
                    └──────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
     ┌──────┴──────┐ ┌────┴────┐  ┌──────┴──────┐
     │  Economic   │ │ Equity  │  │    Risk     │
     │  Regime Mgr │ │ Manager │  │   Manager   │
     └─────────────┘ └─────────┘  └─────────────┘
            ↑              ↑              ↑
            │    can read   │              │
            └──────────────┘──────────────┘
```

- Each manager has a standard interface: `get_signals()`, `get_allocations()`, `get_confidence()`, `get_report()`
- Managers can read Economic Regime signals for context
- Each reports independently to CIO
- CIO combines using confidence-weighted aggregation

### Data Sources

| Source | Coverage | Frequency |
|--------|----------|-----------|
| FRED API | 20+ macro series | Monthly (some weekly) |
| Shiller CAPE | 1871–present | Monthly |
| Yahoo Finance | 30+ ETFs | Daily → Monthly |

### Features

**24 Base Features** (used by ensemble + neural net):
Yield curve spreads (10Y-2Y, 10Y-3M), credit spread + 3M change, unemployment rate + 3M/12M change, industrial production YoY + MoM, CPI YoY, housing starts YoY, payrolls YoY + MoM, consumer sentiment + change, initial claims 4-week change, fed funds + 12M change, M2 YoY, S&P 500 YoY + 3M momentum, VIX, CAPE + z-score.

**4 Experimental Features** (neural net only):
- `oil_yoy` — WTI Crude Oil YoY % change (FRED DCOILWTICO / Yahoo CL=F fallback)
- `permit_yoy` — Building Permits YoY % change (FRED PERMIT)
- `ahe_yoy` — Average Hourly Earnings YoY % change (FRED CES0500000003)
- `lei_yoy` — Conference Board Leading Economic Index YoY % change (FRED USSLIND)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Interactive dashboard |
| GET | `/api/report` | Full JSON report |
| GET | `/api/regime` | Current regime + probabilities |
| GET | `/api/allocations` | Recommended allocations |
| GET | `/api/forecasts` | Forward forecasts (1/3/6M) |
| GET | `/api/alerts` | Active alerts |
| GET | `/api/backtest` | Backtest results |
| GET | `/api/transition` | Transition probability matrix |
| GET | `/api/confidence` | Model confidence/disagreement |
| GET | `/api/nowcast` | Real-time nowcast (XGBoost or Neural Net via `?model=neural_net`) |
| POST | `/api/refresh` | Manual data refresh (inference) |
| POST | `/api/retrain` | Force full retrain |
| GET | `/api/health` | Health check |

---

## Adding New Managers

To add a new manager (e.g., Equity Portfolio Manager):

```python
from agents.economic_manager import BaseManager

class EquityManager(BaseManager):
    name = "Equity Portfolio Manager"

    def __init__(self, economic_manager=None):
        self.economic_mgr = economic_manager  # hybrid: can read signals

    def get_signals(self) -> dict:
        # Read economic context
        regime = self.economic_mgr.get_signals() if self.economic_mgr else {}
        # Your equity selection logic here
        return {...}

    def get_allocations(self) -> dict:
        return {"AAPL": 0.05, "MSFT": 0.05, ...}

    def get_confidence(self) -> float:
        return 0.75

    def get_report(self) -> dict:
        return {**self.get_signals(), "allocations": self.get_allocations()}

    def update(self, **kwargs):
        # Refresh with new data
        pass
```

Register it in `main.py`:
```python
equity_mgr = EquityManager(economic_manager=self.economic_manager)
self.cio.register_manager(equity_mgr)
```

---

## Configuration

All configuration lives in `config/settings.py`:

- **Regime count & labels**: `N_REGIMES`, `REGIME_LABELS`
- **FRED series**: `FRED_SERIES` dict
- **Investment universe**: `EQUITY_SECTORS`, `FIXED_INCOME`, `COMMODITIES`, `INTERNATIONAL`
- **Allocation templates**: `REGIME_ALLOCATIONS`, `REGIME_SUB_ALLOCATIONS`
- **Server settings**: `SERVER_PORT`, `REFRESH_HOUR`, `TIMEZONE`

---

## Version History

### v4.0 — LSTM Neural Regime Classifier + LEI Signal (March 22, 2026)

**Neural Net Classifier (`models/neural_regime_classifier.py`)**
- Independent PyTorch LSTM neural network — display-only, does not affect ensemble or portfolio allocations
- Architecture: LSTM (32 hidden, 1 layer, unidirectional) + Temporal Attention + Linear classifier (8,702 parameters)
- 24-month sequence length (2 years of lookback per prediction)
- 28 features: 24 base (shared with XGBoost) + 4 experimental (neural net only)
- Training: CrossEntropyLoss with sqrt class weights, label smoothing (0.05), AdamW optimizer, CosineAnnealingWarmRestarts schedule
- Temporal oversampling of minority classes (2x max with Gaussian noise augmentation)
- Early stopping on combined metric (70% raw accuracy + 30% balanced accuracy)
- Gradient-based feature importance via attention weights

**Feature Registry (`models/feature_registry.py`)**
- Manages base vs experimental feature sets
- Experimental features computed from FRED data with Yahoo Finance fallback for oil prices
- Conference Board LEI (USSLIND) added as leading indicator — AUC 0.97 for recession prediction per Chicago Fed research

**Integration**
- `main.py` Step 4b: trains neural net after ensemble (non-blocking on failure)
- `agents/economic_manager.py`: displays neural net regime call alongside ensemble vote
- `data/nowcast.py`: neural net nowcast with full 28-feature support (select via dashboard dropdown)
- `dashboard/index.html`: model selector in nowcast panel (XGBoost / Neural Net)

**Baseline KPIs**
- Neural net: 79.5% overall val accuracy, 25.0% balanced accuracy
- Ensemble: 89.7% ground truth accuracy (unchanged)
- Backtest: Sharpe 1.08, MaxDD -15.9% (unchanged)

### v3.0 — Backtest + Nowcast Improvements (March 21, 2026)

**Backtest Engine**
- Walk-forward backtest with proper temporal splits and NaN handling in classifier
- Fixed StandardScaler empty array issue on early splits

**Nowcast Enhancements**
- Added withholding tax signal to nowcast
- Optimizer parameter persistence via .gitignore

**Experimental (reverted)**
- Forward-shifted pre-recession Slowdown labels with optimizer-tuned lead window (added then reverted due to instability)

### v2.0 — Dashboard Redesign + Nowcast + Stability (March 17, 2026)

**Nowcast Module (`data/nowcast.py`)**
- Real-time regime classification using XGBoost only (no HMM/GMM)
- Three-tier data ingestion: BLS API (payrolls, unemployment, CPI), Yahoo Finance (S&P 500, VIX), FRED daily (yields, credit spreads, fed funds)
- 18 of 24 features pulled live; remaining 6 carried forward from last official FRED monthly release
- Signals detection: S&P 3M momentum, VIX elevation, credit spread widening, yield curve inversion, consumer sentiment, unemployment acceleration
- Auto-refreshes every 15 minutes during market hours

**Conviction Framework (Dashboard)**
- Top regime > 65%: displays regime name with "HIGH CONVICTION" green badge
- Top regime 50-65% with > 5% gap: displays regime name with "LOW CONVICTION" amber badge
- Top regime < 50% or top two within 5%: displays "REGIME TRANSITION" with both contenders and red badge
- Replaces the misleading single-regime display when the model is genuinely uncertain

**Retrain Stability**
- 2% improvement threshold: optimizer must find params scoring > 2% better than current model to trigger retrain, preventing marginal parameter changes from flipping the regime call
- Parameter persistence: current model's params are saved to disk after each run, ensuring Retrain button produces consistent results across restarts
- Eliminated the Expansion/Slowdown flip-flop that occurred when optimizer found marginally different params on each run

**Dashboard Redesign (4 Tabs)**
- Tab 1 — Current Regime: conviction badge, probability bars, forecast table + chart, alerts, nowcast card
- Tab 2 — Portfolio & Allocations: asset class doughnut chart, 15 ticker weights in 3-column grid
- Tab 3 — Historical Analysis: full-width timeline, transition matrix, backtest stats + chart
- Tab 4 — Model Diagnostics: individual model votes, ensemble weight sliders, confidence cap sliders, feature importance

**BLS API Integration**
- Added BLS API v2 as a data source for employment and price data
- BLS updates payrolls, unemployment, and CPI 1-3 weeks faster than FRED
- Graceful fallback: if BLS is unavailable, nowcast uses FRED monthly cache

**Bug Fixes**
- Forecast table key case mismatch fixed (API sends lowercase `1m`/`3m`/`6m`)
- Alert re-generation after optimizer retrain
- FRED daily series fallback when API returns 500 errors
- StandardScaler feature name warning suppressed in nowcast

### v1.0 — Initial Release (March 15, 2026)

- HMM + GMM + XGBoost ensemble regime classifier (5 states)
- 24-feature matrix from FRED, Shiller CAPE, Yahoo Finance
- Self-optimizing feedback loop with parameter persistence
- FastAPI dashboard with regime display and alerts
- Walk-forward backtester (Sharpe 0.96, MaxDD -14.3%)
- 91% ground truth accuracy against NBER recession history
- Confidence cap fix for 2001 recession detection

### Planned — v5.0

- Reduce neural net to binary (Expansion vs Non-Expansion) or 3-class classification to address class imbalance
- Regression-based economic health score as alternative to categorical classification
- Add excess bond premium, corporate profits as experimental leading indicators
- Remove lagging experimental features (ahe_yoy) if they don't improve accuracy
- Connect dashboard weight sliders to backend for live recalibration
- Recency-weighted scoring (weight recent decades more heavily in optimizer)

---

## License

Internal use. Not financial advice.
