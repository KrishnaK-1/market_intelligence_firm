# Market Intelligence Firm
### Multi-Manager Investment Decision Platform

A fully ML-driven economic regime classification and asset allocation system built on an ensemble of HMM + GMM + XGBoost, with a FastAPI dashboard for real-time monitoring.

---

## Architecture

```
market_intelligence_firm/
├── config/
│   └── settings.py           # All configuration, tickers, regime templates
├── data/
│   ├── ingestion.py           # FRED API, Yahoo Finance, Shiller CAPE
│   └── cache/                 # Cached data files (auto-generated)
├── models/
│   ├── regime_classifier.py   # Ensemble HMM + GMM + XGBoost
│   └── saved/                 # Persisted trained models
├── agents/
│   └── economic_manager.py    # Economic Regime Manager + CIO Aggregator
├── alerts/
│   └── engine.py              # Regime shift early warning system
├── backtest/
│   └── engine.py              # Walk-forward backtester
├── dashboard/
│   ├── server.py              # FastAPI server + scheduler
│   └── index.html             # Interactive dashboard
├── main.py                    # Orchestrator (full pipeline)
├── requirements.txt
├── .env.example
└── README.md
```

## Quick Start

### 1. Clone and Install

```bash
cd market_intelligence_firm
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your FRED API key:
# FRED_API_KEY=your_key_here
```

Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html

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

### Features (22+)

Yield curve spreads, credit spreads, unemployment dynamics, industrial production growth, CPI inflation, housing momentum, payroll growth, consumer sentiment, initial claims, Fed funds trajectory, M2 growth, S&P momentum, VIX, CAPE ratio + z-score.

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

## License

Internal use. Not financial advice.
