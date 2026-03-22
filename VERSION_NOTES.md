# Version Notes

## v4: Neural Regime Classifier (LSTM) — 2026-03-22

### Summary
Added an independent PyTorch LSTM neural network alongside the existing HMM + GMM + XGBoost
ensemble for economic regime classification. The neural net is **display-only** — it does not
affect portfolio allocations or ensemble predictions. It serves as a secondary signal for
the dashboard and nowcast module.

### New Files
- **`models/neural_regime_classifier.py`** — LSTM-based regime classifier with attention mechanism
- **`models/feature_registry.py`** — Manages base (24) vs experimental (4) feature sets

### Architecture
- LSTM (32 hidden units, 1 layer, unidirectional) + Temporal Attention + Linear classifier
- 8,702 parameters (intentionally small for ~880 training samples)
- Sequence length: 24 months (2 years of lookback per prediction)
- 5-class output: Expansion, Slowdown, Contraction, Recovery, Crisis

### Features (28 total)
- **24 base features** (shared with XGBoost): yield spreads, unemployment, industrial production,
  CPI, payrolls, sentiment, VIX, CAPE, etc.
- **4 experimental features** (neural net only):
  - `oil_yoy` — WTI Crude Oil YoY % change (Yahoo Finance CL=F fallback when FRED daily fails)
  - `permit_yoy` — Building Permits YoY % change (FRED: PERMIT)
  - `ahe_yoy` — Average Hourly Earnings YoY % change (FRED: CES0500000003)
  - `lei_yoy` — Conference Board Leading Economic Index YoY % change (FRED: USSLIND)

### Training Details
- Training period: 1954-11 to ~2012 (80/20 train/val split)
- Validation period: ~2012 to 2026-03
- Loss: CrossEntropyLoss with sqrt class weights and label smoothing (0.05)
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-3)
- LR schedule: CosineAnnealingWarmRestarts (T_0=30, T_mult=2)
- Temporal oversampling of minority classes (2x max with Gaussian noise)
- Early stopping on combined metric: 0.7 * raw_accuracy + 0.3 * balanced_accuracy

### Integration Points
- **`main.py`** — Step 4b trains the neural net after the ensemble (non-blocking on failure)
- **`agents/economic_manager.py`** — Displays neural net regime call alongside ensemble
- **`data/nowcast.py`** — Neural net option in real-time nowcast with full 28-feature support
- **`dashboard/index.html`** — Dropdown to switch between XGBoost and Neural Net in nowcast panel
- **`config/settings.py`** — Added USSLIND, PERMIT, CES0500000003 to FRED series

### Baseline KPIs (as of 2026-03-22)
| Metric | Neural Net | Ensemble |
|--------|-----------|----------|
| Val accuracy (overall) | 79.5% | 89.7% |
| Val balanced accuracy | 25.0% | — |
| Expansion accuracy | ~93% | 89.0% |
| Slowdown/Recovery/Crisis | ~0% | 92.0% (recession) |
| Backtest Sharpe | 1.08 | 1.08 |
| Backtest Max Drawdown | -15.9% | -15.9% |

### Known Limitations
- Neural net produces low-confidence predictions (~21-22%) for recent months
- Cannot reliably detect Slowdown, Recovery, or Crisis transitions (extreme class imbalance)
- Nowcast neural net predictions may differ from diagnostic due to low confidence margins

### Potential Future Improvements
- Reduce to binary (Expansion vs Non-Expansion) or 3-class classification for better balance
- Regression-based economic health score instead of categorical classification
- Add more leading indicators identified by research (excess bond premium, corporate profits)

---

## v3: Backtest, Nowcast, and Optimizer Improvements — 2026-03-21

### Changes (carried from v3 branch, included in this merge)
- Walk-forward backtest with proper temporal splits and NaN handling
- BLS API integration for faster employment/CPI data in nowcast
- Withholding tax signal added to nowcast
- Optimizer parameter persistence via .gitignore
- Forward-shifted pre-recession Slowdown labels (added then reverted)

---

## v2: Nowcast with BLS, Conviction Framework — 2026-03-20
- Real-time nowcast module with BLS API, Yahoo Finance, FRED daily sources
- Conviction framework for regime classification confidence
- Retrain stability improvements

## v1: Initial Release
- HMM + GMM + XGBoost ensemble regime classifier
- FRED data ingestion with Shiller CAPE
- Regime-based portfolio allocation
- FastAPI dashboard
