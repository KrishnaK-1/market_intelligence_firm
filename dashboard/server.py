"""
FastAPI Server - Market Intelligence Dashboard
================================================
Endpoints:
  GET  /                     -> Dashboard (HTML)
  GET  /api/report           -> Full JSON report
  GET  /api/regime           -> Current regime + probabilities
  GET  /api/allocations      -> Recommended allocations
  GET  /api/forecasts        -> Forward regime forecasts
  GET  /api/alerts           -> Active alerts
  GET  /api/backtest         -> Backtest results
  GET  /api/transition       -> Transition probability matrix
  GET  /api/confidence       -> Model confidence/disagreement
  GET  /api/nowcast          -> Real-time nowcast (XGBoost only)
  POST /api/refresh          -> Manual data refresh (inference only)
  POST /api/retrain          -> Force retrain + full pipeline
  GET  /api/health           -> Health check

Auto-refreshes daily at 7 AM ET.
"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import REFRESH_HOUR, SERVER_HOST, SERVER_PORT, TIMEZONE
from main import MarketIntelligenceFirm

logger = logging.getLogger("server")

# =====================================================================
#  Global State
# =====================================================================

firm = MarketIntelligenceFirm()
latest_report: Optional[dict] = None
is_running = False
scheduler = AsyncIOScheduler(timezone=TIMEZONE)


async def scheduled_refresh():
    """Daily auto-refresh at configured hour."""
    global latest_report, is_running
    if is_running:
        logger.info("Refresh already in progress, skipping scheduled run")
        return
    is_running = True
    try:
        logger.info("Scheduled daily refresh starting...")
        latest_report = firm.run_inference_only()
        logger.info("Scheduled refresh complete")
    except Exception as e:
        logger.error(f"Scheduled refresh failed: {e}")
    finally:
        is_running = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: run initial pipeline, schedule daily refresh."""
    global latest_report
    logger.info("Starting Market Intelligence Firm server...")
    try:
        latest_report = firm.run_full_pipeline(force_retrain=True, run_backtest=True)
        logger.info("Initial pipeline complete")
    except Exception as e:
        logger.error(f"Initial pipeline failed: {e}")
        latest_report = {"error": str(e), "timestamp": datetime.now().isoformat()}

    # Schedule daily refresh
    scheduler.add_job(
        scheduled_refresh,
        CronTrigger(hour=REFRESH_HOUR, minute=0, timezone=TIMEZONE),
        id="daily_refresh",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(f"Scheduled daily refresh at {REFRESH_HOUR}:00 {TIMEZONE}")

    yield

    scheduler.shutdown()
    logger.info("Server shutting down")


# =====================================================================
#  App Setup
# =====================================================================

app = FastAPI(
    title="Market Intelligence Firm",
    description="Multi-manager investment decision platform",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
#  API Endpoints
# =====================================================================


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "last_run": latest_report.get("last_run") if latest_report else None,
        "is_running": is_running,
    }


@app.get("/api/report")
async def get_report():
    if latest_report is None:
        raise HTTPException(503, "Pipeline has not run yet")
    return JSONResponse(content=json.loads(json.dumps(latest_report, default=str)))


@app.get("/api/regime")
async def get_regime():
    if latest_report is None:
        raise HTTPException(503, "Pipeline has not run yet")
    econ = latest_report.get("managers", {}).get("Economic Regime Manager", {})
    return {
        "current_regime": econ.get("current_regime"),
        "regime_probabilities": econ.get("regime_probabilities"),
        "confidence": econ.get("confidence"),
        "individual_models": econ.get("individual_models"),
        "model_agreement": econ.get("model_agreement"),
        "ensemble_weights": econ.get("ensemble_weights"),
    }


@app.get("/api/allocations")
async def get_allocations():
    if latest_report is None:
        raise HTTPException(503, "Pipeline has not run yet")
    econ = latest_report.get("managers", {}).get("Economic Regime Manager", {})
    return {
        "asset_class_allocation": econ.get("asset_class_allocation"),
        "ticker_allocation": econ.get("ticker_allocation"),
    }


@app.get("/api/forecasts")
async def get_forecasts():
    if latest_report is None:
        raise HTTPException(503, "Pipeline has not run yet")
    econ = latest_report.get("managers", {}).get("Economic Regime Manager", {})
    return {"forecasts": econ.get("forecasts")}


@app.get("/api/alerts")
async def get_alerts():
    if latest_report is None:
        raise HTTPException(503, "Pipeline has not run yet")
    return {"alerts": latest_report.get("alerts", [])}


@app.get("/api/backtest")
async def get_backtest():
    if latest_report is None:
        raise HTTPException(503, "Pipeline has not run yet")
    return {"backtest": latest_report.get("backtest")}


@app.get("/api/transition")
async def get_transition_matrix():
    if latest_report is None:
        raise HTTPException(503, "Pipeline has not run yet")
    econ = latest_report.get("managers", {}).get("Economic Regime Manager", {})
    return {"transition_matrix": econ.get("transition_matrix")}


@app.get("/api/confidence")
async def get_confidence():
    if latest_report is None:
        raise HTTPException(503, "Pipeline has not run yet")
    econ = latest_report.get("managers", {}).get("Economic Regime Manager", {})
    return {
        "confidence": econ.get("confidence"),
        "model_agreement": econ.get("model_agreement"),
        "individual_models": econ.get("individual_models"),
        "ensemble_weights": econ.get("ensemble_weights"),
        "feature_importance": econ.get("feature_importance"),
    }


@app.get("/api/nowcast")
async def get_nowcast(model: str = "xgboost"):
    """
    Real-time nowcast using XGBoost (default) or Neural Net (LSTM).
    Fetches latest daily/weekly data, constructs feature row,
    and returns regime classification with freshness metadata.
    """
    try:
        from data.nowcast import run_nowcast

        # Pass the firm's FRED data and classifier for efficiency
        result = run_nowcast(
            fred_monthly=firm.fred_data,
            classifier=firm.classifier,
            model=model,
            neural_classifier=getattr(firm, "neural_classifier", None),
        )

        # Add comparison with official model
        if latest_report:
            econ = latest_report.get("managers", {}).get("Economic Regime Manager", {})
            official_regime = econ.get("current_regime", "Unknown")
            result["comparison"] = {
                "official_regime": official_regime,
                "nowcast_regime": result.get("regime", "Unknown"),
                "divergence": official_regime != result.get("regime", "Unknown"),
            }

        return JSONResponse(content=json.loads(json.dumps(result, default=str)))

    except Exception as e:
        logger.error(f"Nowcast failed: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "regime": "Unknown"},
        )


@app.post("/api/refresh")
async def manual_refresh(background_tasks: BackgroundTasks):
    global is_running
    if is_running:
        return {"status": "already_running", "message": "A refresh is already in progress"}

    async def do_refresh():
        global latest_report, is_running
        is_running = True
        try:
            latest_report = firm.run_inference_only()
        except Exception as e:
            logger.error(f"Manual refresh failed: {e}")
        finally:
            is_running = False

    background_tasks.add_task(asyncio.coroutine(do_refresh) if False else do_refresh)
    return {"status": "started", "message": "Inference refresh initiated"}


@app.post("/api/retrain")
async def manual_retrain(background_tasks: BackgroundTasks):
    global is_running
    if is_running:
        return {"status": "already_running"}

    async def do_retrain():
        global latest_report, is_running
        is_running = True
        try:
            latest_report = firm.force_refresh()
        except Exception as e:
            logger.error(f"Retrain failed: {e}")
        finally:
            is_running = False

    background_tasks.add_task(do_retrain)
    return {"status": "started", "message": "Full retrain initiated"}


# =====================================================================
#  Dashboard (served as HTML)
# =====================================================================


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the interactive dashboard."""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Dashboard loading... Run server to generate.</h1>")


# =====================================================================
#  Entry Point
# =====================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "dashboard.server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
        log_level="info",
    )
