"""
FastAPI application for the NASA Turbofan RUL predictor.

Routes
------
GET  /health          → liveness + which models are loaded
POST /predict/xgb     → single-cycle XGBoost prediction
POST /predict/xgb/batch → multi-cycle XGBoost prediction
POST /predict/lstm    → multi-cycle LSTM prediction
"""

from __future__ import annotations

import traceback
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.inference.predict import ModelRegistry
from src.inference.schemas import (
    HealthResponse,
    LSTMPrediction,
    SequenceRequest,
    SensorReading,
    XGBPrediction,
    LGBMPrediction,
)


# ---------------------------------------------------------------------------
# Lifespan — load models once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models on startup.

    Catches ALL exceptions so a missing scaler, bad pickle, or any other
    artifact problem is logged clearly instead of crashing the process.
    """
    try:
        registry = ModelRegistry.from_paths()
        app.state.registry = registry
        app.state.startup_error = None
        print("✅ Models loaded:", registry.models_loaded)
    except Exception as exc:
        # Log the full traceback so the cause is visible in the terminal
        print("⚠️  Model loading failed — API starting in degraded mode.")
        print(traceback.format_exc())
        app.state.registry = None
        app.state.startup_error = str(exc)
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NASA Turbofan RUL API",
    description=(
        "Predicts Remaining Useful Life (RUL) of turbofan engines "
        "from raw sensor readings using XGBoost and LSTM models."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


def get_registry(request: Request) -> ModelRegistry:
    registry: ModelRegistry | None = request.app.state.registry
    if registry is None:
        error = getattr(request.app.state, "startup_error", "Unknown error")
        raise HTTPException(
            status_code=503,
            detail=f"Models failed to load at startup: {error}",
        )
    return registry


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
# / → simple landing endpoint to confirm API is alive.
@app.get("/")
def root():
    return {"message": "NASA Turbofan RUL Prediction API is running 🚀"}


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Monitoring"],
)
def health(request: Request) -> HealthResponse:
    """Returns server status and which models are loaded."""
    registry: ModelRegistry | None = request.app.state.registry
    if registry is None:
        error = getattr(request.app.state, "startup_error", "Unknown error")
        return HealthResponse(
            status=f"degraded: {error}",
            models_loaded={"xgboost": False, "lightgbm": False, "lstm": False},
        )
    return HealthResponse(
        status="ok",
        models_loaded=registry.models_loaded,
    )


@app.post(
    "/predict/xgb",
    response_model=XGBPrediction,
    summary="XGBoost RUL prediction (single cycle)",
    tags=["Prediction"],
)
def predict_xgb(
    payload: SensorReading,
    registry: ModelRegistry = Depends(get_registry),
) -> XGBPrediction:
    """
    Predict RUL for a **single cycle** using XGBoost.

    Rolling/diff features will be zero since there is no history.
    For better accuracy send multiple cycles to `/predict/xgb/batch`.
    """
    readings = [payload.model_dump()]
    try:
        rul = registry.predict_xgb(readings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return XGBPrediction(
        engine_id=payload.engine_id,
        cycle=payload.cycle,
        predicted_rul=round(rul, 2),
    )

@app.post(
    "/predict/lgbm",
    response_model=LGBMPrediction,
    summary="LightGBM RUL prediction (single cycle)",
    tags=["Prediction"],
)
def predict_lgbm(
    payload: SensorReading,
    registry: ModelRegistry = Depends(get_registry),
) -> LGBMPrediction:
    """
    Predict RUL for a **single cycle** using LightGBM.

    Rolling/diff features will be zero since there is no history.
    For better accuracy send multiple cycles to `/predict/lgbm/batch`.
    """
    readings = [payload.model_dump()]
    try:
        rul = registry.predict_lgbm(readings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return LGBMPrediction(
        engine_id=payload.engine_id,
        cycle=payload.cycle,
        predicted_rul=round(rul, 2),
    )


@app.post(
    "/predict/xgb/batch",
    response_model=XGBPrediction,
    summary="XGBoost RUL prediction (with history)",
    tags=["Prediction"],
)
def predict_xgb_batch(
    payload: SequenceRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> XGBPrediction:
    """
    Predict RUL from an **ordered history of cycles** using XGBoost.

    Send readings oldest → newest. At least 5 cycles recommended so
    rolling-mean features are meaningful. Prediction is for the last reading.
    """
    readings = [r.model_dump() for r in payload.readings]
    try:
        rul = registry.predict_xgb(readings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    last = payload.readings[-1]
    return XGBPrediction(
        engine_id=last.engine_id,
        cycle=last.cycle,
        predicted_rul=round(rul, 2),
    )

@app.post(
    "/predict/lgbm/batch",
    response_model=LGBMPrediction,
    summary="LightGBM RUL prediction (with history)",
    tags=["Prediction"],
)
def predict_lgbm_batch(
    payload: SequenceRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> LGBMPrediction:
    """
    Predict RUL from an **ordered history of cycles** using LightGBM.

    Send readings oldest → newest. At least 5 cycles recommended so
    rolling-mean features are meaningful. Prediction is for the last reading.
    """
    readings = [r.model_dump() for r in payload.readings]
    try:
        rul = registry.predict_lgbm(readings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    last = payload.readings[-1]
    return LGBMPrediction(
        engine_id=last.engine_id,
        cycle=last.cycle,
        predicted_rul=round(rul, 2),
    )


@app.post(
    "/predict/lstm",
    response_model=LSTMPrediction,
    summary="LSTM RUL prediction",
    tags=["Prediction"],
)
def predict_lstm(
    payload: SequenceRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> LSTMPrediction:
    """
    Predict RUL from an **ordered sequence of cycles** using LSTM.

    Send readings oldest → newest. The model uses the last `seq_len` cycles
    (zero-padded at the front if fewer are provided).
    """
    if not registry.models_loaded.get("lstm"):
        raise HTTPException(
            status_code=503,
            detail="LSTM model is not loaded. Train it first.",
        )

    readings = [r.model_dump() for r in payload.readings]
    try:
        rul = registry.predict_lstm(readings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    last = payload.readings[-1]
    return LSTMPrediction(
        engine_id=last.engine_id,
        cycle=last.cycle,
        predicted_rul=round(rul, 2),
    )


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )