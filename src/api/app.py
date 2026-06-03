"""
FastAPI application for the NASA Turbofan RUL predictor.

Routes
------
GET  /health          → liveness + which models are loaded
POST /predict/xgb     → single-cycle XGBoost prediction
POST /predict/lstm    → multi-cycle LSTM prediction

Startup
-------
The ModelRegistry is created once when the server starts and
stored in app.state so every request can access it cheaply.
"""

from __future__ import annotations

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
)


# ---------------------------------------------------------------------------
# Lifespan — load models once, share across all requests
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup; nothing to tear down on shutdown."""
    try:
        registry = ModelRegistry.from_paths()
        app.state.registry = registry
        print("✅ Models loaded:", registry.models_loaded)
    except FileNotFoundError as exc:
        # Start anyway — /health will report the missing model
        print(f"⚠️  Model loading warning: {exc}")
        app.state.registry = None
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
# Dependency — injects the registry into each route that needs it
# ---------------------------------------------------------------------------

def get_registry(request: Request) -> ModelRegistry:
    registry: ModelRegistry | None = request.app.state.registry
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded. Check server logs.",
        )
    return registry


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Monitoring"],
)
def health(request: Request) -> HealthResponse:
    """Returns server status and which models are available."""
    registry: ModelRegistry | None = request.app.state.registry
    if registry is None:
        return HealthResponse(
            status="degraded",
            models_loaded={"xgboost": False, "lstm": False},
        )
    return HealthResponse(
        status="ok",
        models_loaded=registry.models_loaded,
    )


@app.post(
    "/predict/xgb",
    response_model=XGBPrediction,
    summary="XGBoost RUL prediction",
    tags=["Prediction"],
)
def predict_xgb(
    payload: SensorReading,
    registry: ModelRegistry = Depends(get_registry),
) -> XGBPrediction:
    """
    Predict RUL for a **single cycle** using XGBoost.

    Send the current sensor reading. The model uses only that row
    (rolling/diff features will be zero since there is no history).
    For better accuracy with history, send the last N readings
    to `/predict/xgb/batch` instead (see below).
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
    "/predict/xgb/batch",
    response_model=XGBPrediction,
    summary="XGBoost RUL prediction with sensor history",
    tags=["Prediction"],
)
def predict_xgb_batch(
    payload: SequenceRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> XGBPrediction:
    """
    Predict RUL from an **ordered history of cycles** using XGBoost.

    Send the last N readings (oldest → newest).
    At least 5 cycles recommended so rolling-mean features are meaningful.
    The prediction is made for the **last reading** in the list.
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

    The model was trained with `seq_len=41`.
    - Send ≥ 41 cycles → the last 41 are used.
    - Send < 41 cycles → the sequence is zero-padded at the front.

    Readings must be ordered **oldest → newest**.
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
# Global exception handler — keeps error responses consistent
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )