"""
Pydantic schemas for the RUL prediction API.

These define what JSON shape the API accepts (request)
and what it returns (response).
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sensor reading — one row of raw sensor data for a single engine
# ---------------------------------------------------------------------------

class SensorReading(BaseModel):
    """
    One cycle of raw sensor readings for a single engine.

    Field names match the NASA FD001 column layout exactly.
    All 21 sensors must be provided; the API will internally
    drop the low-information ones (sensor_1/5/10/16/18/19).
    """

    engine_id: int = Field(..., ge=1, description="Engine unit number")
    cycle: int     = Field(..., ge=1, description="Operational cycle index")

    # Operational settings
    setting_1: float
    setting_2: float
    setting_3: float

    # Sensor measurements
    sensor_1:  float
    sensor_2:  float
    sensor_3:  float
    sensor_4:  float
    sensor_5:  float
    sensor_6:  float
    sensor_7:  float
    sensor_8:  float
    sensor_9:  float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "engine_id": 1,
                "cycle": 50,
                "setting_1": -0.0007,
                "setting_2": -0.0004,
                "setting_3": 100.0,
                "sensor_1":  518.67,
                "sensor_2":  641.82,
                "sensor_3":  1589.70,
                "sensor_4":  1400.60,
                "sensor_5":  14.62,
                "sensor_6":  21.61,
                "sensor_7":  554.36,
                "sensor_8":  2388.02,
                "sensor_9":  9046.19,
                "sensor_10": 1.30,
                "sensor_11": 47.47,
                "sensor_12": 521.66,
                "sensor_13": 2388.02,
                "sensor_14": 8138.62,
                "sensor_15": 8.4195,
                "sensor_16": 0.03,
                "sensor_17": 392.0,
                "sensor_18": 2388.0,
                "sensor_19": 100.0,
                "sensor_20": 39.06,
                "sensor_21": 23.4190,
            }
        }
    }


# ---------------------------------------------------------------------------
# Multi-cycle request — a sequence of readings for the LSTM endpoint
# ---------------------------------------------------------------------------

class SequenceRequest(BaseModel):
    """
    An ordered list of sensor readings for one engine.

    Send at least `seq_len` cycles (41 by default).
    If fewer are sent, the API zero-pads the beginning.
    """

    readings: list[SensorReading] = Field(
        ...,
        min_length=1,
        description="Ordered list of sensor readings (oldest → newest)",
    )


# ---------------------------------------------------------------------------
# Prediction responses
# ---------------------------------------------------------------------------

class XGBPrediction(BaseModel):
    """Response from the XGBoost endpoint."""

    engine_id: int
    cycle: int
    predicted_rul: float = Field(..., description="Predicted RUL in cycles")
    model: str = "xgboost"


class LSTMPrediction(BaseModel):
    """Response from the LSTM endpoint."""

    engine_id: int
    cycle: int
    predicted_rul: float = Field(..., description="Predicted RUL in cycles")
    model: str = "lstm"


class HealthResponse(BaseModel):
    """Response from the health-check endpoint."""

    status: str
    models_loaded: dict[str, bool]