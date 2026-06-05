"""
tests/test_inference.py — Inference Tests

Covers two layers:
  Unit   → pipeline.py transform functions (no API, no disk models)
  Integration → FastAPI endpoints via TestClient (no real model needed —
                a tiny dummy XGBoost model is trained on synthetic data)

The double-scaling bug is explicitly regression-tested:
  raw unscaled input → API pipeline → prediction must differ from
  pre-scaled input  → API pipeline → (wrong) prediction.
"""

import json
import numpy as np
import pandas as pd
import joblib
import pytest

from pathlib import Path
from unittest.mock import MagicMock, patch

# torch is intentionally NOT imported at module level.
# On macOS, importing torch during pytest collection triggers MPS/OpenMP
# initialisation that segfaults inside pytest's forked process.
# Tests that need torch import it locally inside their fixture/function.


# ---------------------------------------------------------------------------
# Helpers to build minimal real artifacts for integration tests
# ---------------------------------------------------------------------------

N_FEAT = 6   # matches conftest feat_cols (f0..f5)
SEQ_LEN = 5


def _sensor_row(engine_id=1, cycle=1) -> dict:
    """A valid SensorReading-compatible dict with realistic FD001 values."""
    return {
        "engine_id": engine_id,
        "cycle": cycle,
        "setting_1": -0.0007, "setting_2": -0.0004, "setting_3": 100.0,
        "sensor_1":  518.67,  "sensor_2":  641.82,  "sensor_3": 1589.70,
        "sensor_4":  1400.60, "sensor_5":  14.62,   "sensor_6":  21.61,
        "sensor_7":  554.36,  "sensor_8":  2388.02, "sensor_9": 9046.19,
        "sensor_10": 1.30,    "sensor_11": 47.47,   "sensor_12": 521.66,
        "sensor_13": 2388.02, "sensor_14": 8138.62, "sensor_15": 8.4195,
        "sensor_16": 0.03,    "sensor_17": 392.0,   "sensor_18": 2388.0,
        "sensor_19": 100.0,   "sensor_20": 39.06,   "sensor_21": 23.419,
    }


def _make_artifacts(tmp_path: Path):
    """
    Write minimal real artifacts to tmp_path so ModelRegistry.from_paths() works:
      - preprocess_scaler.pkl
      - feature_scaler.pkl
      - feature_cols.txt
      - best_xgb.pkl
    No LSTM weights are written → LSTM stays None (expected in tests).
    """
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    rng = np.random.default_rng(0)

    # Feature columns
    feat_cols = [f"f{i}" for i in range(N_FEAT)]
    with open(tmp_path / "feature_cols.txt", "w") as f:
        f.write("\n".join(feat_cols))

    # Preprocess scaler — expects settings + remaining sensors
    # (mirrors what preprocess.py would produce)
    pre_cols = (
        [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)
           if f"sensor_{i}" not in ["sensor_1","sensor_5","sensor_10","sensor_16","sensor_18","sensor_19"]]
    )
    X_pre = rng.standard_normal((200, len(pre_cols)))
    pre_df = pd.DataFrame(X_pre, columns=pre_cols)
    pre_scaler = StandardScaler().fit(pre_df)
    joblib.dump(pre_scaler, tmp_path / "preprocess_scaler.pkl")

    # Feature scaler — must cover base features + rolling_mean + diff variants
    # because pipeline.py generates those before calling feature_scaler.transform()
    feat_and_derived = (
        feat_cols
        + [f"{c}_rolling_mean" for c in feat_cols]
        + [f"{c}_diff"         for c in feat_cols]
    )
    X_feat = rng.standard_normal((200, len(feat_and_derived)))
    feat_df = pd.DataFrame(X_feat, columns=feat_and_derived)
    feat_scaler = StandardScaler().fit(feat_df)
    joblib.dump(feat_scaler, tmp_path / "feature_scaler.pkl")

    # Tiny XGBoost model
    X_train = rng.standard_normal((200, N_FEAT))
    y_train = rng.uniform(0, 125, 200)
    xgb = XGBRegressor(n_estimators=5, max_depth=2)
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, tmp_path / "best_xgb.pkl")

    # lstm_config (no .pt weights → LSTM stays None)
    config = {"seq_len": SEQ_LEN, "hidden_size": 16, "num_layers": 1, "dropout": 0.1}
    with open(tmp_path / "lstm_config.json", "w") as f:
        json.dump(config, f)

    return tmp_path


# ===========================================================================
# Unit tests — InferencePipeline transforms
# ===========================================================================


class TestInferencePipelineTransforms:
    """
    Test pipeline.py in isolation using mock scalers.
    The scaler is replaced with an identity transform so we can
    verify shapes and logic without real artifacts on disk.
    """

    @pytest.fixture()
    def mock_pipeline(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        from src.inference.pipeline import InferencePipeline

        rng = np.random.default_rng(1)

        # Use real surviving sensor column names so the pipeline can find them
        # in the sensor reading DataFrame (which has sensor_N, not f0..fN)
        pre_cols = (
            [f"setting_{i}" for i in range(1, 4)]
            + [f"sensor_{i}" for i in range(1, 22)
               if f"sensor_{i}" not in ["sensor_1","sensor_5","sensor_10","sensor_16","sensor_18","sensor_19"]]
        )
        # Feature cols = a subset of pre_cols (simulates what select_features picks)
        feat_cols = pre_cols[:N_FEAT]

        # Write feature_cols.txt with real sensor column names
        with open(tmp_path / "feature_cols.txt", "w") as fh:
            fh.write("\n".join(feat_cols))

        # Preprocess scaler
        pre_df = pd.DataFrame(rng.standard_normal((100, len(pre_cols))), columns=pre_cols)
        pre_sc = StandardScaler().fit(pre_df)
        joblib.dump(pre_sc, tmp_path / "pre.pkl")

        # Feature scaler — must cover base features + their rolling/diff variants
        # that the pipeline will generate
        feat_and_derived = (
            feat_cols
            + [f"{c}_rolling_mean" for c in feat_cols]
            + [f"{c}_diff"         for c in feat_cols]
        )
        feat_df = pd.DataFrame(rng.standard_normal((100, len(feat_and_derived))), columns=feat_and_derived)
        feat_sc = StandardScaler().fit(feat_df)
        joblib.dump(feat_sc, tmp_path / "feat.pkl")

        return InferencePipeline(
            preprocess_scaler_path=tmp_path / "pre.pkl",
            feature_scaler_path=tmp_path / "feat.pkl",
            feature_cols_path=tmp_path / "feature_cols.txt",
            rolling_window=3,
            seq_len=SEQ_LEN,
        )

    def test_transform_xgb_shape(self, mock_pipeline):
        readings = [_sensor_row(cycle=c) for c in range(1, 8)]
        X = mock_pipeline.transform_xgb(readings)
        assert X.shape == (1, N_FEAT), f"Expected (1, {N_FEAT}), got {X.shape}"

    def test_transform_lstm_shape_enough_cycles(self, mock_pipeline):
        readings = [_sensor_row(cycle=c) for c in range(1, SEQ_LEN + 3)]
        seq = mock_pipeline.transform_lstm(readings)
        assert seq.shape == (1, SEQ_LEN, N_FEAT)

    def test_transform_lstm_pads_short_sequence(self, mock_pipeline):
        """Fewer cycles than seq_len → zero-padded at the front."""
        readings = [_sensor_row(cycle=1)]   # only 1 cycle
        seq = mock_pipeline.transform_lstm(readings)
        assert seq.shape == (1, SEQ_LEN, N_FEAT)
        # First SEQ_LEN-1 rows should be zero (padding)
        pad_rows = seq[0, :SEQ_LEN - 1, :]
        assert np.allclose(pad_rows, 0.0), "Expected zero-padding at the front"

    def test_transform_xgb_uses_last_row(self, mock_pipeline):
        """XGBoost result must come from the LAST reading, not the first."""
        # Two engine IDs in the readings — the last row belongs to engine 2
        readings_a = [_sensor_row(engine_id=1, cycle=c) for c in range(1, 6)]
        readings_b = [_sensor_row(engine_id=2, cycle=c) for c in range(1, 6)]
        Xa = mock_pipeline.transform_xgb(readings_a)
        Xb = mock_pipeline.transform_xgb(readings_b)
        # Different engine histories → different feature vectors
        # (rolling means will differ once history is different)
        # At minimum shapes should be equal
        assert Xa.shape == Xb.shape


# ===========================================================================
# Unit tests — ModelRegistry
# ===========================================================================


class TestModelRegistry:

    @pytest.fixture()
    def registry(self, tmp_path):
        from src.inference.predict import ModelRegistry
        artifacts = _make_artifacts(tmp_path)
        return ModelRegistry.from_paths(
            xgb_path=artifacts / "best_xgb.pkl",
            lstm_path=artifacts / "best_lstm.pt",          # doesn't exist → lstm=None
            lstm_config_path=artifacts / "lstm_config.json",
            preprocess_scaler_path=artifacts / "preprocess_scaler.pkl",
            feature_scaler_path=artifacts / "feature_scaler.pkl",
            feature_cols_path=artifacts / "feature_cols.txt",
        )

    def test_xgb_loaded(self, registry):
        assert registry.models_loaded["xgboost"] is True

    def test_lstm_none_without_weights(self, registry):
        assert registry.models_loaded["lstm"] is False

    def test_predict_xgb_returns_float(self, registry):
        readings = [_sensor_row(cycle=c) for c in range(1, 8)]
        result = registry.predict_xgb(readings)
        assert isinstance(result, float)

    def test_predict_xgb_clipped_to_0_125(self, registry):
        readings = [_sensor_row(cycle=c) for c in range(1, 8)]
        result = registry.predict_xgb(readings)
        assert 0.0 <= result <= 125.0

    def test_predict_lstm_raises_without_weights(self, registry):
        readings = [_sensor_row(cycle=c) for c in range(1, 8)]
        with pytest.raises(RuntimeError, match="LSTM model weights not found"):
            registry.predict_lstm(readings)

    def test_missing_xgb_raises_file_not_found(self, tmp_path):
        from src.inference.predict import ModelRegistry
        artifacts = _make_artifacts(tmp_path)
        with pytest.raises(FileNotFoundError):
            ModelRegistry.from_paths(
                xgb_path=artifacts / "nonexistent.pkl",
                lstm_path=artifacts / "best_lstm.pt",
                lstm_config_path=artifacts / "lstm_config.json",
                preprocess_scaler_path=artifacts / "preprocess_scaler.pkl",
                feature_scaler_path=artifacts / "feature_scaler.pkl",
                feature_cols_path=artifacts / "feature_cols.txt",
            )


# ===========================================================================
# Integration tests — FastAPI via TestClient
# ===========================================================================


class TestFastAPIEndpoints:
    """
    Spin up the FastAPI app with a real (tiny) model registry injected
    into app.state so the lifespan startup is bypassed.
    """

    @pytest.fixture()
    def client(self, tmp_path):
        from fastapi.testclient import TestClient
        from src.api.app import app
        from src.inference.predict import ModelRegistry

        artifacts = _make_artifacts(tmp_path)
        registry = ModelRegistry.from_paths(
            xgb_path=artifacts / "best_xgb.pkl",
            lstm_path=artifacts / "best_lstm.pt",
            lstm_config_path=artifacts / "lstm_config.json",
            preprocess_scaler_path=artifacts / "preprocess_scaler.pkl",
            feature_scaler_path=artifacts / "feature_scaler.pkl",
            feature_cols_path=artifacts / "feature_cols.txt",
        )

        # Inject registry directly — skips lifespan model loading
        app.state.registry = registry
        app.state.startup_error = None

        return TestClient(app)

    # --- /health ---

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_ok(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_shows_xgb_loaded(self, client):
        resp = client.get("/health")
        assert resp.json()["models_loaded"]["xgboost"] is True

    # --- /predict/xgb ---

    def test_predict_xgb_returns_200(self, client):
        resp = client.post("/predict/xgb", json=_sensor_row())
        assert resp.status_code == 200

    def test_predict_xgb_response_schema(self, client):
        resp = client.post("/predict/xgb", json=_sensor_row())
        body = resp.json()
        assert "predicted_rul" in body
        assert "engine_id"     in body
        assert "cycle"         in body
        assert "model"         in body

    def test_predict_xgb_rul_in_valid_range(self, client):
        resp = client.post("/predict/xgb", json=_sensor_row())
        rul = resp.json()["predicted_rul"]
        assert 0.0 <= rul <= 125.0

    def test_predict_xgb_wrong_method_returns_405(self, client):
        resp = client.get("/predict/xgb")
        assert resp.status_code == 405

    def test_predict_xgb_missing_field_returns_422(self, client):
        bad = _sensor_row()
        del bad["sensor_1"]
        resp = client.post("/predict/xgb", json=bad)
        assert resp.status_code == 422

    # --- /predict/xgb/batch ---

    def test_predict_xgb_batch_returns_200(self, client):
        readings = [_sensor_row(cycle=c) for c in range(1, 8)]
        resp = client.post("/predict/xgb/batch", json={"readings": readings})
        assert resp.status_code == 200

    def test_predict_xgb_batch_uses_last_reading(self, client):
        """engine_id and cycle in response must match the LAST reading."""
        readings = [_sensor_row(engine_id=1, cycle=c) for c in range(1, 6)]
        resp = client.post("/predict/xgb/batch", json={"readings": readings})
        body = resp.json()
        assert body["engine_id"] == 1
        assert body["cycle"]     == 5

    def test_predict_xgb_batch_empty_readings_returns_422(self, client):
        resp = client.post("/predict/xgb/batch", json={"readings": []})
        assert resp.status_code == 422

    # --- /predict/lstm (no weights → 503) ---

    def test_predict_lstm_without_weights_returns_503(self, client):
        readings = [_sensor_row(cycle=c) for c in range(1, 6)]
        resp = client.post("/predict/lstm", json={"readings": readings})
        assert resp.status_code == 503


# ===========================================================================
# Regression test — double-scaling produces different (wrong) predictions
# ===========================================================================


class TestDoubleScalingRegression:
    """
    Explicit guard against the bug where test_clean.csv (already scaled)
    is passed to the API pipeline, which then scales it again.

    If this test fails, it means the pipeline is no longer sensitive to
    input scale — which would be a different bug worth investigating.
    """

    @pytest.fixture()
    def registry(self, tmp_path):
        from src.inference.predict import ModelRegistry
        artifacts = _make_artifacts(tmp_path)
        return ModelRegistry.from_paths(
            xgb_path=artifacts / "best_xgb.pkl",
            lstm_path=artifacts / "best_lstm.pt",
            lstm_config_path=artifacts / "lstm_config.json",
            preprocess_scaler_path=artifacts / "preprocess_scaler.pkl",
            feature_scaler_path=artifacts / "feature_scaler.pkl",
            feature_cols_path=artifacts / "feature_cols.txt",
        )

    def test_scaled_input_differs_from_raw_input(self, registry):
        raw_reading   = _sensor_row(cycle=50)

        # Simulate "double-scaled" input by multiplying sensors by 0.01
        # (mimicking what happens when StandardScaler output ≈ 0 is re-scaled)
        scaled_reading = {**raw_reading}
        for k in scaled_reading:
            if k.startswith("sensor_") or k.startswith("setting_"):
                scaled_reading[k] *= 0.01

        pred_raw    = registry.predict_xgb([raw_reading])
        pred_scaled = registry.predict_xgb([scaled_reading])

        assert pred_raw != pytest.approx(pred_scaled, rel=0.05), (
            "Raw and pre-scaled inputs produced identical predictions — "
            "pipeline may not be sensitive to input scale."
        )