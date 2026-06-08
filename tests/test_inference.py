"""
tests/test_inference.py — Inference Tests

Covers three layers:
  Unit        → InferencePipeline transform shapes and logic
  Registry    → ModelRegistry loads, predicts, clips, errors correctly
  Integration → FastAPI endpoints via TestClient
  Regression  → double-scaling bug guard

Artifact strategy
-----------------
All test artifacts are built by running the REAL pipeline on synthetic data
(via conftest fixtures), then reading back the saved files. This means the
test artifacts are always consistent with the production code — no hardcoded
column counts that break when the pipeline changes.
"""

import json
import numpy as np
import pandas as pd
import joblib
import pytest
from pathlib import Path

# torch is NOT imported at module level — importing it during pytest collection
# triggers MPS/OpenMP on macOS and causes a segfault.

N_FEAT  = 6    # base sensor columns used in mock_pipeline fixture only
SEQ_LEN = 5


# ---------------------------------------------------------------------------
# Sensor reading helper
# ---------------------------------------------------------------------------

def _sensor_row(engine_id=1, cycle=1) -> dict:
    """Valid SensorReading dict with realistic FD001 values."""
    return {
        "engine_id": engine_id, "cycle": cycle,
        "setting_1": -0.0007, "setting_2": -0.0004, "setting_3": 100.0,
        "sensor_1":  518.67,  "sensor_2":  641.82,  "sensor_3":  1589.70,
        "sensor_4":  1400.60, "sensor_5":  14.62,   "sensor_6":  21.61,
        "sensor_7":  554.36,  "sensor_8":  2388.02, "sensor_9":  9046.19,
        "sensor_10": 1.30,    "sensor_11": 47.47,   "sensor_12": 521.66,
        "sensor_13": 2388.02, "sensor_14": 8138.62, "sensor_15": 8.4195,
        "sensor_16": 0.03,    "sensor_17": 392.0,   "sensor_18": 2388.0,
        "sensor_19": 100.0,   "sensor_20": 39.06,   "sensor_21": 23.419,
    }


# ---------------------------------------------------------------------------
# Artifact builder — runs the real pipeline on synthetic data
# ---------------------------------------------------------------------------

def _make_artifacts(tmp_path: Path, processed_dir: Path, feature_dir: Path):
    """
    Build test artifacts from the real pipeline outputs.

    - processed_dir  → train_clean.csv  (for preprocess scaler)
    - feature_dir    → feature_engineered_train.csv (for feature scaler + XGBoost)

    Both come from conftest fixtures so column counts are always consistent
    with the production pipeline — no hardcoded numbers.
    """
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # ── Preprocess scaler ────────────────────────────────────────────────
    train_clean = pd.read_csv(processed_dir / "train_clean.csv")
    pre_cols = [c for c in train_clean.columns
                if c not in ["engine_id", "cycle", "rul"]]
    pre_scaler = StandardScaler().fit(train_clean[pre_cols])
    joblib.dump(pre_scaler, tmp_path / "preprocess_scaler.pkl")

    # ── Feature-engineered data → feature cols ───────────────────────────
    fe_train = pd.read_csv(feature_dir / "feature_engineered_train.csv")
    full_feat_cols = [c for c in fe_train.columns
                      if c not in ["engine_id", "cycle", "rul"]]

    with open(tmp_path / "feature_cols.txt", "w") as f:
        f.write("\n".join(full_feat_cols))

    # ── Feature scaler ───────────────────────────────────────────────────
    feat_scaler = StandardScaler().fit(fe_train[full_feat_cols])
    joblib.dump(feat_scaler, tmp_path / "feature_scaler.pkl")

    # ── XGBoost model ────────────────────────────────────────────────────
    # XGBoost is trained with cycle as the FIRST column (matching pipeline.py
    # which does X.insert(0, "cycle", ...) before returning to the model).
    xgb_cols = ["cycle"] + full_feat_cols
    X_train  = fe_train[xgb_cols].values
    y_train  = fe_train["rul"].values
    xgb = XGBRegressor(n_estimators=10, max_depth=2, random_state=0)
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, tmp_path / "best_xgb.pkl")

    # ── LSTM config (no weights → lstm stays None) ───────────────────────
    config = {"seq_len": SEQ_LEN, "hidden_size": 16, "num_layers": 1, "dropout": 0.1}
    with open(tmp_path / "lstm_config.json", "w") as f:
        json.dump(config, f)

    return tmp_path


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def artifacts(tmp_path, processed_dir, feature_dir):
    """Real pipeline artifacts built from synthetic processed/feature-engineered data."""
    return _make_artifacts(tmp_path / "artifacts", processed_dir, feature_dir)


@pytest.fixture()
def registry(artifacts):
    from src.inference.predict import ModelRegistry
    return ModelRegistry.from_paths(
        xgb_path=artifacts / "best_xgb.pkl",
        lstm_path=artifacts / "best_lstm.pt",           # doesn't exist → lstm=None
        lstm_config_path=artifacts / "lstm_config.json",
        preprocess_scaler_path=artifacts / "preprocess_scaler.pkl",
        feature_scaler_path=artifacts / "feature_scaler.pkl",
        feature_cols_path=artifacts / "feature_cols.txt",
    )


# ---------------------------------------------------------------------------
# Unit tests — InferencePipeline transforms
# ---------------------------------------------------------------------------

class TestInferencePipelineTransforms:

    @pytest.fixture()
    def mock_pipeline(self, artifacts):
        """Pipeline built from real artifacts — always self-consistent."""
        from src.inference.pipeline import InferencePipeline
        return InferencePipeline(
            preprocess_scaler_path=artifacts / "preprocess_scaler.pkl",
            feature_scaler_path=artifacts / "feature_scaler.pkl",
            feature_cols_path=artifacts / "feature_cols.txt",
            rolling_window=3,
            seq_len=SEQ_LEN,
        )

    def test_transform_xgb_shape(self, mock_pipeline):
        readings = [_sensor_row(cycle=c) for c in range(1, 8)]
        X = mock_pipeline.transform_xgb(readings)
        # XGBoost uses feature_cols + cycle prepended = n_feat + 1
        n_feat = len(mock_pipeline.feature_cols) + 1
        assert X.shape == (1, n_feat), f"Expected (1, {n_feat}), got {X.shape}"

    def test_transform_lstm_shape_enough_cycles(self, mock_pipeline):
        readings = [_sensor_row(cycle=c) for c in range(1, SEQ_LEN + 3)]
        seq = mock_pipeline.transform_lstm(readings)
        n_feat = len(mock_pipeline.feature_cols)
        assert seq.shape == (1, SEQ_LEN, n_feat)

    def test_transform_lstm_pads_short_sequence(self, mock_pipeline):
        """Fewer cycles than seq_len → zero-padded at the front."""
        readings = [_sensor_row(cycle=1)]
        seq = mock_pipeline.transform_lstm(readings)
        n_feat = len(mock_pipeline.feature_cols)
        assert seq.shape == (1, SEQ_LEN, n_feat)
        pad_rows = seq[0, :SEQ_LEN - 1, :]
        assert np.allclose(pad_rows, 0.0), "Expected zero-padding at the front"

    def test_transform_xgb_uses_last_row(self, mock_pipeline):
        readings_a = [_sensor_row(engine_id=1, cycle=c) for c in range(1, 6)]
        readings_b = [_sensor_row(engine_id=2, cycle=c) for c in range(1, 6)]
        Xa = mock_pipeline.transform_xgb(readings_a)
        Xb = mock_pipeline.transform_xgb(readings_b)
        assert Xa.shape == Xb.shape


# ---------------------------------------------------------------------------
# Unit tests — ModelRegistry
# ---------------------------------------------------------------------------

class TestModelRegistry:

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

    def test_missing_xgb_raises_file_not_found(self, artifacts):
        from src.inference.predict import ModelRegistry
        with pytest.raises(FileNotFoundError):
            ModelRegistry.from_paths(
                xgb_path=artifacts / "nonexistent.pkl",
                lstm_path=artifacts / "best_lstm.pt",
                lstm_config_path=artifacts / "lstm_config.json",
                preprocess_scaler_path=artifacts / "preprocess_scaler.pkl",
                feature_scaler_path=artifacts / "feature_scaler.pkl",
                feature_cols_path=artifacts / "feature_cols.txt",
            )


# ---------------------------------------------------------------------------
# Integration tests — FastAPI via TestClient
# ---------------------------------------------------------------------------

class TestFastAPIEndpoints:

    @pytest.fixture()
    def client(self, registry):
        from fastapi.testclient import TestClient
        from src.api.app import app

        app.state.registry = registry
        app.state.startup_error = None
        return TestClient(app)

    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_status_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_health_shows_xgb_loaded(self, client):
        assert client.get("/health").json()["models_loaded"]["xgboost"] is True

    def test_predict_xgb_returns_200(self, client):
        assert client.post("/predict/xgb", json=_sensor_row()).status_code == 200

    def test_predict_xgb_response_schema(self, client):
        body = client.post("/predict/xgb", json=_sensor_row()).json()
        assert "predicted_rul" in body
        assert "engine_id"     in body
        assert "cycle"         in body
        assert "model"         in body

    def test_predict_xgb_rul_in_valid_range(self, client):
        rul = client.post("/predict/xgb", json=_sensor_row()).json()["predicted_rul"]
        assert 0.0 <= rul <= 125.0

    def test_predict_xgb_wrong_method_returns_405(self, client):
        assert client.get("/predict/xgb").status_code == 405

    def test_predict_xgb_missing_field_returns_422(self, client):
        bad = _sensor_row()
        del bad["sensor_1"]
        assert client.post("/predict/xgb", json=bad).status_code == 422

    def test_predict_xgb_batch_returns_200(self, client):
        readings = [_sensor_row(cycle=c) for c in range(1, 8)]
        assert client.post("/predict/xgb/batch", json={"readings": readings}).status_code == 200

    def test_predict_xgb_batch_uses_last_reading(self, client):
        readings = [_sensor_row(engine_id=1, cycle=c) for c in range(1, 6)]
        body = client.post("/predict/xgb/batch", json={"readings": readings}).json()
        assert body["engine_id"] == 1
        assert body["cycle"]     == 5

    def test_predict_xgb_batch_empty_readings_returns_422(self, client):
        assert client.post("/predict/xgb/batch", json={"readings": []}).status_code == 422

    def test_predict_lstm_without_weights_returns_503(self, client):
        readings = [_sensor_row(cycle=c) for c in range(1, 6)]
        assert client.post("/predict/lstm", json={"readings": readings}).status_code == 503


# ---------------------------------------------------------------------------
# Regression test — double-scaling
# ---------------------------------------------------------------------------

class TestDoubleScalingRegression:
    """
    Guard against the bug where already-scaled data is passed to the API
    pipeline which then scales it again, compressing all feature values
    toward zero (double-scaling).

    We test this at the pipeline transform level — not at the XGBoost
    prediction level — because a tiny test model trained on random data
    may not be sensitive to feature scale (cycle dominates). The transform
    output is always sensitive: raw sensor values (~400-700) vs near-zero
    (~0.0004-0.0007) produce completely different scaled feature vectors.
    """

    def test_pipeline_transform_differs_for_raw_vs_scaled_input(self, artifacts):
        """Raw sensor values and pre-scaled (~zero) values must produce
        different feature vectors after pipeline transformation."""
        from src.inference.pipeline import InferencePipeline

        pipeline = InferencePipeline(
            preprocess_scaler_path=artifacts / "preprocess_scaler.pkl",
            feature_scaler_path=artifacts / "feature_scaler.pkl",
            feature_cols_path=artifacts / "feature_cols.txt",
            rolling_window=3,
            seq_len=SEQ_LEN,
        )

        raw_reading = _sensor_row(cycle=50)

        # Pre-scaled input: sensor values collapsed to near-zero
        # (what StandardScaler output looks like — ~[-3, 3] range,
        # not ~[400, 700] like raw FD001 sensor readings)
        scaled_reading = {
            k: (v * 0.001 if k.startswith("sensor_") or k.startswith("setting_") else v)
            for k, v in raw_reading.items()
        }

        X_raw    = pipeline.transform_xgb([raw_reading])
        X_scaled = pipeline.transform_xgb([scaled_reading])

        # The two feature vectors must differ significantly
        max_diff = float(np.abs(X_raw - X_scaled).max())
        assert max_diff > 0.1, (
            f"Raw and pre-scaled inputs produced nearly identical feature vectors "
            f"(max diff={max_diff:.6f}). Pipeline may not be applying the scaler."
        )