"""
aws/lambda/src/inference/predict.py
=====================================
Lambda-specific ModelRegistry — identical public API to the Railway version
but uses ONNX Runtime instead of PyTorch for LSTM inference.

Why this matters for Lambda
----------------------------
Package size limit on Lambda:  250 MB (zipped) / ~550 MB (unzipped)
torch==2.11.0 unpacked:        ~800 MB   ← exceeds limit, deployment fails
onnxruntime==1.20.0 unpacked:  ~50  MB   ← fits with room to spare

Everything else (XGBoost, LightGBM, scikit-learn, FastAPI) is unchanged.
The ONNX model file (best_lstm.onnx) was produced by scripts/export_onnx.py
and is bundled inside the Lambda Docker image under /var/task/models/.

The InferencePipeline (feature engineering) is identical to Railway —
the same scalers and feature_cols.txt are used. Only the final model
inference step changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort  # replaces torch entirely for LSTM

from src.inference.pipeline import InferencePipeline


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """
    Loads and holds all models + the inference pipeline.

    Identical interface to the Railway version — the FastAPI routes in
    app.py call the same methods (predict_xgb, predict_lgbm, predict_lstm)
    without knowing which backend is being used.
    """

    def __init__(
        self,
        xgb_model,
        lgbm_model,
        lstm_session: ort.InferenceSession | None,   # ← was LSTMModel
        pipeline:     InferencePipeline,
        lstm_config:  dict,
    ) -> None:
        self.xgb          = xgb_model
        self.lgbm         = lgbm_model
        self.lstm_session = lstm_session   # ONNX Runtime session, not a PyTorch model
        self.pipe         = pipeline
        self.lstm_config  = lstm_config

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_paths(
        cls,
        xgb_path:               str | Path = "models/best_xgb.pkl",
        lgbm_path:              str | Path = "models/best_lgbm.pkl",
        # .onnx replaces .pt — no torch.load needed
        lstm_path:              str | Path = "models/best_lstm.onnx",
        lstm_config_path:       str | Path = "models/lstm_config.json",
        preprocess_scaler_path: str | Path = "models/preprocess_scaler.pkl",
        feature_scaler_path:    str | Path = "models/feature_scaler.pkl",
        feature_cols_path:      str | Path = "models/feature_cols.txt",
    ) -> "ModelRegistry":

        # --- XGBoost (required) ---
        xgb_path = Path(xgb_path)
        if not xgb_path.exists():
            raise FileNotFoundError(
                f"XGBoost model not found at {xgb_path}. "
                "Run: python -m src.training.tune_xgb_optuna_mlflow"
            )
        xgb_model = joblib.load(xgb_path)

        # --- LightGBM (required) ---
        lgbm_path = Path(lgbm_path)
        if not lgbm_path.exists():
            raise FileNotFoundError(
                f"LightGBM model not found at {lgbm_path}. "
                "Run: python -m src.training.tune_lgbm_optuna_mlflow"
            )
        lgbm_model = joblib.load(lgbm_path)

        # Force LightGBM single-threaded — Lambda runs on a single vCPU,
        # so multi-threading offers no benefit and wastes cold-start time.
        try:
            lgbm_model.set_params(n_jobs=1)
        except Exception:
            pass

        # --- LSTM config ---
        with open(lstm_config_path) as f:
            lstm_config = json.load(f)

        # --- Inference pipeline (same as Railway) ---
        pipeline = InferencePipeline(
            preprocess_scaler_path = preprocess_scaler_path,
            feature_scaler_path    = feature_scaler_path,
            feature_cols_path      = feature_cols_path,
            rolling_window         = 5,
            seq_len                = lstm_config["seq_len"],
        )

        # --- LSTM via ONNX Runtime (optional — graceful degradation) ---
        # If best_lstm.onnx is missing the API still starts and serves
        # XGBoost + LightGBM. Only the /predict/lstm endpoint returns 503.
        lstm_session: ort.InferenceSession | None = None
        lstm_path = Path(lstm_path)

        if lstm_path.exists():
            # CPUExecutionProvider is the only provider available on Lambda.
            # Explicitly listing it prevents a warning from ONNX Runtime
            # trying to find CUDA/TensorRT providers that don't exist.
            lstm_session = ort.InferenceSession(
                str(lstm_path),
                providers=["CPUExecutionProvider"],
            )
            print(f"✅ ONNX LSTM loaded from {lstm_path}")
        else:
            print(f"⚠️  ONNX LSTM not found at {lstm_path} — LSTM endpoint disabled")

        return cls(xgb_model, lgbm_model, lstm_session, pipeline, lstm_config)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def models_loaded(self) -> dict[str, bool]:
        return {
            "xgboost":  self.xgb          is not None,
            "lightgbm": self.lgbm         is not None,
            "lstm":     self.lstm_session  is not None,  # ONNX session, not .lstm
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_xgb(self, readings: list[dict]) -> float:
        """XGBoost prediction — unchanged from Railway version."""
        X    = self.pipe.transform_xgb(readings)
        pred = float(self.xgb.predict(X)[0])
        return max(0.0, min(pred, 125.0))

    def predict_lgbm(self, readings: list[dict]) -> float:
        """LightGBM prediction — unchanged from Railway version."""
        X    = self.pipe.transform_lgbm(readings)
        pred = float(self.lgbm.predict(X, num_threads=1)[0])
        return max(0.0, min(pred, 125.0))

    def predict_lstm(self, readings: list[dict]) -> float:
        """
        LSTM prediction via ONNX Runtime (no PyTorch required).

        The ONNX session.run() call is equivalent to model(tensor) in PyTorch
        but uses a pre-compiled computation graph that runs ~20% faster on CPU
        and requires no GPU driver or CUDA libraries.

        Input name "input" and output name "predicted_rul" were set during
        the ONNX export in scripts/export_onnx.py.
        """
        if self.lstm_session is None:
            raise RuntimeError(
                "ONNX LSTM not loaded. Run scripts/export_onnx.py first."
            )

        # transform_lstm returns (1, seq_len, n_features) float32 numpy array
        seq = self.pipe.transform_lstm(readings)

        # ONNX Runtime expects a plain numpy array, not a torch.Tensor.
        # The input name "input" must match what was set during export.
        ort_outputs = self.lstm_session.run(
            output_names = ["predicted_rul"],
            input_feed   = {"input": seq},
        )

        # ort_outputs is a list of numpy arrays — one per output_name
        pred = float(ort_outputs[0].flatten()[0])
        return max(0.0, min(pred, 125.0))