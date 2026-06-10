"""
Model loader and prediction functions.

Keeps all ML-specific logic out of the FastAPI layer.
`ModelRegistry` is instantiated once at startup and injected
into route handlers via FastAPI dependency injection.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import torch
import torch.nn as nn

from src.inference.pipeline import InferencePipeline


# ---------------------------------------------------------------------------
# LSTM architecture (must match training exactly)
# ---------------------------------------------------------------------------


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),  # works on any batch size, including 1
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.head(out).squeeze(-1)


# ---------------------------------------------------------------------------
# Model registry — one instance shared for the life of the server
# ---------------------------------------------------------------------------


class ModelRegistry:
    """
    Loads and holds all models + the inference pipeline.

    Call `from_paths()` once at startup; then use `predict_xgb()`
    and `predict_lstm()` inside route handlers.
    """

    def __init__(
        self,
        xgb_model,
        lgbm_model,
        lstm_model: LSTMModel | None,
        pipeline: InferencePipeline,
        lstm_config: dict,
        device: torch.device,
    ) -> None:
        self.xgb = xgb_model
        self.lgbm = lgbm_model
        self.lstm = lstm_model
        self.pipe = pipeline
        self.lstm_config = lstm_config
        self.device = device

    # ------------------------------------------------------------------

    @classmethod
    def from_paths(
        cls,
        xgb_path: str | Path = "models/best_xgb.pkl",
        lgbm_path: str | Path = "models/best_lgbm.pkl",
        lstm_path: str | Path = "models/best_lstm.pt",
        lstm_config_path: str | Path = "models/lstm_config.json",
        preprocess_scaler_path: str | Path = "models/preprocess_scaler.pkl",
        feature_scaler_path: str | Path = "models/feature_scaler.pkl",
        feature_cols_path: str | Path = "models/feature_cols.txt",
    ) -> "ModelRegistry":
        """
        Load all artifacts from disk.

        Missing LSTM weights → lstm stays None (XGBoost-only mode).
        Missing XGBoost pickle → raises FileNotFoundError immediately.
        """
        # MPS (Apple Silicon GPU) is intentionally skipped — it segfaults at
        # torch.load() time on several macOS + PyTorch 2.x combinations.
        # CPU inference is fast enough for this model size.
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # --- XGBoost (required) ---
        xgb_path = Path(xgb_path)
        if not xgb_path.exists():
            raise FileNotFoundError(
                f"XGBoost model not found at {xgb_path}. "
                "Run the tuning script first: python -m src.training.tune_xgb_optuna_mlflow"
            )
        xgb_model = joblib.load(xgb_path)

        # --- LightGBM (required) ---
        lgbm_path = Path(lgbm_path)
        if not lgbm_path.exists():
            raise FileNotFoundError(
                f"LightGBM model not found at {lgbm_path}. "
                "Run the tuning script first: python -m src.training.tune_lgbm_optuna_mlflow"
            )
        lgbm_model = joblib.load(lgbm_path)

        # --- LSTM config ---
        with open(lstm_config_path) as f:
            lstm_config = json.load(f)

        # --- Inference pipeline ---
        pipeline = InferencePipeline(
            preprocess_scaler_path=preprocess_scaler_path,
            feature_scaler_path=feature_scaler_path,
            feature_cols_path=feature_cols_path,
            rolling_window=5,
            seq_len=lstm_config["seq_len"],
        )

        # --- LSTM (optional) ---
        lstm_model: LSTMModel | None = None
        lstm_path = Path(lstm_path)
        if lstm_path.exists():
            n_features = len(pipeline.feature_cols)
            lstm_model = LSTMModel(
                input_size=n_features,
                hidden_size=lstm_config["hidden_size"],
                num_layers=lstm_config["num_layers"],
                dropout=lstm_config.get("dropout", 0.2),
            ).to(device)
            lstm_model.load_state_dict(
                torch.load(lstm_path, map_location=device, weights_only=True)
            )
            lstm_model.eval()

            # print("=== MODEL FEATURES ===")
            # print(list(xgb_model.feature_names_in_))

            # print("=== PIPELINE FEATURES ===")
            # print(pipeline.feature_cols)

            # print(
            #    "Missing from pipeline:",
            #    set(xgb_model.feature_names_in_) - set(pipeline.feature_cols)
            # )

            # print(
            #    "Extra in pipeline:",
            #    set(pipeline.feature_cols) - set(xgb_model.feature_names_in_)
            # )

        return cls(xgb_model, lgbm_model, lstm_model, pipeline, lstm_config, device)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @property
    def models_loaded(self) -> dict[str, bool]:
        return {
            "xgboost": self.xgb is not None,
            "lgbm": self.lgbm is not None,
            "lstm": self.lstm is not None,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_xgb(self, readings: list[dict]) -> float:
        """
        Run XGBoost on a list of sensor reading dicts.

        The last reading in the list is the prediction point.
        Earlier readings are used only to build rolling / diff features.
        We recommend sending at least 5 cycles (rolling_window size).
        """
        X = self.pipe.transform_xgb(readings)

        pred = float(self.xgb.predict(X)[0])
        # Clip to [0, 125] — matches the RUL cap used during training
        return max(0.0, min(pred, 125.0))
    
    def predict_lgbm(self, readings: list[dict]) -> float:
        """
        Run LightGBM on a list of sensor reading dicts.

        The last reading in the list is the prediction point.
        Earlier readings are used only to build rolling / diff features.
        We recommend sending at least 5 cycles (rolling_window size).
        """
        X = self.pipe.transform_lgbm(readings)

        pred = float(self.lgbm.predict(X)[0])
        # Clip to [0, 125] — matches the RUL cap used during training
        return max(0.0, min(pred, 125.0))

    def predict_lstm(self, readings: list[dict]) -> float:
        """
        Run LSTM on an ordered list of sensor reading dicts.

        Send at least `seq_len` cycles for best accuracy.
        """
        if self.lstm is None:
            raise RuntimeError(
                "LSTM model weights not found. "
                "Train the model first: python -m src.training.train_lstm_optuna_mlflow"
            )
        seq = self.pipe.transform_lstm(readings)  # (1, seq_len, n_features)
        tensor = torch.tensor(seq, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred = float(self.lstm(tensor).cpu().item())
        return max(0.0, min(pred, 125.0))
