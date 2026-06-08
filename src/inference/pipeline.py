"""
Inference feature pipeline.

Reproduces — without re-fitting any scaler — the same transformations
that training applied:

  1. Drop low-information sensors          (preprocess.py  → drop_unused_sensors)
  2. Normalise with the saved scaler       (preprocess.py  → normalize_features)
  3. Select features saved after training  (build_feature.py → select_features)
  4. Add rolling-mean features             (build_feature.py → add_rolling_features)
  5. Add diff features                     (build_feature.py → add_difference_features)
  6. Scale engineered features             (build_feature.py → scale_features)

The two scalers (one from preprocess, one from build_feature) are persisted
to disk during the export step (see scripts/export_artifacts.py).

At inference we just load them and call .transform() — never .fit().
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# No hardcoded sensor drop list — the pipeline derives the correct
# columns from the preprocess scaler's feature_names_in_ attribute.

# The columns the feature pipeline keeps, in the exact order XGBoost expects.
# This list is written to disk by export_artifacts.py after training.
_FEATURE_COLS_FILE = "models/feature_cols.txt"


def _load_feature_cols(feature_cols_path: str | Path) -> list[str]:
    path = Path(feature_cols_path)
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


class InferencePipeline:
    """
    Stateless (after __init__) inference pipeline.

    Load once at startup, call transform_xgb() or transform_lstm() per request.
    """

    def __init__(
        self,
        preprocess_scaler_path: str | Path,
        feature_scaler_path: str | Path,
        feature_cols_path: str | Path,
        rolling_window: int = 5,
        seq_len: int = 41,
    ) -> None:
        self.preprocess_scaler = joblib.load(preprocess_scaler_path)
        self.feature_scaler    = joblib.load(feature_scaler_path)
        self.feature_cols      = _load_feature_cols(feature_cols_path)
        self.rolling_window    = rolling_window
        self.seq_len           = seq_len

        # Columns scaled by preprocess scaler (settings + sensors that survive)
        # We reconstruct this from the scaler's own feature_names_in_ attribute.
        self._preprocess_cols: list[str] = list(
            self.preprocess_scaler.feature_names_in_
        )

        # Columns scaled by feature scaler
        self._feature_scaler_cols: list[str] = list(
            self.feature_scaler.feature_names_in_
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_dataframe(self, readings: list[dict]) -> pd.DataFrame:
        """Convert a list of sensor dicts into a DataFrame."""
        return pd.DataFrame(readings)

    def _align_to_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only columns the preprocess scaler was fitted on + meta cols."""
        meta = ["engine_id", "cycle"]
        scaler_cols = list(self._preprocess_cols)
        keep = meta + [c for c in scaler_cols if c in df.columns]
        return df[[c for c in keep if c in df.columns]]

    def _apply_preprocess_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise with the scaler fitted during preprocess.py."""
        df = df.copy()
        cols = [c for c in self._preprocess_cols if c in df.columns]
        df[cols] = self.preprocess_scaler.transform(df[cols])
        return df

    def _select_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only the base features selected by correlation threshold."""
        # base features = feature_cols minus the _rolling_mean / _diff variants
        base = [
            c for c in self.feature_cols
            if not c.endswith("_rolling_mean") and not c.endswith("_diff")
        ]
        available = [c for c in base if c in df.columns]
        keep = ["engine_id", "cycle"] + available
        if "rul" in df.columns:
            keep.append("rul")
        return df[[c for c in keep if c in df.columns]]

    def _add_rolling(self, df: pd.DataFrame, base_features: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in base_features:
            if col not in df.columns:
                continue
            df[f"{col}_rolling_mean"] = (
                df.groupby("engine_id")[col]
                .transform(
                    lambda x: x.rolling(
                        window=self.rolling_window, min_periods=1
                    ).mean()
                )
                .round(3)
            )
        return df

    def _add_diff(self, df: pd.DataFrame, base_features: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in base_features:
            if col not in df.columns:
                continue
            df[f"{col}_diff"] = (
                df.groupby("engine_id")[col]
                .diff()
                .fillna(0)
                .round(3)
            )
        return df

    def _apply_feature_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols = [c for c in self._feature_scaler_cols if c in df.columns]
        df[cols] = self.feature_scaler.transform(df[cols])
        return df

    def _build_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run steps 3-6 of the pipeline on an already-preprocess-scaled df."""
        base_features = [
            c for c in self.feature_cols
            if not c.endswith("_rolling_mean") and not c.endswith("_diff")
        ]

        df = self._select_base_features(df)
        df = self._add_rolling(df, base_features)
        df = self._add_diff(df, base_features)
        df = self._apply_feature_scaler(df)

        # Return only the columns the model was trained on, in the right order
        available = [c for c in self.feature_cols if c in df.columns]
        return df[available]

    # ------------------------------------------------------------------
    # Public transform methods
    # ------------------------------------------------------------------

    def transform_xgb(self, readings: list[dict]) -> np.ndarray:
        """
        Transform a list of sensor dicts into an XGBoost feature matrix.

        For a single-row prediction pass a list with one dict.
        The last row is used as the prediction point (mirrors how the
        training test set takes the last cycle of each engine).
        """
        df = self._to_dataframe(readings)
        df = self._align_to_scaler(df)
        df = self._apply_preprocess_scaler(df)
        X = self._build_feature_matrix(df)

        X.insert(0, "cycle", df["cycle"].iloc[-1])

        # XGBoost predicts on the final cycle row
        return X.iloc[[-1]].values

    def transform_lstm(self, readings: list[dict]) -> np.ndarray:
        """
        Transform an ordered list of sensor dicts into an LSTM sequence tensor.

        Shape: (1, seq_len, n_features) — ready to pass to the model.
        Zero-pads the front if fewer than seq_len cycles are provided.
        """
        df = self._to_dataframe(readings)
        df = self._align_to_scaler(df)
        df = self._apply_preprocess_scaler(df)
        X = self._build_feature_matrix(df)  # shape (n_cycles, n_features)

        vals = X.values
        n, f = vals.shape

        if n >= self.seq_len:
            seq = vals[-self.seq_len:]
        else:
            pad = np.zeros((self.seq_len - n, f), dtype=np.float32)
            seq = np.vstack([pad, vals])

        return seq[np.newaxis, :, :].astype(np.float32)  # (1, seq_len, n_features)