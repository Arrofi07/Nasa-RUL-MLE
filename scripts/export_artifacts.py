"""
Export training artifacts needed by the inference pipeline.

Run this ONCE after training is complete:

    python scripts/export_artifacts.py

What it does
------------
1. Re-runs the preprocessing and feature-engineering pipelines on the
   training data to recover the fitted scalers.
2. Saves two scalers to models/:
     - preprocess_scaler.pkl  (StandardScaler from preprocess.py)
     - feature_scaler.pkl     (StandardScaler from build_feature.py)
3. Saves the ordered list of model feature columns to:
     - models/feature_cols.txt

These three files are the bridge between the training code and the API.
Without them, the API cannot replicate the exact feature transformation
that was applied during training.

Why not just pickle the whole pipeline?
----------------------------------------
Your training scripts apply the scalers inline (mutating DataFrames directly)
rather than returning fitted scaler objects, so there's no single Pipeline
object to save.  This script extracts the scalers by re-fitting them on the
full training data — which is correct because fit() on the same data always
produces the same result.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


from src.features.build_feature import select_features

# ---------------------------------------------------------------------------
# Paths — change these if your layout differs
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1: Preprocess scaler
# ---------------------------------------------------------------------------

print("📦 Fitting preprocess scaler …")

train_raw = pd.read_csv(RAW_DIR / "train.csv")


# Remove constant columns (same logic as remove_constant_features)
nunique = train_raw.nunique()
constant_cols = nunique[nunique <= 1].index.tolist()
if constant_cols:
    train_raw = train_raw.drop(columns=constant_cols)

EXCLUDE = ["engine_id", "cycle", "rul"]
preprocess_feature_cols = [c for c in train_raw.columns if c not in EXCLUDE]

preprocess_scaler = StandardScaler()
preprocess_scaler.fit(train_raw[preprocess_feature_cols])

joblib.dump(preprocess_scaler, MODEL_DIR / "preprocess_scaler.pkl")
print(f"  ✅ Saved preprocess_scaler.pkl  ({len(preprocess_feature_cols)} features)")

# ---------------------------------------------------------------------------
# Step 2: Feature-engineering scaler
# ---------------------------------------------------------------------------

print("📦 Fitting feature scaler …")

# Load the already-preprocessed (but not yet feature-engineered) clean CSV
train_clean = pd.read_csv(PROCESSED_DIR / "train_clean.csv")

# Select base features (correlation threshold must match build_feature.py default)
selected_features = select_features(train_clean, threshold=0.2)

train_clean = train_clean[["engine_id", "cycle"] + selected_features + ["rul"]]

# Add rolling mean  (window=5 matches build_feature.py default)
WINDOW = 5
for col in selected_features:
    train_clean[f"{col}_rolling_mean"] = (
        train_clean.groupby("engine_id")[col]
        .transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())
        .round(3)
    )

# Add diff features
for col in selected_features:
    train_clean[f"{col}_diff"] = (
        train_clean.groupby("engine_id")[col].diff().fillna(0).round(3)
    )

# Columns to scale (everything except meta columns)
FE_EXCLUDE = ["engine_id", "cycle", "rul"]
feature_scaler_cols = [c for c in train_clean.columns if c not in FE_EXCLUDE]

feature_scaler = StandardScaler()
feature_scaler.fit(train_clean[feature_scaler_cols])

joblib.dump(feature_scaler, MODEL_DIR / "feature_scaler.pkl")
print(f"  ✅ Saved feature_scaler.pkl  ({len(feature_scaler_cols)} features)")

# ---------------------------------------------------------------------------
# Step 3: Feature column list (what the model was actually trained on)
# ---------------------------------------------------------------------------

print("📦 Saving feature column list …")

# Load the feature-engineered training data to read the exact column order
# that was passed to XGBoost / LSTM
fe_train = pd.read_csv(PROCESSED_DIR / "feature_engineered_train.csv")
model_feature_cols = [
    c for c in fe_train.columns if c not in ["engine_id", "cycle", "rul"]
]

with open(MODEL_DIR / "feature_cols.txt", "w") as f:
    for col in model_feature_cols:
        f.write(col + "\n")

print(f"  ✅ Saved feature_cols.txt  ({len(model_feature_cols)} columns)")

print("\n🎉 All artifacts exported.  You can now start the API:\n")
print("   uvicorn src.api.app:app --reload")
