"""
Preprocessing: cleaning + normalization

Changes from v1
---------------
- drop_unused_sensors() REMOVED — no more hardcoded sensor blacklist.
  All sensors now pass through to build_feature.py where selection is
  data-driven (correlation + XGBoost feature importance).
- remove_constant_features() still runs to drop zero-variance columns.
- normalize_features() unchanged — train-only fit, no test leakage.
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def remove_constant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with zero variance (useless for any model)."""
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        df = df.drop(columns=constant_cols)
        print(f"✅ Removed constant columns: {constant_cols}")
    else:
        print("✅ No constant columns found.")
    return df


def normalize_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exclude_cols: list[str] = ["engine_id", "cycle", "rul"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize numerical features using train statistics only."""
    scaler = StandardScaler()
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])
    print(f"✅ Normalized {len(feature_cols)} features.")
    return train_df, test_df


def preprocess_data(
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    normalize: bool = True,
):
    """Main preprocessing pipeline (no hardcoded sensor drops)."""
    raw_dir       = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(raw_dir / "train.csv")
    test_df  = pd.read_csv(raw_dir / "test.csv")
    rul_df   = pd.read_csv(raw_dir / "rul.csv")

    # Drop zero-variance columns (determined from train, applied to both)
    constant_cols = [
        c for c in train_df.columns
        if train_df[c].nunique() <= 1
        and c not in ["engine_id", "cycle", "rul"]
    ]
    if constant_cols:
        train_df = train_df.drop(columns=constant_cols)
        # Only drop columns that also exist in test
        test_df  = test_df.drop(columns=[c for c in constant_cols if c in test_df.columns])
        print(f"✅ Removed constant columns: {constant_cols}")
    else:
        print("✅ No constant columns found.")

    if normalize:
        train_df, test_df = normalize_features(train_df, test_df)

    train_df.to_csv(processed_dir / "train_clean.csv", index=False)
    test_df.to_csv( processed_dir / "test_clean.csv",  index=False)
    rul_df.to_csv(  processed_dir / "rul_clean.csv",   index=False)

    print(f"✅ Preprocessing completed (saved to {processed_dir})")
    print(f"   Train: {train_df.shape}")
    print(f"   Test:  {test_df.shape}")

    return train_df, test_df, rul_df


if __name__ == "__main__":
    preprocess_data()