"""
⚡ Preprocessing Script for NASA Turbofan (RUL)

- Reads train/test/rul CSVs from data/raw/
- Cleans column names and removes unused sensors
- Performs basic feature selection (constant / low variance removal)
- Optionally normalizes sensor values
- Saves cleaned datasets to data/processed/

"""

"""
Preprocessing: cleaning + feature selection + normalization

- Production defaults read from data/raw/ and write to data/processed/
- Tests can override `raw_dir`, `processed_dir`
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Known useless sensors (constant or near-constant in FD001)
DROP_COLUMNS = [
    "sensor_1", "sensor_5", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19"
]


def drop_unused_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Remove sensors with no predictive power."""
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"✅ Dropped {len(cols_to_drop)} low-informative sensors.")
    return df


def remove_constant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with zero variance."""
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
):
    """Standardize numerical features using train statistics only."""

    scaler = StandardScaler()

    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    print(f"✅ Normalized {len(feature_cols)} features.")
    return train_df, test_df


def preprocess_data(
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    normalize: bool = True,
):
    """Main preprocessing pipeline."""

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load
    train_df = pd.read_csv(raw_dir / "train.csv")
    test_df = pd.read_csv(raw_dir / "test.csv")
    rul_df = pd.read_csv(raw_dir / "rul.csv")

    # Cleaning
    train_df = drop_unused_sensors(train_df)
    test_df = drop_unused_sensors(test_df)

    train_df = remove_constant_features(train_df)
    test_df = remove_constant_features(test_df)

    # Normalization
    if normalize:
        train_df, test_df = normalize_features(train_df, test_df)

    # Save outputs
    train_df.to_csv(processed_dir / "train_clean.csv", index=False)
    test_df.to_csv(processed_dir / "test_clean.csv", index=False)
    rul_df.to_csv(processed_dir / "rul_clean.csv", index=False)

    print(f"✅ Preprocessing completed (saved to {processed_dir})")
    print(f"   Train: {train_df.shape}")
    print(f"   Test: {test_df.shape}")

    return train_df, test_df, rul_df


if __name__ == "__main__":
    preprocess_data()