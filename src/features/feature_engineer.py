"""
⚡ Feature Engineering Script for NASA Turbofan (RUL)

- Reads cleaned train/test/rul CSVs from data/processed/
- Performs correlation-based feature selection
- Adds rolling mean features to capture degradation trends
- Adds difference features to capture cycle-to-cycle changes
- Normalizes all features
- Saves feature-engineered datasets back to data/processed/
"""

"""
Feature Engineering: selection + rolling stats + diff + scaling

- Production defaults read from data/processed/
- Tests can override `processed_dir`
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data(processed_dir: Path):
    """Load cleaned datasets."""
    train_df = pd.read_csv(processed_dir / "train_clean.csv")
    test_df = pd.read_csv(processed_dir / "test_clean.csv")
    rul_df = pd.read_csv(processed_dir / "rul_clean.csv")

    print("📂 Data loaded:")
    print(f"   Train: {train_df.shape}")
    print(f"   Test: {test_df.shape}")
    print(f"   RUL: {rul_df.shape}")

    return train_df, test_df, rul_df


def select_features(train_df: pd.DataFrame, threshold: float = 0.2):
    """Select features based on correlation with RUL."""
    corr = train_df.corr(numeric_only=True)["rul"].abs()

    selected_features = corr[corr > threshold].index.tolist()

    # Remove non-feature columns
    for col in ["cycle", "rul"]:
        if col in selected_features:
            selected_features.remove(col)

    print(f"✅ Selected {len(selected_features)} features (corr > {threshold})")
    return selected_features


def apply_feature_selection(train_df, test_df, selected_features):
    """Keep only selected features."""
    train_df = train_df[["engine_id", "cycle"] + selected_features + ["rul"]]
    test_df = test_df[["engine_id", "cycle"] + selected_features]

    print("✅ Applied feature selection")
    print(f"   Train: {train_df.shape}")
    print(f"   Test: {test_df.shape}")

    return train_df, test_df


def add_rolling_features(train_df, test_df, selected_features, window=5):
    """Add rolling mean features per engine."""
    for col in selected_features:
        train_df[f"{col}_rolling_mean"] = (
            train_df.groupby("engine_id")[col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean().round(3))
        )

        test_df[f"{col}_rolling_mean"] = (
            test_df.groupby("engine_id")[col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean().round(3))
        )

    print(f"✅ Added rolling mean features (window={window})")
    return train_df, test_df


def add_diff_features(train_df, test_df, selected_features):
    """Add cycle-to-cycle difference features."""
    for col in selected_features:
        train_df[f"{col}_diff"] = (
            train_df.groupby("engine_id")[col].diff().fillna(0).round(3)
        )
        test_df[f"{col}_diff"] = (
            test_df.groupby("engine_id")[col].diff().fillna(0).round(3)
        )

    print("✅ Added difference features")
    return train_df, test_df


def normalize_features(train_df, test_df):
    """Standardize all numerical features."""
    features = [
        col for col in train_df.columns
        if col not in ["engine_id", "cycle", "rul"]
    ]

    scaler = StandardScaler()

    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])

    print(f"✅ Normalized {len(features)} features")
    return train_df, test_df


def save_data(train_df, test_df, processed_dir: Path):
    """Save feature-engineered datasets."""
    train_df.to_csv(processed_dir / "feature_engineered_train.csv", index=False)
    test_df.to_csv(processed_dir / "feature_engineered_test.csv", index=False)

    print("💾 Saved feature-engineered datasets")


def feature_engineering_pipeline(
    processed_dir: Path | str = PROCESSED_DIR,
    corr_threshold: float = 0.2,
    window: int = 5,
):
    """Main feature engineering pipeline."""

    processed_dir = Path(processed_dir)

    # Load
    train_df, test_df, rul_df = load_data(processed_dir)

    # Feature selection
    selected_features = select_features(train_df, corr_threshold)
    train_df, test_df = apply_feature_selection(train_df, test_df, selected_features)

    # Feature engineering
    train_df, test_df = add_rolling_features(train_df, test_df, selected_features, window)
    train_df, test_df = add_diff_features(train_df, test_df, selected_features)

    # Normalization
    train_df, test_df = normalize_features(train_df, test_df)

    # Save
    save_data(train_df, test_df, processed_dir)

    print("🚀 Feature engineering completed")
    print(f"   Final Train: {train_df.shape}")
    print(f"   Final Test: {test_df.shape}")

    return train_df, test_df, rul_df


if __name__ == "__main__":
    feature_engineering_pipeline()