"""
⚡ Feature Engineering Script for NASA Turbofan (RUL)

- Reads cleaned train/test datasets from data/processed/
- Selects informative features based on RUL correlation
- Creates rolling mean features
- Creates difference (degradation) features
- Standardizes engineered features
- Saves feature-engineered datasets to data/processed/

"""

"""
Feature Engineering: feature selection + rolling features + degradation features

- Production defaults read from data/processed/ and write to data/processed/
- Tests can override `processed_dir`
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path("data/processed")


def select_features(
    train_df: pd.DataFrame,
    threshold: float = 0.2,
) -> list[str]:
    """
    Select features correlated with RUL above threshold.
    """

    corr = train_df.corr(numeric_only=True)["rul"].abs()

    selected_features = corr[corr > threshold].index.tolist()

    for col in ["cycle", "rul"]:
        if col in selected_features:
            selected_features.remove(col)

    print(
        f"✅ Selected {len(selected_features)} features "
        f"(correlation threshold = {threshold})"
    )

    return selected_features


def add_rolling_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    window: int = 5,
):
    """
    Create rolling mean features to capture degradation trends.
    """

    for col in features:
        train_df[f"{col}_rolling_mean"] = (
            train_df.groupby("engine_id")[col]
            .transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=1,
                ).mean()
            )
            .round(3)
        )

        test_df[f"{col}_rolling_mean"] = (
            test_df.groupby("engine_id")[col]
            .transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=1,
                ).mean()
            )
            .round(3)
        )

    print(f"✅ Added rolling mean features (window={window})")

    return train_df, test_df


def add_difference_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
):
    """
    Create cycle-to-cycle degradation features.
    """

    for col in features:
        train_df[f"{col}_diff"] = (
            train_df.groupby("engine_id")[col]
            .diff()
            .fillna(0)
            .round(3)
        )

        test_df[f"{col}_diff"] = (
            test_df.groupby("engine_id")[col]
            .diff()
            .fillna(0)
            .round(3)
        )

    print("✅ Added difference features")

    return train_df, test_df


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Standardize features using train statistics only.
    """

    feature_cols = [
        col
        for col in train_df.columns
        if col not in ["engine_id", "cycle", "rul"]
    ]

    scaler = StandardScaler()

    train_df[feature_cols] = scaler.fit_transform(
        train_df[feature_cols]
    )

    test_df[feature_cols] = scaler.transform(
        test_df[feature_cols]
    )

    print(f"✅ Standardized {len(feature_cols)} features")

    return train_df, test_df


def build_features(
    processed_dir: Path | str = PROCESSED_DIR,
    correlation_threshold: float = 0.2,
    rolling_window: int = 5,
):
    """
    Run full feature engineering pipeline.
    """

    processed_dir = Path(processed_dir)

    train_df = pd.read_csv(processed_dir / "train_clean.csv")
    test_df = pd.read_csv(processed_dir / "test_clean.csv")

    # Feature Selection
    selected_features = select_features(
        train_df,
        threshold=correlation_threshold,
    )

    train_df = train_df[
        ["engine_id", "cycle"]
        + selected_features
        + ["rul"]
    ]

    test_df = test_df[
        ["engine_id", "cycle"]
        + selected_features
    ]

    # Rolling Features
    train_df, test_df = add_rolling_features(
        train_df,
        test_df,
        selected_features,
        window=rolling_window,
    )

    # Difference Features
    train_df, test_df = add_difference_features(
        train_df,
        test_df,
        selected_features,
    )

    # Scaling
    train_df, test_df = scale_features(
        train_df,
        test_df,
    )

    # Save
    train_df.to_csv(
        processed_dir / "feature_engineered_train.csv",
        index=False,
    )

    test_df.to_csv(
        processed_dir / "feature_engineered_test.csv",
        index=False,
    )

    print("✅ Feature engineering completed")
    print(f"   Train: {train_df.shape}")
    print(f"   Test : {test_df.shape}")

    return train_df, test_df


if __name__ == "__main__":
    build_features()