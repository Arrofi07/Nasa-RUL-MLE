"""
⚡ Feature Engineering: selection + rolling + diff + scaling

Feature selection strategy (two-stage)
---------------------------------------
Stage 1 — Correlation filter (fast, model-free):
  Keep features where |Pearson r| with RUL > correlation_threshold.
  Removes sensors that have no linear relationship with degradation.

Stage 2 — XGBoost importance filter (optional, model-based):
  Train a lightweight XGBoost on the correlation-filtered features and
  keep the top `xgb_top_k` by gain importance.
  Catches non-linear relationships missed by Pearson r.
  Disabled when xgb_top_k=None (default keeps all correlation survivors).

This replaces the old hardcoded DROP_COLUMNS list in preprocess.py.
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path("data/processed")

# Meta columns never selected as features
_META_COLS = {"engine_id", "cycle", "rul"}


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


def select_features(
    train_df: pd.DataFrame,
    threshold: float = 0.2,
    xgb_top_k: int | None = None,
) -> list[str]:
    """
    Select features using correlation with RUL, with optional XGBoost
    importance re-ranking.

    Parameters
    ----------
    train_df        DataFrame with a 'rul' column.
    threshold       Minimum |Pearson r| with RUL to survive stage 1.
    xgb_top_k       If set, keep only the top-k features by XGBoost gain
                    importance after the correlation filter.

    Returns
    -------
    List of selected feature column names (excludes engine_id, cycle, rul).
    """
    # Stage 1: correlation filter
    corr = train_df.corr(numeric_only=True)["rul"].abs()
    selected = [c for c in corr[corr > threshold].index if c not in _META_COLS]

    print(
        f"✅ Selected {len(selected)} features after correlation filter (threshold={threshold})"
    )

    if not selected:
        return selected

    # Stage 2: XGBoost importance filter (optional)
    if xgb_top_k is not None and xgb_top_k < len(selected):
        try:
            from xgboost import XGBRegressor

            X = train_df[selected].values
            y = train_df["rul"].values

            xgb = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                verbosity=0,
            )
            xgb.fit(X, y)

            importances = dict(zip(selected, xgb.feature_importances_))
            selected = sorted(importances, key=importances.get, reverse=True)[
                :xgb_top_k
            ]
            selected = sorted(selected, key=lambda c: list(train_df.columns).index(c))

            print(f"✅ Reduced to top {xgb_top_k} features by XGBoost gain importance")
        except ImportError:
            print("⚠️  xgboost not available — skipping importance filter")

    return selected


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------


def add_rolling_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    window: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create rolling mean features to capture degradation trends."""
    for col in features:
        for df in [train_df, test_df]:
            df[f"{col}_rolling_mean"] = (
                df.groupby("engine_id")[col]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
                .round(3)
            )
    print(f"✅ Added rolling mean features (window={window})")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Difference features
# ---------------------------------------------------------------------------


def add_difference_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create cycle-to-cycle degradation features."""
    for col in features:
        for df in [train_df, test_df]:
            df[f"{col}_diff"] = df.groupby("engine_id")[col].diff().fillna(0).round(3)
    print("✅ Added difference features")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize features using train statistics only."""
    feature_cols = [c for c in train_df.columns if c not in _META_COLS]
    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    print(f"✅ Standardized {len(feature_cols)} features")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_features(
    processed_dir: Path | str = PROCESSED_DIR,
    correlation_threshold: float = 0.2,
    xgb_top_k: int | None = None,
    rolling_window: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full feature engineering pipeline."""
    processed_dir = Path(processed_dir)

    train_df = pd.read_csv(processed_dir / "train_clean.csv")
    test_df = pd.read_csv(processed_dir / "test_clean.csv")

    selected = select_features(
        train_df,
        threshold=correlation_threshold,
        xgb_top_k=xgb_top_k,
    )

    train_df = train_df[["engine_id", "cycle"] + selected + ["rul"]]
    test_df = test_df[["engine_id", "cycle"] + selected]

    train_df, test_df = add_rolling_features(
        train_df, test_df, selected, rolling_window
    )
    train_df, test_df = add_difference_features(train_df, test_df, selected)
    train_df, test_df = scale_features(train_df, test_df)

    train_df.to_csv(processed_dir / "feature_engineered_train.csv", index=False)
    test_df.to_csv(processed_dir / "feature_engineered_test.csv", index=False)

    print("✅ Feature engineering completed")
    print(f"   Train: {train_df.shape}")
    print(f"   Test : {test_df.shape}")

    return train_df, test_df


if __name__ == "__main__":
    build_features()
