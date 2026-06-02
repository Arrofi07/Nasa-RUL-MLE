"""
⚡ Baseline Model Training for NASA Turbofan (RUL)

- Reads feature-engineered datasets from data/processed/
- Creates validation split
- Trains baseline models:
    - Linear Regression
    - Ridge Regression
    - Random Forest
    - XGBoost
- Evaluates on validation and test sets
- Saves results and best model

"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(model, X, y):
    """Compute RMSE, MAE and R²."""

    preds = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    return rmse, mae, r2


def load_data(processed_dir: Path | str = PROCESSED_DIR):
    """Load engineered datasets."""

    processed_dir = Path(processed_dir)

    train_df = pd.read_csv(
        processed_dir / "feature_engineered_train.csv"
    )

    test_df = pd.read_csv(
        processed_dir / "feature_engineered_test.csv"
    )

    rul_df = pd.read_csv(
        processed_dir / "rul_clean.csv"
    )

    return train_df, test_df, rul_df


def prepare_data(train_df, test_df, rul_df):
    """Create train/validation/test matrices."""

    X = train_df.drop(
        columns=["engine_id", "rul"]
    )

    y = train_df["rul"]

    test_last = (
        test_df.groupby("engine_id")
        .last()
        .reset_index()
    )

    X_test = test_last.drop(
        columns=["engine_id"]
    )

    y_test = rul_df["rul"]

    return X, y, X_test, y_test


def train_models(
    X_train,
    y_train,
):
    """Train baseline models."""

    models = {
        "Linear": LinearRegression(),

        "Ridge": Ridge(
            alpha=1.0
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),

        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ),
    }

    for name, model in models.items():
        print(f"🚀 Training {name}")
        model.fit(X_train, y_train)

    return models


def evaluate_models(
    models,
    X_val,
    y_val,
    X_test,
    y_test,
):
    """Evaluate all models."""

    results = []

    for name, model in models.items():

        rmse_val, mae_val, r2_val = evaluate(
            model,
            X_val,
            y_val,
        )

        rmse_test, mae_test, r2_test = evaluate(
            model,
            X_test,
            y_test,
        )

        results.append(
            {
                "Model": name,
                "RMSE_Val": rmse_val,
                "RMSE_Test": rmse_test,
                "MAE_Val": mae_val,
                "MAE_Test": mae_test,
                "R2_Val": r2_val,
                "R2_Test": r2_test,
            }
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values("RMSE_Test")
        .reset_index(drop=True)
    )

    return results_df


def save_best_model(
    models,
    results_df,
    output_dir: Path | str = MODEL_DIR,
):
    """Save best model based on test RMSE."""

    output_dir = Path(output_dir)

    best_model_name = results_df.iloc[0]["Model"]

    best_model = models[best_model_name]

    model_path = (
        output_dir
        / f"best_{best_model_name.lower().replace(' ', '_')}.pkl"
    )

    joblib.dump(best_model, model_path)

    print(
        f"✅ Best model: {best_model_name}"
    )

    print(
        f"✅ Saved model to {model_path}"
    )

    return best_model_name, model_path


def train_pipeline(
    processed_dir: Path | str = PROCESSED_DIR,
    model_dir: Path | str = MODEL_DIR,
):
    """Run full training pipeline."""

    train_df, test_df, rul_df = load_data(
        processed_dir
    )

    X, y, X_test, y_test = prepare_data(
        train_df,
        test_df,
        rul_df,
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
    )

    models = train_models(
        X_train,
        y_train,
    )

    results_df = evaluate_models(
        models,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    results_df.to_csv(
        Path(model_dir) / "baseline_results.csv",
        index=False,
    )

    print("\n📊 Model Results")
    print(results_df)

    save_best_model(
        models,
        results_df,
        model_dir,
    )

    return results_df


if __name__ == "__main__":
    train_pipeline()