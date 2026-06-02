"""
⚡ XGBoost Hyperparameter Tuning with Optuna + MLflow

- Reads feature-engineered datasets
- Uses GroupShuffleSplit to avoid engine leakage
- Tunes XGBoost with Optuna
- Tracks experiments in MLflow
- Logs best model and metrics
- Saves Optuna study results

"""

from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor


PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MLRUNS_DIR = Path("mlruns")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data(
    processed_dir: Path | str = PROCESSED_DIR,
):
    """Load feature-engineered datasets."""

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


def create_group_split(
    train_df: pd.DataFrame,
    test_size: float = 0.2,
):
    """
    Split by engine_id to avoid leakage.
    """

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=42,
    )

    train_idx, val_idx = next(
        splitter.split(
            train_df,
            groups=train_df["engine_id"],
        )
    )

    train_subset = train_df.iloc[train_idx]
    val_subset = train_df.iloc[val_idx]

    X_train = train_subset.drop(
        columns=["engine_id", "rul"]
    )

    y_train = train_subset["rul"]

    X_val = val_subset.drop(
        columns=["engine_id", "rul"]
    )

    y_val = val_subset["rul"]

    return X_train, X_val, y_train, y_val


def create_test_set(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
):
    """
    Create official NASA test set.
    """

    test_last = (
        test_df.groupby("engine_id")
        .last()
        .reset_index()
    )

    X_test = test_last.drop(
        columns=["engine_id"]
    )

    y_test = rul_df["rul"]

    return X_test, y_test


def objective(
    trial,
    X_train,
    y_train,
    X_val,
    y_val,
):
    """Optuna objective."""

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            200,
            1000,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.5,
            1.0,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.5,
            1.0,
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10,
        ),
        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            5.0,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-8,
            10.0,
            log=True,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-8,
            10.0,
            log=True,
        ),
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    with mlflow.start_run(nested=True):

        model = XGBRegressor(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)

        rmse = np.sqrt(
            mean_squared_error(
                y_val,
                preds,
            )
        )

        mae = mean_absolute_error(
            y_val,
            preds,
        )

        r2 = r2_score(
            y_val,
            preds,
        )

        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        )

    return rmse


def tune_model(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials: int = 30,
):
    """Run Optuna tuning."""

    study = optuna.create_study(
        direction="minimize"
    )

    with mlflow.start_run(
        run_name="xgboost_optuna"
    ):
        study.optimize(
            lambda trial: objective(
                trial,
                X_train,
                y_train,
                X_val,
                y_val,
            ),
            n_trials=n_trials,
        )

        mlflow.log_params(
            study.best_params
        )

        mlflow.log_metric(
            "best_rmse",
            study.best_value,
        )

    return study


def train_final_model(
    study,
    X_train,
    y_train,
):
    """Train final model."""

    model = XGBRegressor(
        **study.best_params,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def evaluate_test(
    model,
    X_test,
    y_test,
):
    """Evaluate on NASA holdout test."""

    preds = model.predict(X_test)

    metrics = {
        "rmse": np.sqrt(
            mean_squared_error(
                y_test,
                preds,
            )
        ),
        "mae": mean_absolute_error(
            y_test,
            preds,
        ),
        "r2": r2_score(
            y_test,
            preds,
        ),
    }

    return metrics

def save_model(
    model,
    output_dir: Path | str = MODEL_DIR,
):

    output_dir = Path(output_dir)

    model_path = output_dir / "best_xgboost_model.pkl"

    joblib.dump(
        model,
        model_path,
    )

    print(f"✅ Model saved to {model_path}")

    return model_path


def run_tuning_pipeline(
    processed_dir: Path | str = PROCESSED_DIR,
    experiment_name: str = "NASA_Turbofan_RUL",
    n_trials: int = 30,
):

    mlflow.set_tracking_uri(
        str(MLRUNS_DIR.resolve())
    )

    mlflow.set_experiment(
        experiment_name
    )

    train_df, test_df, rul_df = load_data(
        processed_dir
    )

    X_train, X_val, y_train, y_val = (
        create_group_split(
            train_df
        )
    )

    X_test, y_test = create_test_set(
        test_df,
        rul_df,
    )

    study = tune_model(
        X_train,
        y_train,
        X_val,
        y_val,
        n_trials=n_trials,
    )

    final_model = train_final_model(
        study,
        X_train,
        y_train,
    )

    save_model(final_model)

    metrics = evaluate_test(
        final_model,
        X_test,
        y_test,
    )

    with mlflow.start_run(
        run_name="best_xgboost_model"
    ):

        mlflow.log_params(
            study.best_params
        )

        mlflow.log_metrics(
            metrics
        )

        mlflow.xgboost.log_model(
            final_model,
            name="model",
        )

    pd.DataFrame(
        study.trials_dataframe()
    ).to_csv(
        MODEL_DIR / "xgb_optuna_trials.csv",
        index=False,
    )

    print("✅ Best Parameters")
    print(study.best_params)

    print("\n✅ Test Metrics")
    print(metrics)

    return study, final_model, metrics


if __name__ == "__main__":
    run_tuning_pipeline()