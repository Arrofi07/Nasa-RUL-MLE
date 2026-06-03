"""
⚡ LSTM Training + Optuna + MLflow

- Reads feature engineered dataset
- Creates sequences using GroupShuffleSplit
- Tunes LSTM with Optuna
- Tracks experiments with MLflow
- Uses Early Stopping
- Saves best model
- Logs best model to MLflow

"""

from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import json

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.features.sequence_builder import (
    create_group_split_sequences,
)

# ==================================================
# Config
# ==================================================

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MLRUNS_DIR = Path("mlruns")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ==================================================
# Model
# ==================================================


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(
            hidden_size,
            1,
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out.squeeze(-1)


# ==================================================
# Evaluation
# ==================================================


def evaluate_model(
    model,
    X,
    y,
):

    model.eval()

    with torch.no_grad():

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32,
        ).to(device)

        preds = (
            model(X_tensor)
            .cpu()
            .numpy()
        )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            preds,
        )
    )

    mae = mean_absolute_error(
        y,
        preds,
    )

    r2 = r2_score(
        y,
        preds,
    )

    return rmse, mae, r2


# ==================================================
# Test Sequence Creation
# ==================================================


def create_test_sequences(
    df,
    seq_len,
    feature_cols,
):

    sequences = []

    for engine_id in df["engine_id"].unique():

        engine_data = df[
            df["engine_id"] == engine_id
        ]

        X_engine = engine_data[
            feature_cols
        ].values

        if len(X_engine) >= seq_len:

            seq = X_engine[-seq_len:]

        else:

            padding = np.zeros(
                (
                    seq_len - len(X_engine),
                    X_engine.shape[1],
                )
            )

            seq = np.vstack(
                (
                    padding,
                    X_engine,
                )
            )

        sequences.append(seq)

    return np.array(sequences)


# ==================================================
# Load Data
# ==================================================


def load_data():

    train_df = pd.read_csv(
        PROCESSED_DIR /
        "feature_engineered_train.csv"
    )

    test_df = pd.read_csv(
        PROCESSED_DIR /
        "feature_engineered_test.csv"
    )

    rul_df = pd.read_csv(
        PROCESSED_DIR /
        "rul_clean.csv"
    )

    return train_df, test_df, rul_df


# ==================================================
# Optuna Objective
# ==================================================


def objective(
    trial,
    train_df,
    feature_cols,
):

    seq_len = trial.suggest_int(
        "seq_len",
        20,
        50,
    )

    hidden_size = (
        trial.suggest_categorical(
            "hidden_size",
            [32, 64, 128],
        )
    )

    num_layers = (
        trial.suggest_int(
            "num_layers",
            1,
            3,
        )
    )

    lr = (
        trial.suggest_float(
            "learning_rate",
            1e-4,
            1e-2,
            log=True,
        )
    )

    batch_size = (
        trial.suggest_categorical(
            "batch_size",
            [32, 64],
        )
    )

    X_train, X_val, y_train, y_val = (
        create_group_split_sequences(
            train_df=train_df,
            feature_cols=feature_cols,
            seq_len=seq_len,
        )
    )

    train_dataset = (
        torch.utils.data.TensorDataset(
            torch.tensor(
                X_train,
                dtype=torch.float32,
            ),
            torch.tensor(
                y_train,
                dtype=torch.float32,
            ),
        )
    )

    train_loader = (
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
    )

    model = LSTMModel(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    with mlflow.start_run(
        nested=True
    ):

        mlflow.log_params(
            {
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "learning_rate": lr,
                "batch_size": batch_size,
            }
        )

        EPOCHS = 20

        for epoch in range(EPOCHS):

            model.train()

            for xb, yb in train_loader:

                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()

                preds = model(xb)

                loss = criterion(
                    preds,
                    yb,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

            rmse, _, _ = evaluate_model(
                model,
                X_val,
                y_val,
            )

            trial.report(
                rmse,
                epoch,
            )

            if trial.should_prune():
                raise optuna.TrialPruned()

        rmse, mae, r2 = evaluate_model(
            model,
            X_val,
            y_val,
        )

        mlflow.log_metrics(
            {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        )

    return rmse


# ==================================================
# Main Pipeline
# ==================================================


def run_lstm_pipeline(
    n_trials=20,
):

    mlflow.set_tracking_uri(
        str(
            MLRUNS_DIR.resolve()
        )
    )

    mlflow.set_experiment(
        "NASA_Turbofan_RUL"
    )

    train_df, test_df, rul_df = (
        load_data()
    )

    feature_cols = [
        col
        for col in train_df.columns
        if col
        not in [
            "engine_id",
            "cycle",
            "rul",
        ]
    ]

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )

    with mlflow.start_run(
        run_name="lstm_optuna"
    ):

        study.optimize(
            lambda trial: objective(
                trial,
                train_df,
                feature_cols,
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

    # ==========================================
    # Final Training
    # ==========================================

    best_params = study.best_params

    X_train, X_val, y_train, y_val = (
        create_group_split_sequences(
            train_df,
            feature_cols,
            best_params["seq_len"],
        )
    )

    model = LSTMModel(
        input_size=len(feature_cols),
        hidden_size=best_params[
            "hidden_size"
        ],
        num_layers=best_params[
            "num_layers"
        ],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params[
            "learning_rate"
        ],
    )

    criterion = nn.MSELoss()

    best_rmse = float("inf")
    patience = 5
    counter = 0

    best_state = None

    for epoch in range(50):

        model.train()

        for i in range(
            0,
            len(X_train),
            best_params["batch_size"],
        ):

            xb = torch.tensor(
                X_train[
                    i:
                    i + best_params["batch_size"]
                ],
                dtype=torch.float32,
            ).to(device)

            yb = torch.tensor(
                y_train[
                    i:
                    i + best_params["batch_size"]
                ],
                dtype=torch.float32,
            ).to(device)

            optimizer.zero_grad()

            preds = model(xb)

            loss = criterion(
                preds,
                yb,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

        rmse, _, _ = evaluate_model(
            model,
            X_val,
            y_val,
        )

        if rmse < best_rmse:

            best_rmse = rmse
            best_state = model.state_dict()
            counter = 0

        else:

            counter += 1

        if counter >= patience:

            print(
                "Early stopping"
            )

            break

    model.load_state_dict(
        best_state
    )

    # ==========================================
    # Test Evaluation
    # ==========================================

    X_test = create_test_sequences(
        test_df,
        best_params["seq_len"],
        feature_cols,
    )

    y_test = rul_df["rul"].values

    rmse, mae, r2 = evaluate_model(
        model,
        X_test,
        y_test,
    )

    torch.save(
        model.state_dict(),
        MODEL_DIR /
        "best_lstm.pt",
    )

    pd.DataFrame(
        study.trials_dataframe()
    ).to_csv(
        MODEL_DIR /
        "lstm_optuna_trials.csv",
        index=False,
    )

    with open(
        "models/lstm_config.json",
        "w",
    ) as f:

        json.dump(
            {
                "seq_len":
                    best_params["seq_len"],
                "hidden_size":
                    best_params["hidden_size"],
                "num_layers":
                    best_params["num_layers"],
            },
            f,
            indent=4,
        )

    with mlflow.start_run(
        run_name="best_lstm_model"
    ):

        mlflow.log_params(
            best_params
        )

        mlflow.log_metrics(
            {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        )

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
        )

    print("\nBest Params:")
    print(best_params)

    print("\nTest Metrics:")
    print(
        {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }
    )


if __name__ == "__main__":
    run_lstm_pipeline()