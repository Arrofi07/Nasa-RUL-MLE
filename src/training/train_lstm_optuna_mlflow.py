"""
⚡ LSTM Training + Optuna + MLflow  (upgraded)

Improvements over v1
---------------------
Architecture
  - Dropout between LSTM layers (tunable, prevents overfitting)
  - Deeper prediction head: BatchNorm → Linear(hidden, hidden//2) → ReLU → Linear → output
  - Wider hyperparameter search: hidden_size up to 256, num_layers up to 4

Training
  - DataLoader used consistently (Optuna loop and final training loop)
  - ReduceLROnPlateau scheduler: backs off LR when val RMSE plateaus
  - Early stopping patience increased 5 → 10 to give the scheduler room
  - Gradient clipping kept at max_norm=1.0

Optuna
  - HyperbandPruner replaces MedianPruner (more sample-efficient for sequences)
  - dropout added as a tunable hyperparameter
  - batch_size search space extended to [32, 64, 128]
  - n_trials default raised to 30

MLflow
  - Train loss and val RMSE logged every epoch (visible in the UI chart view)
  - training_curve.csv saved as an artifact for offline plotting
"""

import copy
import json
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.sequence_builder import create_group_split_sequences

# ==================================================
# Config
# ==================================================

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MLRUNS_DIR = Path("mlruns")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================================================
# Model
# ==================================================


class LSTMModel(nn.Module):
    """
    Multi-layer LSTM with dropout and a two-layer prediction head.

    Architecture
    ------------
    LSTM (num_layers, dropout between layers)
        └─ last time-step output
            └─ BatchNorm1d
                └─ Linear(hidden_size → hidden_size // 2)
                    └─ ReLU
                        └─ Linear(hidden_size // 2 → 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # dropout only applied between stacked layers (ignored when num_layers=1)
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),  # works on any batch size, including 1
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time-step: (batch, hidden_size)
        return self.head(out).squeeze(-1)


# ==================================================
# Evaluation
# ==================================================


def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        preds = model(X_tensor).cpu().numpy()

    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))
    return rmse, mae, r2


# ==================================================
# Test sequence creation
# ==================================================


def create_test_sequences(
    df: pd.DataFrame,
    seq_len: int,
    feature_cols: list[str],
) -> np.ndarray:
    sequences = []
    for engine_id in df["engine_id"].unique():
        engine_data = df[df["engine_id"] == engine_id]
        X_engine = engine_data[feature_cols].values

        if len(X_engine) >= seq_len:
            seq = X_engine[-seq_len:]
        else:
            padding = np.zeros((seq_len - len(X_engine), X_engine.shape[1]))
            seq = np.vstack((padding, X_engine))

        sequences.append(seq)
    return np.array(sequences)


# ==================================================
# DataLoader helper
# ==================================================


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=(device.type == "cuda"),
    )


# ==================================================
# Load data
# ==================================================


def load_data():
    train_df = pd.read_csv(PROCESSED_DIR / "feature_engineered_train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "feature_engineered_test.csv")
    rul_df = pd.read_csv(PROCESSED_DIR / "rul_clean.csv")
    return train_df, test_df, rul_df


# ==================================================
# Optuna objective
# ==================================================


def objective(trial: optuna.Trial, train_df: pd.DataFrame, feature_cols: list[str]):
    # ---- Hyperparameter search space ----
    seq_len = trial.suggest_int("seq_len", 20, 50)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # ---- Data ----
    X_train, X_val, y_train, y_val = create_group_split_sequences(
        train_df=train_df,
        feature_cols=feature_cols,
        seq_len=seq_len,
    )
    train_loader = make_loader(X_train, y_train, batch_size)

    # ---- Model ----
    model = LSTMModel(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- Training loop ----
    EPOCHS = 30

    with mlflow.start_run(nested=True):
        mlflow.log_params(
            {
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "learning_rate": lr,
                "batch_size": batch_size,
            }
        )

        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)

            train_loss = epoch_loss / len(X_train)
            val_rmse, _, _ = evaluate_model(model, X_val, y_val)

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_rmse": val_rmse}, step=epoch
            )

            trial.report(val_rmse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        final_rmse, final_mae, final_r2 = evaluate_model(model, X_val, y_val)
        mlflow.log_metrics({"rmse": final_rmse, "mae": final_mae, "r2": final_r2})

    return final_rmse


# ==================================================
# Final training with best params
# ==================================================


def train_final_model(
    best_params: dict,
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> nn.Module:
    """
    Re-trains on the full train split using best hyperparameters.

    Uses ReduceLROnPlateau and early stopping (patience=10) so the
    scheduler has room to reduce the LR before giving up.
    """
    X_train, X_val, y_train, y_val = create_group_split_sequences(
        train_df,
        feature_cols,
        best_params["seq_len"],
    )

    train_loader = make_loader(X_train, y_train, best_params["batch_size"])

    model = LSTMModel(
        input_size=len(feature_cols),
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])

    # Backs off LR by 0.5 if val RMSE doesn't improve for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        # verbose=True,
    )

    best_rmse = float("inf")
    best_state = None
    patience = 10
    counter = 0
    history = []  # for the training curve artifact

    EPOCHS = 100

    print(
        f"\nFinal training (up to {EPOCHS} epochs, early stopping patience={patience})"
    )

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        train_loss = epoch_loss / len(X_train)
        val_rmse, _, _ = evaluate_model(model, X_val, y_val)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "val_rmse": round(val_rmse, 4),
                "lr": current_lr,
            }
        )

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Train Loss: {train_loss:10.2f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        scheduler.step(val_rmse)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)

    # Save training curve for offline plotting
    curve_path = MODEL_DIR / "lstm_training_curve.csv"
    pd.DataFrame(history).to_csv(curve_path, index=False)
    print(f"\nTraining curve saved → {curve_path}")

    return model, history


# ==================================================
# Main pipeline
# ==================================================


def run_lstm_pipeline(n_trials: int = 30):
    mlflow.set_tracking_uri(str(MLRUNS_DIR.resolve()))
    mlflow.set_experiment("NASA_Turbofan_RUL")

    train_df, test_df, rul_df = load_data()

    feature_cols = [
        col for col in train_df.columns if col not in ["engine_id", "cycle", "rul"]
    ]

    # ---- Optuna study ----
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=5,
            max_resource=30,
            reduction_factor=3,
        ),
    )

    with mlflow.start_run(run_name="lstm_optuna"):
        study.optimize(
            lambda trial: objective(trial, train_df, feature_cols),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_rmse", study.best_value)

    print("\nBest params:", study.best_params)
    print(f"Best val RMSE: {study.best_value:.4f}")

    # ---- Final training ----
    best_params = study.best_params
    model, history = train_final_model(best_params, train_df, feature_cols)

    # ---- Test evaluation ----
    X_test = create_test_sequences(test_df, best_params["seq_len"], feature_cols)
    y_test = rul_df["rul"].values

    rmse, mae, r2 = evaluate_model(model, X_test, y_test)

    print("\nFinal LSTM performance on test data:")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")

    # ---- Save artifacts ----
    torch.save(model.state_dict(), MODEL_DIR / "best_lstm.pt")

    pd.DataFrame(study.trials_dataframe()).to_csv(
        MODEL_DIR / "lstm_optuna_trials.csv", index=False
    )

    with open(MODEL_DIR / "lstm_config.json", "w") as f:
        json.dump(
            {
                "seq_len": best_params["seq_len"],
                "hidden_size": best_params["hidden_size"],
                "num_layers": best_params["num_layers"],
                "dropout": best_params["dropout"],
            },
            f,
            indent=4,
        )

    # ---- Log final run to MLflow ----
    with mlflow.start_run(run_name="best_lstm_model"):
        mlflow.log_params(best_params)
        mlflow.log_metrics({"test_rmse": rmse, "test_mae": mae, "test_r2": r2})

        # Log training curve artifact
        mlflow.log_artifact(str(MODEL_DIR / "lstm_training_curve.csv"))

        # Log epoch-level metrics for the UI chart view
        for row in history:
            mlflow.log_metrics(
                {
                    "final_train_loss": row["train_loss"],
                    "final_val_rmse": row["val_rmse"],
                },
                step=row["epoch"],
            )

        mlflow.pytorch.log_model(pytorch_model=model, name="model")


if __name__ == "__main__":
    run_lstm_pipeline()
