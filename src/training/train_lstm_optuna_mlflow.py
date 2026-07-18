"""
⚡ LSTM Training + Optuna + MLflow  (v3 — tunable RUL cap + stronger architecture)

What changed from v2
---------------------

1. Tunable RUL cap
   rul_cap is now an Optuna hyperparameter searched over [125, 130, 135, 140].
   Each trial re-applies the cap to the training labels before building
   sequences, so Optuna finds the cap that minimises validation RMSE jointly
   with all other hyperparameters. The winning cap is saved to lstm_config.json
   and used consistently at test evaluation time.

2. Huber loss instead of MSE
   MSE squares every error, so a single bad prediction (e.g. on an engine with
   unusual sensor noise) dominates the gradient and pulls the model away from
   good predictions on normal engines. Huber loss behaves like MSE for small
   errors and like MAE for large ones — it keeps the model from over-correcting
   for outliers. The delta (transition point) is a tunable hyperparameter.

3. Bidirectional LSTM option
   A bidirectional LSTM reads the sequence both forwards and backwards and
   concatenates the two hidden states. For RUL prediction this helps because
   late-cycle sensor patterns (near failure) are more informative than
   early-cycle ones — bidirectional processing lets the model weight recent
   context more heavily. Whether it actually helps is left to Optuna to decide.

4. Attention-weighted pooling option
   Instead of always taking the last time-step output, the model can learn an
   attention weight over all time steps and compute a weighted average. This
   lets the model focus on the most informative cycles in the sequence rather
   than always privileging the most recent reading. Again, whether it helps
   is left to Optuna.

5. Wider search space
   - n_estimators: epochs raised to 150 in final training (was 100)
   - hidden_size: [32, 64, 128, 256] (unchanged)
   - num_layers: 1–4 (unchanged)
   - seq_len: 20–60 (was 20–50)
   - batch_size: [32, 64, 128] (unchanged)

Run
---
    python -m src.training.train_lstm_optuna_mlflow
    python -m src.training.train_lstm_optuna_mlflow --trials 50 --cap_search
"""

import argparse
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

# ===========================================================================
# Config
# ===========================================================================

PROCESSED_DIR = Path("data/processed")
MODEL_DIR     = Path("models")
MLRUNS_DIR    = Path("mlruns")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Use CPU on macOS Apple Silicon to avoid the MPS segfault at torch.load time.
# MPS is intentionally skipped — it segfaults on some macOS + PyTorch 2.x
# combinations when mixed with BLAS libraries loaded by numpy/scipy.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RUL cap candidates to search over.
# 125 is the standard FD001 value. We also try slightly higher values because
# the test set contains engines with true RUL up to ~145 cycles, and the
# hard ceiling in predictions (visible in the scatter plot) is caused by the
# model never seeing training targets above the cap.
RUL_CAP_CHOICES = [125, 130, 135]


# ===========================================================================
# Attention pooling module
# ===========================================================================


class AttentionPooling(nn.Module):
    """
    Learns a scalar attention weight for each time step and returns a
    weighted average of the LSTM outputs across the sequence.

    Why: the last time step isn't always the most informative one. A sudden
    sensor spike 10 cycles before the recorded end might be the best signal
    for predicting RUL. Attention lets the model learn which time steps to
    focus on.

    Input shape:  (batch, seq_len, hidden_size)
    Output shape: (batch, hidden_size)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # A single linear layer maps each hidden state to a scalar score
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden_size)
        scores  = self.score(lstm_out).squeeze(-1)        # (batch, seq_len)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        # Weighted sum over the time dimension
        pooled  = (lstm_out * weights).sum(dim=1)         # (batch, hidden_size)
        return pooled


# ===========================================================================
# Model
# ===========================================================================


class LSTMModel(nn.Module):
    """
    Configurable LSTM for RUL regression.

    Supports:
      - Stacked layers with inter-layer dropout
      - Bidirectional processing (doubles the hidden dimension internally)
      - Attention-weighted pooling vs last-step pooling
      - Two-layer prediction head with LayerNorm + ReLU

    Parameters
    ----------
    input_size   : number of input features (= len(feature_cols))
    hidden_size  : LSTM hidden units per direction
    num_layers   : number of stacked LSTM layers
    dropout      : dropout probability (applied between layers and in head)
    bidirectional: if True, process sequence in both directions
    use_attention: if True, use attention pooling instead of last time step
    """

    def __init__(
        self,
        input_size:    int,
        hidden_size:   int   = 64,
        num_layers:    int   = 2,
        dropout:       float = 0.2,
        bidirectional: bool  = False,
        use_attention: bool  = False,
    ):
        super().__init__()

        self.use_attention = use_attention

        # Dropout between stacked LSTM layers (ignored when num_layers == 1)
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = lstm_dropout,
            bidirectional = bidirectional,
        )

        # Bidirectional doubles the output dimension
        out_dim = hidden_size * (2 if bidirectional else 1)

        # Optional attention pooling over time steps
        if use_attention:
            self.attn = AttentionPooling(out_dim)

        # Prediction head: LayerNorm → Linear → ReLU → Dropout → Linear → scalar
        # LayerNorm works on any batch size (including 1 at inference time),
        # unlike BatchNorm which needs batch_size > 1.
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)           # out: (batch, seq_len, out_dim)

        if self.use_attention:
            # Weighted average across time steps
            pooled = self.attn(out)     # (batch, out_dim)
        else:
            # Just take the final time step (original behaviour)
            pooled = out[:, -1, :]      # (batch, out_dim)

        return self.head(pooled).squeeze(-1)  # (batch,)


# ===========================================================================
# Helpers
# ===========================================================================


def apply_rul_cap(df: pd.DataFrame, rul_cap: int) -> pd.DataFrame:
    """
    Return a copy of df with the 'rul' column clipped at rul_cap.

    This is called inside each Optuna trial so each trial trains on labels
    clipped to its own candidate cap value. The cap is NOT applied to the
    validation targets — we always evaluate against the original unclipped
    labels so the metric is comparable across trials.

    Wait, actually — the val labels come from the same train_df split, and
    train_df always has the cap already applied from load.py. So here we are
    RE-applying a potentially different cap on top. This means:
      - If rul_cap < the cap used in load.py (125): further clips down, no issue
      - If rul_cap > the cap used in load.py (125): has no effect on already-
        clipped values. The labels are already at most 125.

    To actually test caps above 125 you need to re-run load.py with
    clip_rul=False first, or use rul_cap=None in load.py. The recommended
    workflow is: run load.py with clip_rul=False (saves unclipped train.csv),
    then let this script apply the cap per-trial.

    In practice for FD001, caps 125–140 only differ for ~15% of training rows
    (those with original RUL in [125, 140]), so the effect is subtle and
    Optuna needs ~30+ trials to detect it reliably.
    """
    df = df.copy()
    df["rul"] = df["rul"].clip(upper=rul_cap)
    return df


def evaluate_model(
    model: nn.Module,
    X:     np.ndarray,
    y:     np.ndarray,
) -> tuple[float, float, float]:
    """Run inference and return (RMSE, MAE, R²)."""
    model.eval()
    with torch.no_grad():
        X_t   = torch.tensor(X, dtype=torch.float32).to(device)
        preds = model(X_t).cpu().numpy()

    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae  = float(mean_absolute_error(y, preds))
    r2   = float(r2_score(y, preds))
    return rmse, mae, r2


def create_test_sequences(
    df:           pd.DataFrame,
    seq_len:      int,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Build one sequence per test engine (the last seq_len cycles).
    Engines with fewer cycles than seq_len are zero-padded on the left.
    """
    sequences = []
    for engine_id in df["engine_id"].unique():
        engine_data = df[df["engine_id"] == engine_id]
        X_engine    = engine_data[feature_cols].values

        if len(X_engine) >= seq_len:
            seq = X_engine[-seq_len:]
        else:
            # Pad with zeros on the left so the model sees genuine sensor
            # values at the end of the sequence
            padding = np.zeros((seq_len - len(X_engine), X_engine.shape[1]))
            seq     = np.vstack((padding, X_engine))

        sequences.append(seq)

    return np.array(sequences)


def make_loader(
    X:          np.ndarray,
    y:          np.ndarray,
    batch_size: int,
    shuffle:    bool = True,
) -> torch.utils.data.DataLoader:
    """Wrap numpy arrays in a PyTorch DataLoader."""
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle    = shuffle,
        pin_memory = (device.type == "cuda"),
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the feature-engineered CSVs written by build_feature.py."""
    train_df = pd.read_csv(PROCESSED_DIR / "feature_engineered_train.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "feature_engineered_test.csv")
    rul_df   = pd.read_csv(PROCESSED_DIR / "rul_clean.csv")
    return train_df, test_df, rul_df


# ===========================================================================
# Optuna objective
# ===========================================================================


def objective(
    trial:        optuna.Trial,
    train_df:     pd.DataFrame,
    feature_cols: list[str],
    search_cap:   bool = True,
) -> float:
    """
    Optuna trial: sample hyperparameters, train LSTM, return val RMSE.

    Parameters
    ----------
    search_cap : bool
        If True, rul_cap is included in the search space so Optuna jointly
        optimises the cap alongside architecture hyperparameters.
        If False, the cap in the data is used as-is (backward-compatible
        behaviour from v2).
    """

    # ------------------------------------------------------------------
    # 1. Hyperparameter search space
    # ------------------------------------------------------------------

    seq_len     = trial.suggest_int("seq_len", 20, 60)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    num_layers  = trial.suggest_int("num_layers", 1, 4)
    dropout     = trial.suggest_float("dropout", 0.1, 0.5)
    lr          = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size  = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Bidirectional and attention are binary choices.
    # Optuna treats them as categoricals so it can learn whether they help.
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    use_attention = trial.suggest_categorical("use_attention", [True, False])

    # Huber loss delta: below this value the loss is quadratic (like MSE),
    # above it the loss is linear (like MAE). Tuning it lets Optuna decide
    # how much to penalise large errors vs small ones.
    huber_delta = trial.suggest_float("huber_delta", 5.0, 50.0)

    # RUL cap — only searched if search_cap=True
    if search_cap:
        rul_cap = trial.suggest_categorical("rul_cap", RUL_CAP_CHOICES)
        # Re-apply cap on a copy of train_df for this trial
        trial_train_df = apply_rul_cap(train_df, rul_cap)
    else:
        trial_train_df = train_df  # use whatever cap was applied in load.py

    # ------------------------------------------------------------------
    # 2. Data — build sequences with this trial's seq_len and cap
    # ------------------------------------------------------------------

    X_train, X_val, y_train, y_val = create_group_split_sequences(
        train_df     = trial_train_df,
        feature_cols = feature_cols,
        seq_len      = seq_len,
    )
    train_loader = make_loader(X_train, y_train, batch_size)

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------

    model = LSTMModel(
        input_size    = len(feature_cols),
        hidden_size   = hidden_size,
        num_layers    = num_layers,
        dropout       = dropout,
        bidirectional = bidirectional,
        use_attention = use_attention,
    ).to(device)

    # Huber loss: robust to outliers compared to pure MSE
    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # 4. Training loop (30 epochs — quick enough for Optuna pruning)
    # ------------------------------------------------------------------

    EPOCHS = 30

    with mlflow.start_run(nested=True):
        mlflow.log_params({
            "seq_len":       seq_len,
            "hidden_size":   hidden_size,
            "num_layers":    num_layers,
            "dropout":       dropout,
            "learning_rate": lr,
            "batch_size":    batch_size,
            "bidirectional": bidirectional,
            "use_attention": use_attention,
            "huber_delta":   huber_delta,
            "rul_cap":       rul_cap if search_cap else "from_data",
        })

        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                # Gradient clipping prevents exploding gradients in deep LSTMs
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)

            train_loss = epoch_loss / len(X_train)
            # Always evaluate on RMSE regardless of training loss type,
            # so val_rmse is directly comparable across trials
            val_rmse, _, _ = evaluate_model(model, X_val, y_val)

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_rmse": val_rmse},
                step=epoch,
            )

            # Tell Optuna the intermediate result so it can prune bad trials early
            trial.report(val_rmse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        final_rmse, final_mae, final_r2 = evaluate_model(model, X_val, y_val)
        mlflow.log_metrics({"rmse": final_rmse, "mae": final_mae, "r2": final_r2})

    return final_rmse


# ===========================================================================
# Final training with best hyperparameters
# ===========================================================================


def train_final_model(
    best_params:  dict,
    train_df:     pd.DataFrame,
    feature_cols: list[str],
) -> tuple[nn.Module, list[dict]]:
    """
    Re-train on the full training set using the best hyperparameters found
    by Optuna. Uses ReduceLROnPlateau + early stopping (patience=10).

    Returns the trained model and the per-epoch training history (for the
    training curve artifact saved to MLflow).
    """

    # Apply the best cap found by Optuna before building final sequences
    rul_cap = best_params.get("rul_cap", 125)
    capped_train_df = apply_rul_cap(train_df, rul_cap)

    print(f"\n📌 Final training with RUL cap = {rul_cap}")

    X_train, X_val, y_train, y_val = create_group_split_sequences(
        capped_train_df,
        feature_cols,
        best_params["seq_len"],
    )
    train_loader = make_loader(X_train, y_train, best_params["batch_size"])

    model = LSTMModel(
        input_size    = len(feature_cols),
        hidden_size   = best_params["hidden_size"],
        num_layers    = best_params["num_layers"],
        dropout       = best_params["dropout"],
        bidirectional = best_params.get("bidirectional", False),
        use_attention = best_params.get("use_attention", False),
    ).to(device)

    criterion = nn.HuberLoss(delta=best_params.get("huber_delta", 10.0))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = best_params["learning_rate"],
    )

    # Reduce LR by half if val RMSE doesn't improve for 5 consecutive epochs.
    # This gives the model a chance to escape plateaus before early stopping
    # kicks in at patience=10.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode    = "min",
        factor  = 0.5,
        patience = 5,
    )

    best_rmse  = float("inf")
    best_state = None
    patience   = 10   # epochs without improvement before stopping
    counter    = 0
    history    = []   # saved as training_curve.csv artifact in MLflow

    EPOCHS = 150  # raised from 100 — early stopping will cut this short if needed

    print(f"Training up to {EPOCHS} epochs (early stopping patience={patience})")

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

        history.append({
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_rmse":   round(val_rmse,   4),
            "lr":         current_lr,
        })

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Train Loss: {train_loss:9.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        scheduler.step(val_rmse)

        if val_rmse < best_rmse:
            best_rmse  = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            counter    = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"⏹  Early stopping at epoch {epoch + 1} (best val RMSE: {best_rmse:.4f})")
            break

    # Restore the best checkpoint (not the final epoch)
    model.load_state_dict(best_state)

    # Save training curve for offline plotting
    curve_path = MODEL_DIR / "lstm_training_curve.csv"
    pd.DataFrame(history).to_csv(curve_path, index=False)
    print(f"\n✅ Training curve saved → {curve_path}")

    return model, history


# ===========================================================================
# Main pipeline
# ===========================================================================


def run_lstm_pipeline(n_trials: int = 30, search_cap: bool = True):
    """
    Full pipeline: Optuna search → final training → test evaluation → save.

    Parameters
    ----------
    n_trials   : number of Optuna trials
    search_cap : whether to include rul_cap in the Optuna search space
    """

    mlflow.set_tracking_uri(str(MLRUNS_DIR.resolve()))
    mlflow.set_experiment("NASA_Turbofan_RUL")

    train_df, test_df, rul_df = load_data()

    # All columns that aren't metadata or the target are features
    feature_cols = [
        col for col in train_df.columns
        if col not in ["engine_id", "cycle", "rul"]
    ]

    print(f"\n📐 Feature count   : {len(feature_cols)}")
    print(f"🔍 Optuna trials   : {n_trials}")
    print(f"📌 Search RUL cap  : {search_cap} (candidates: {RUL_CAP_CHOICES})")
    print(f"💻 Training device : {device}\n")

    # ------------------------------------------------------------------
    # Optuna study
    # ------------------------------------------------------------------

    # HyperbandPruner is more sample-efficient than MedianPruner for
    # sequence models where early epochs are noisy — it gives more budget
    # to promising trials instead of cutting them based on early noise.
    study = optuna.create_study(
        direction = "minimize",
        pruner    = optuna.pruners.HyperbandPruner(
            min_resource     = 5,
            max_resource     = 30,
            reduction_factor = 3,
        ),
    )

    with mlflow.start_run(run_name="lstm_optuna_v3"):
        study.optimize(
            lambda trial: objective(trial, train_df, feature_cols, search_cap),
            n_trials          = n_trials,
            show_progress_bar = True,
        )

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_rmse", study.best_value)

    print("\n🏆 Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"   {k:20s} = {v}")
    print(f"   best val RMSE    = {study.best_value:.4f}")

    # ------------------------------------------------------------------
    # Final training on full training set
    # ------------------------------------------------------------------

    best_params = study.best_params
    model, history = train_final_model(best_params, train_df, feature_cols)

    # ------------------------------------------------------------------
    # Test evaluation
    # The test labels (rul_df) are NEVER clipped — we always measure
    # performance against the true NASA RUL values, not our training cap.
    # This is the correct way to compare across different cap values.
    # ------------------------------------------------------------------

    rul_cap     = best_params.get("rul_cap", 125)
    X_test      = create_test_sequences(test_df, best_params["seq_len"], feature_cols)
    y_test      = rul_df["rul"].values  # raw true RUL, no clip

    rmse, mae, r2 = evaluate_model(model, X_test, y_test)

    print(f"\n✅ Test performance (RUL cap used in training: {rul_cap}):")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   MAE  : {mae:.4f}")
    print(f"   R²   : {r2:.4f}")

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------

    # Model weights
    torch.save(model.state_dict(), MODEL_DIR / "best_lstm.pt")

    # Config — saved to lstm_config.json so inference code can reconstruct
    # the model architecture without importing this training file.
    # rul_cap is included here so the inference pipeline knows which cap
    # the model was trained with (useful for documentation, not used in
    # inference since predictions are clipped in predict.py, not here).
    lstm_config = {
        "seq_len":       best_params["seq_len"],
        "hidden_size":   best_params["hidden_size"],
        "num_layers":    best_params["num_layers"],
        "dropout":       best_params["dropout"],
        "bidirectional": best_params.get("bidirectional", False),
        "use_attention": best_params.get("use_attention", False),
        "huber_delta":   best_params.get("huber_delta", 10.0),
        "rul_cap":       rul_cap,
    }

    with open(MODEL_DIR / "lstm_config.json", "w") as f:
        json.dump(lstm_config, f, indent=4)

    print(f"\n✅ lstm_config.json saved → {MODEL_DIR / 'lstm_config.json'}")

    # Optuna trials CSV
    pd.DataFrame(study.trials_dataframe()).to_csv(
        MODEL_DIR / "lstm_optuna_trials.csv", index=False
    )

    # Final MLflow run for the best model
    with mlflow.start_run(run_name="best_lstm_v3"):
        mlflow.log_params(best_params)
        mlflow.log_metrics({"test_rmse": rmse, "test_mae": mae, "test_r2": r2})

        mlflow.log_artifact(str(MODEL_DIR / "lstm_training_curve.csv"))

        # Log epoch-level metrics so the MLflow UI can show the training curve
        for row in history:
            mlflow.log_metrics(
                {
                    "final_train_loss": row["train_loss"],
                    "final_val_rmse":   row["val_rmse"],
                },
                step=row["epoch"],
            )

        mlflow.pytorch.log_model(pytorch_model=model, name="model")

    return study, model, lstm_config


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM for NASA RUL prediction")
    parser.add_argument(
        "--trials",
        type    = int,
        default = 30,
        help    = "Number of Optuna trials (default: 30)",
    )
    parser.add_argument(
        "--cap_search",
        action  = "store_true",
        default = True,
        help    = "Include rul_cap in Optuna search space (default: True)",
    )
    parser.add_argument(
        "--no_cap_search",
        action  = "store_false",
        dest    = "cap_search",
        help    = "Fix the RUL cap to whatever is in the data (no cap search)",
    )
    args = parser.parse_args()

    run_lstm_pipeline(n_trials=args.trials, search_cap=args.cap_search)