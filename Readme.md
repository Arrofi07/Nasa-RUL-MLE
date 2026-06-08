# NASA Turbofan RUL Prediction

An end-to-end machine learning engineering project for predicting the **Remaining Useful Life (RUL)** of turbofan engines using the NASA CMAPSS FD001 dataset. The project follows ML engineering best practices with modular pipelines, experiment tracking via MLflow, hyperparameter tuning via Optuna, and a production-ready REST API built with FastAPI.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Models & Results](#models--results)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Common Commands](#common-commands)
- [Development Roadmap](#development-roadmap)

---

## Project Overview

Turbofan engines degrade over time. Predicting when an engine will fail — and how many cycles remain — allows operators to schedule maintenance proactively and avoid costly unplanned downtime.

This project builds a full ML pipeline that takes raw sensor readings from a running engine and outputs a predicted RUL in operational cycles. Two models are trained and served:

- **XGBoost** — tree-based model tuned with Optuna, served on single or batched readings
- **LSTM** — sequence model that captures temporal degradation patterns across engine cycles

---

## Architecture

```
Raw Sensor Data
      │
      ▼
┌─────────────────┐
│   Data Loading  │  load.py
│  (preprocess)   │  preprocess.py
└────────┬────────┘
         │  Drop low-info sensors, normalize
         ▼
┌─────────────────┐
│    Feature      │  build_feature.py
│  Engineering    │  sequence_builder.py
└────────┬────────┘
         │  Correlation selection, rolling mean, diff features
         ▼
┌─────────────────────────────────┐
│          Model Training         │
│                                 │
│  XGBoost + Optuna + MLflow      │  tune_xgb_optuna_mlflow.py
│  LSTM    + Optuna + MLflow      │  train_lstm_optuna_mlflow.py
└────────┬────────────────────────┘
         │  Saved: best_xgb.pkl, best_lstm.pt
         ▼
┌─────────────────┐
│  Artifact       │  export_artifacts.py
│  Export         │  Saves scalers + feature list for inference
└────────┬────────┘
         ▼
┌─────────────────┐
│   FastAPI       │  src/api/app.py
│   REST API      │  src/inference/
└─────────────────┘
```

---

## Dataset

**NASA CMAPSS Turbofan Engine Degradation Simulation — FD001 subset**

| Split     | Engines | Rows   | Description                              |
|-----------|---------|--------|------------------------------------------|
| Train     | 100     | 20,631 | Full run-to-failure cycles               |
| Test      | 100     | 13,096 | Cycles stopped before failure            |
| RUL       | 100     | 100    | True RUL for each test engine at cutoff  |

Each row contains 3 operational settings and 21 sensor measurements per cycle. Engine lifetimes range from 128 to 362 cycles (mean: 206 cycles).

**Preprocessing:**
- 6 low-information sensors dropped (`sensor_1`, `5`, `10`, `16`, `18`, `19`)
- RUL clipped at 125 cycles during training (standard FD001 practice)
- Features standardized using `StandardScaler` fitted on train only

**Feature Engineering:**
- Correlation-based feature selection (threshold `|r| > 0.2` with RUL)
- Rolling mean features (window = 5 cycles) to smooth degradation signal
- Cycle-to-cycle difference features to capture rate of change
- Final feature matrix: sensors + rolling means + diffs

---

## Pipeline

```
notebooks/00_load.ipynb                →  Load raw FD001 data
notebooks/01_eda_cleaning.ipynb        →  EDA, sensor analysis, cleaning
notebooks/02_feature_engineering.ipynb →  Feature selection & engineering
notebooks/03_baseline_model.ipynb      →  Linear, Ridge, RF, XGBoost baselines
notebooks/04_tuning_xgb_optuna_mlflow.ipynb  →  XGBoost tuning + MLflow
notebooks/05_model_lstm.ipynb          →  LSTM tuning + MLflow
```

Production scripts mirror the notebooks step by step:

```
src/data/load.py             →  Load & compute RUL labels
src/data/preprocess.py       →  Clean, drop sensors, normalize
src/features/build_feature.py        →  Feature engineering
src/features/sequence_builder.py     →  LSTM sequence construction
src/training/train_baseline.py       →  Baseline model comparison
src/training/tune_xgb_optuna_mlflow.py  →  XGBoost tuning pipeline
src/training/train_lstm_optuna_mlflow.py →  LSTM tuning pipeline
scripts/export_artifacts.py          →  Export scalers for inference
```

---

## Models & Results

### Baseline Comparison (Test Set)

| Model         | RMSE  | MAE   | R²    |
|---------------|-------|-------|-------|
| Linear        | 22.23 | 17.34 | 0.770 |
| Ridge         | 22.23 | 17.34 | 0.770 |
| Random Forest | 19.11 | 14.06 | 0.789 |
| XGBoost       | 19.31 | 14.15 | —     |

### After Tuning (Optuna + MLflow)

| Model            | Val RMSE | Test RMSE | Test MAE | Test R² |
|------------------|----------|-----------|----------|---------|
| XGBoost (tuned)  | 12.44    | 20.26     | 14.59    | 0.762   |
| LSTM (tuned)     | 3.71     | 17.46     | 12.83    | 0.823   |

**Best XGBoost hyperparameters** (`models/best_params_xgb.json`):

```json
{
    "n_estimators": 487,
    "max_depth": 4,
    "learning_rate": 0.014,
    "subsample": 0.607,
    "colsample_bytree": 0.822,
    "min_child_weight": 4,
    "gamma": 2.628
}
```

**Best LSTM configuration** (`models/lstm_config.json`):

```json
{
    "seq_len": 41,
    "hidden_size": 64,
    "num_layers": 3
}
```

All experiments are tracked in MLflow. Run `mlflow ui` to explore runs.

---

## API Endpoints

| Method | Endpoint              | Model   | Description                                      |
|--------|-----------------------|---------|--------------------------------------------------|
| GET    | `/health`             | —       | Server status and which models are loaded        |
| POST   | `/predict/xgb`        | XGBoost | Single-cycle prediction                          |
| POST   | `/predict/xgb/batch`  | XGBoost | Multi-cycle prediction (uses rolling/diff features) |
| POST   | `/predict/lstm`       | LSTM    | Sequence-based prediction (41 cycles recommended) |

Interactive docs available at **http://localhost:8000/docs** after starting the server.

**Example request** (`/predict/xgb`):

```bash
curl -X POST http://localhost:8000/predict/xgb \
  -H "Content-Type: application/json" \
  -d '{
    "engine_id": 1,
    "cycle": 50,
    "setting_1": -0.0007,
    "setting_2": -0.0004,
    "setting_3": 100.0,
    "sensor_1": 518.67, "sensor_2": 641.82, "sensor_3": 1589.70,
    "sensor_4": 1400.60, "sensor_5": 14.62, "sensor_6": 21.61,
    "sensor_7": 554.36, "sensor_8": 2388.02, "sensor_9": 9046.19,
    "sensor_10": 1.30, "sensor_11": 47.47, "sensor_12": 521.66,
    "sensor_13": 2388.02, "sensor_14": 8138.62, "sensor_15": 8.4195,
    "sensor_16": 0.03, "sensor_17": 392.0, "sensor_18": 2388.0,
    "sensor_19": 100.0, "sensor_20": 39.06, "sensor_21": 23.419
  }'
```

**Example response:**

```json
{
  "engine_id": 1,
  "cycle": 50,
  "predicted_rul": 87.43,
  "model": "xgboost"
}
```

---

## Project Structure

```
nasa-rul-mle/
│
├── main.py                          # Uvicorn entry point
├── pyproject.toml                   # Dependencies and tool config
│
├── data/
│   ├── raw/                         # train_FD001.txt, test_FD001.txt, RUL_FD001.txt
│   └── processed/                   # Cleaned and feature-engineered CSVs
│
├── models/
│   ├── best_xgb.pkl                 # Trained XGBoost model
│   ├── best_lstm.pt                 # Trained LSTM weights
│   ├── best_params_xgb.json         # Best XGBoost hyperparameters
│   ├── lstm_config.json             # LSTM architecture config
│   ├── preprocess_scaler.pkl        # Scaler from preprocessing step
│   ├── feature_scaler.pkl           # Scaler from feature engineering step
│   └── feature_cols.txt             # Ordered feature list for inference
│
├── notebooks/
│   ├── 00_load.ipynb
│   ├── 01_eda_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_tuning_xgb_optuna_mlflow.ipynb
│   └── 05_model_lstm.ipynb
│
├── src/
│   ├── data/
│   │   ├── load.py                  # Load raw data and compute RUL labels
│   │   └── preprocess.py            # Drop sensors, normalize
│   │
│   ├── features/
│   │   ├── build_feature.py         # Feature selection and engineering
│   │   └── sequence_builder.py      # LSTM sequence construction
│   │
│   ├── training/
│   │   ├── train_baseline.py        # Baseline model comparison
│   │   ├── tune_xgb_optuna_mlflow.py
│   │   └── train_lstm_optuna_mlflow.py
│   │
│   ├── inference/
│   │   ├── schemas.py               # Pydantic request/response models
│   │   ├── pipeline.py              # Feature transformation at inference
│   │   └── predict.py               # Model loader and prediction functions
│   │
│   └── api/
│       └── app.py                   # FastAPI routes
│
├── scripts/
│   └── export_artifacts.py          # Export scalers after training
│
├── mlruns/                          # MLflow experiment tracking
└── tests/                           # (coming soon)
```

---

## Setup & Installation

**Prerequisites:** Python 3.13, the NASA FD001 dataset files.

```bash
# 1. Clone the repository
git clone https://github.com/Arrofi07/nasa-rul-mle.git
cd nasa-rul-mle

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Place the raw dataset files in data/raw/
#    train_FD001.txt  test_FD001.txt  RUL_FD001.txt
```

> The dataset is available from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

---

## Common Commands

### Testing
```bash
# Run all tests
PYTHONPATH=. pytest tests/ -v

# With coverage:
PYTHONPATH=. pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test modules  
PYTHONPATH=. pytest tests/conftest.py
PYTHONPATH=. pytest tests/test_data_quality.py
PYTHONPATH=. pytest tests/test_features.py
PYTHONPATH=. pytest tests/test_training.py
PYTHONPATH=. pytest tests/test_inference.py

### Data Pipeline

```bash
python -m src.data.load
python -m src.data.preprocess
python -m src.features.build_feature
```

### Training

```bash
# Baseline comparison
python -m src.training.train_baseline

# XGBoost tuning (tracked in MLflow)
python -m src.training.tune_xgb_optuna_mlflow

# LSTM tuning (tracked in MLflow)
python -m src.training.train_lstm_optuna_mlflow
```

### Export Artifacts (run once after training)

```bash
python -m scripts.export_artifacts
```

### Start the API

```bash
# Development (auto-reload)
python main.py

# Or directly
uvicorn src.api.app:app --reload --port 8000
```

### Start Dashboard
# Terminal 1 — keep this running
python main.py

# Terminal 2
streamlit run app.py
# Opens: http://localhost:8501

### MLflow UI

```bash
mlflow ui
# Open: http://localhost:5000
```

---

## Development Roadmap

- [x] Data loading and preprocessing pipeline
- [x] Feature engineering (rolling, diff, selection)
- [x] Baseline model comparison
- [x] XGBoost tuning with Optuna + MLflow
- [x] LSTM training with Optuna + MLflow
- [x] FastAPI inference API
- [ ] Unit and integration tests
- [ ] Dockerfile and container deployment
- [ ] CI/CD with GitHub Actions