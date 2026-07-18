# ✈️ NASA Turbofan — Remaining Useful Life Prediction

[![CI](https://github.com/Arrofi07/NASA-RUL-MLE/actions/workflows/ci.yml/badge.svg)](https://github.com/Arrofi07/NASA-RUL-MLE/actions/workflows/ci.yml)
[![Live API](https://img.shields.io/badge/API-Railway-6366f1?logo=railway)](https://nasa-rul-mle-production.up.railway.app/health)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b?logo=streamlit)](https://nasa-rul-dashboard.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python)](https://www.python.org)

An end-to-end **Machine Learning Engineering** project that predicts the Remaining Useful Life (RUL) of turbofan engines from raw sensor readings. Built on the NASA CMAPSS FD001 dataset with a full ML pipeline, three trained models, a production REST API, and a live interactive dashboard.

**Live links:**
- 🌐 **Dashboard** — [nasa-rul-dashboard.streamlit.app](https://nasa-rul-dashboard.streamlit.app)
- 🔌 **API** — [nasa-rul-mle-production.up.railway.app](https://nasa-rul-mle-production.up.railway.app/health)
- 📖 **Swagger UI** — [/docs](https://nasa-rul-mle-production.up.railway.app/docs)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Models & Results](#models--results)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Common Commands](#common-commands)
- [CI/CD](#cicd)
- [Deployment](#deployment)
- [Roadmap](#roadmap)

---

## Overview

Turbofan engines degrade over time through wear and stress on components. Predicting **how many operational cycles remain** before failure allows maintenance teams to act proactively — avoiding costly unplanned downtime and unnecessary early replacements.

This project builds a complete ML pipeline from raw sensor data to live predictions:

- **Three models trained and served:** XGBoost, LightGBM, and LSTM
- **Feature engineering pipeline** reproduced faithfully at inference (no data leakage)
- **FastAPI REST API** deployed on Railway — all three models available via HTTP
- **Streamlit dashboard** for interactive exploration, batch evaluation, and live prediction
- **Experiment tracking** with MLflow + hyperparameter tuning with Optuna
- **CI/CD** with GitHub Actions → automatic Railway deployment on push

---

## Architecture

```
Raw Sensor Data (NASA FD001)
         │
         ▼
┌─────────────────────┐
│   Data Pipeline     │  load.py → preprocess.py
│                     │  Drop low-info sensors, normalise, compute RUL labels
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Feature Engineering│  build_feature.py, sequence_builder.py
│                     │  Correlation selection, rolling mean, diff features
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────────────────┐
│             Model Training           │
│                                      │
│  XGBoost  + Optuna + MLflow          │
│  LightGBM + Optuna + MLflow          │
│  LSTM     + Optuna + MLflow          │
└──────────┬───────────────────────────┘
           │  Saved: best_xgb.pkl, best_lgbm.pkl, best_lstm.pt
           ▼
┌─────────────────────┐
│  Artifact Export    │  export_artifacts.py
│                     │  Scalers + feature list saved for inference
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────────────────┐
│         Production System            │
│                                      │
│  FastAPI (Railway)                   │
│    └── /predict/xgb                  │
│    └── /predict/lgbm                 │
│    └── /predict/lstm                 │
│                                      │
│  Streamlit Dashboard (Streamlit Cloud)│
│    └── Engine Explorer               │
│    └── Batch Evaluation              │
│    └── Live Prediction               │
│    └── Model Comparison              │
└──────────────────────────────────────┘
```

**Key design decisions:**

- The inference pipeline (`src/inference/pipeline.py`) reproduces every training transformation — scaler, feature selection, rolling mean, diff features — without refitting. This prevents double-scaling and guarantees predictions match training behaviour.
- `OMP_NUM_THREADS=1` + `KMP_DUPLICATE_LIB_OK=TRUE` are set before any import in `main.py` to prevent an OpenMP SIGSEGV on macOS Apple Silicon caused by XGBoost and LightGBM each loading their own `libomp.dylib`.
- The Streamlit app trims readings to the last 5 cycles (tree models) and 41 cycles (LSTM) before sending to the API — avoids serialising full engine histories (up to 300+ rows) on every request.

---

## Dataset

**NASA CMAPSS Turbofan Engine Degradation Simulation — FD001 subset**

| Split | Engines | Rows   | Description                             |
|-------|---------|--------|-----------------------------------------|
| Train | 100     | 20,631 | Full run-to-failure cycles              |
| Test  | 100     | 13,096 | Cycles stopped before failure           |
| RUL   | 100     | 100    | True RUL for each test engine at cutoff |

Each row contains 3 operational settings and 21 sensor measurements per cycle. Engine lifetimes range from 128 to 362 cycles (mean: 206 cycles).

**Preprocessing:**
- 6 low-information sensors dropped (`sensor_1`, `5`, `10`, `16`, `18`, `19`) based on near-zero variance
- RUL clipped at 125 cycles during training (standard FD001 practice — engines are healthy early on)
- Features standardised with `StandardScaler` fitted on train only, applied to test

**Feature Engineering:**
- Correlation-based feature selection (threshold `|r| > 0.2` with RUL)
- Rolling mean (window = 5 cycles) to smooth degradation signal
- Cycle-to-cycle difference features to capture rate of change
- Final feature matrix: 15 base sensors + rolling means + diffs = 43 features

> Dataset available at the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

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
| LightGBM (tuned) | 14.81    | 18.36     | 13.32    | 0.805   |
| LSTM (tuned)     | 3.71     | 17.46     | 12.83    | 0.823   |

LSTM achieves the best test performance by capturing temporal degradation patterns across the full engine history. LightGBM outperforms XGBoost on the test set despite a worse validation RMSE — the tree ensemble generalises better on the held-out engines.

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

**Best LightGBM configuration**(`models/best_params_lgbm.json`)

```json
{
    "learning_rate": 0.0497598982744983,
    "num_leaves": 205,
    "max_depth": 4,
    "min_child_samples": 35,
    "feature_fraction": 0.8460949396176609,
    "bagging_fraction": 0.7399677958277459,
    "lambda_l1": 1.2237182584986743e-05,
    "lambda_l2": 1.62555911245314e-05
}
```

**Best LSTM configuration** (`models/lstm_config.json`):

```json
{
    "seq_len": 41,
    "hidden_size": 64,
    "num_layers": 3,
    "dropout": 0.2
}
```

All experiments are logged in MLflow. Run `mlflow ui` to explore runs, compare hyperparameters, and view metrics.

---

## API Reference

Base URL: `https://nasa-rul-mle-production.up.railway.app`

| Method | Endpoint             | Model    | Description                                          |
|--------|----------------------|----------|------------------------------------------------------|
| GET    | `/health`            | —        | Liveness check + which models are loaded             |
| GET    | `/docs`              | —        | Interactive Swagger UI                               |
| POST   | `/predict/xgb`       | XGBoost  | Single-cycle prediction                              |
| POST   | `/predict/xgb/batch` | XGBoost  | Multi-cycle prediction (rolling/diff features apply) |
| POST   | `/predict/lgbm`      | LightGBM | Single-cycle prediction                              |
| POST   | `/predict/lgbm/batch`| LightGBM | Multi-cycle prediction                               |
| POST   | `/predict/lstm`      | LSTM     | Sequence-based prediction (41 cycles recommended)    |

### Example: Single-cycle prediction

```bash
curl -X POST https://nasa-rul-mle-production.up.railway.app/predict/xgb \
  -H "Content-Type: application/json" \
  -d '{
    "engine_id": 1,
    "cycle": 50,
    "setting_1": -0.0007,
    "setting_2": -0.0004,
    "setting_3": 100.0,
    "sensor_1": 518.67,  "sensor_2": 641.82,  "sensor_3": 1589.70,
    "sensor_4": 1400.60, "sensor_5": 14.62,   "sensor_6": 21.61,
    "sensor_7": 554.36,  "sensor_8": 2388.02, "sensor_9": 9046.19,
    "sensor_10": 1.30,   "sensor_11": 47.47,  "sensor_12": 521.66,
    "sensor_13": 2388.02,"sensor_14": 8138.62,"sensor_15": 8.4195,
    "sensor_16": 0.03,   "sensor_17": 392.0,  "sensor_18": 2388.0,
    "sensor_19": 100.0,  "sensor_20": 39.06,  "sensor_21": 23.419
  }'
```

```json
{
  "engine_id": 1,
  "cycle": 50,
  "predicted_rul": 87.43,
  "model": "xgboost"
}
```

### Example: Batch prediction with history

```bash
curl -X POST https://nasa-rul-mle-production.up.railway.app/predict/lgbm/batch \
  -H "Content-Type: application/json" \
  -d '{
    "readings": [
      { "engine_id": 1, "cycle": 48, "setting_1": -0.0007, ... },
      { "engine_id": 1, "cycle": 49, "setting_1": -0.0007, ... },
      { "engine_id": 1, "cycle": 50, "setting_1": -0.0007, ... }
    ]
  }'
```

Send readings **oldest → newest**. The API predicts RUL for the last reading in the list, using earlier cycles to compute rolling and difference features. At least 5 cycles recommended; the pipeline zero-pads if fewer are sent.

---

## Project Structure

```
nasa-rul-mle/
│
├── main.py                              # Uvicorn entry point (sets OpenMP env vars)
├── app.py                               # Streamlit dashboard
├── pyproject.toml                       # Dependencies + tool config (single source of truth)
├── uv.lock                              # Locked dependency versions
├── Dockerfile                           # Container for Railway deployment
├── docker-compose.yml
│
├── data/
│   ├── raw/                             # train_FD001.txt, test_FD001.txt, RUL_FD001.txt
│   └── processed/                       # Cleaned and feature-engineered CSVs
│
├── models/
│   ├── best_xgb.pkl                     # Trained XGBoost model
│   ├── best_lgbm.pkl                    # Trained LightGBM model
│   ├── best_lstm.pt                     # Trained LSTM weights
│   ├── best_params_xgb.json             # Best XGBoost hyperparameters
│   ├── best_params_lgbm.json            # Best LightGBM hyperparameters
│   ├── lstm_config.json                 # LSTM architecture + seq_len
│   ├── preprocess_scaler.pkl            # StandardScaler from preprocessing
│   ├── feature_scaler.pkl               # StandardScaler from feature engineering
│   └── feature_cols.txt                 # Ordered feature list for inference
│
├── notebooks/
│   ├── 00_load.ipynb                    # Load raw data, compute RUL labels
│   ├── 01_eda_cleaning.ipynb            # EDA, sensor analysis
│   ├── 02_feature_engineering.ipynb     # Feature selection and engineering
│   ├── 03_baseline_model.ipynb          # Baseline model comparison
│   ├── 04_tuning_xgb_optuna_mlflow.ipynb
│   └── 05_model_lstm.ipynb
│
├── src/
│   ├── data/
│   │   ├── load.py                      # Load raw data, compute RUL labels
│   │   └── preprocess.py               # Drop sensors, normalise
│   │
│   ├── features/
│   │   ├── build_feature.py             # Feature selection + rolling/diff engineering
│   │   └── sequence_builder.py          # LSTM sequence construction
│   │
│   ├── training/
│   │   ├── train_baseline.py            # Baseline comparison
│   │   ├── tune_xgb_optuna_mlflow.py    # XGBoost tuning pipeline
│   │   ├── tune_lgbm_optuna_mlflow.py   # LightGBM tuning pipeline
│   │   └── train_lstm_optuna_mlflow.py  # LSTM tuning pipeline
│   │
│   ├── inference/
│   │   ├── schemas.py                   # Pydantic request/response models
│   │   ├── pipeline.py                  # Feature transformation at inference time
│   │   └── predict.py                   # Model registry + prediction methods
│   │
│   └── api/
│       └── app.py                       # FastAPI routes + lifespan
│
├── scripts/
│   └── export_artifacts.py              # Export scalers + feature list after training
│
├── tests/
│   ├── conftest.py
│   ├── test_inference.py                # Pipeline transforms + API endpoint tests
│   └── ...
│
├── .github/
│   └── workflows/
│       └── ci.yml                       # GitHub Actions CI
│
└── mlruns/                              # MLflow experiment tracking (local)
```

---

## Setup & Installation

**Prerequisites:** Python 3.13, [uv](https://docs.astral.sh/uv/)

```bash
# 1. Clone
git clone https://github.com/Arrofi07/nasa-rul-mle.git
cd nasa-rul-mle

# 2. Install all dependencies (runtime + dev)
uv sync --extra dev

# 3. Place raw dataset files in data/raw/
#    train_FD001.txt   test_FD001.txt   RUL_FD001.txt
#    Available from: https://www.nasa.gov/pcoe-data-set-repository
```

---

## Common Commands

### Run the full data + training pipeline

```bash
# Data pipeline
uv run python -m src.data.load
uv run python -m src.data.preprocess
uv run python -m src.features.build_feature

# Baseline comparison
uv run python -m src.training.train_baseline

# Hyperparameter tuning (each tracked in MLflow)
uv run python -m src.training.tune_xgb_optuna_mlflow
uv run python -m src.training.tune_lgbm_optuna_mlflow
uv run python -m src.training.train_lstm_optuna_mlflow

# Export scalers + feature list for inference
uv run python -m scripts.export_artifacts
```

### Start the API + dashboard locally

```bash
# Terminal 1 — API
python main.py
# → http://localhost:8000
# → http://localhost:8000/docs

# Terminal 2 — Dashboard
streamlit run app.py
# → http://localhost:8501
```

### Testing

```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ -v --cov=src --cov-report=term-missing

# Specific test file
uv run pytest tests/test_inference.py -v
```

### Docker

```bash
# Build and run
docker build -t nasa-rul-api .
docker run -p 8000:8000 nasa-rul-api

# Or with Compose
docker compose up -d

# Verify
curl http://localhost:8000/health
```

### MLflow UI

```bash
mlflow ui
# → http://localhost:5000
```

---

## CI/CD

Every push to `main` or `develop` triggers a GitHub Actions pipeline:

```
git push
    │
    ▼
GitHub Actions
    ├── Setup Python 3.13 + uv
    ├── uv sync --frozen --extra dev   (installs from uv.lock exactly)
    ├── ruff check .                   (linting)
    └── pytest -v                      (110 tests)
```

The `--frozen` flag fails CI if `pyproject.toml` and `uv.lock` are out of sync — catching the case where a dependency is added locally but the lock file isn't committed.

---

## Deployment

### Railway (API)

The repository is connected to Railway via GitHub Integration. Every push to `main` triggers an automatic deployment:

```
git push origin main
    │
    ▼
Railway detects commit
    │
    ▼
Docker build (from Dockerfile)
    │
    ▼
Container deployed + health check
    │
    ▼
https://nasa-rul-mle-production.up.railway.app
```

### Streamlit Cloud (Dashboard)

The dashboard is deployed at [nasa-rul-dashboard.streamlit.app](https://nasa-rul-dashboard.streamlit.app) and connects to the Railway API automatically. It redeploys on every push to `main`.

To run locally against the live Railway API:

```bash
API_URL=https://nasa-rul-mle-production.up.railway.app streamlit run app.py
```

---

## Roadmap

- [x] Data loading and preprocessing pipeline
- [x] Exploratory data analysis notebooks
- [x] Feature engineering (correlation selection, rolling, diff)
- [x] Baseline model comparison (Linear, Ridge, RF, XGBoost)
- [x] XGBoost tuning with Optuna + MLflow
- [x] LightGBM tuning with Optuna + MLflow
- [x] LSTM training with Optuna + MLflow
- [ ] LSTM tuning with Optuna + MLflow
- [x] Inference pipeline (scaler + feature reproducibility at serve time)
- [x] FastAPI REST API with Pydantic validation
- [x] Unit and integration tests (110 tests)
- [x] Dockerfile + Docker Compose
- [x] CI/CD with GitHub Actions
- [x] Production deployment on Railway
- [x] Streamlit dashboard on Streamlit Cloud
- [x] Add a model comparison section to the dashboard