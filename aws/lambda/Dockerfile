# aws/lambda/Dockerfile
# ======================
# Builds a Docker image for AWS Lambda that serves the NASA RUL API.
#
# Key differences from the Railway Dockerfile
# -------------------------------------------
# 1. Base image: public.ecr.aws/lambda/python:3.12
#    AWS provides a Lambda-compatible Python base image. It includes the
#    Lambda Runtime Interface Client (RIC) — the binary that handles the
#    event loop between Lambda's runtime and your handler function.
#    Without this, Lambda doesn't know how to call your code.
#
# 2. No torch — onnxruntime instead
#    torch==2.11.0 unpacked is ~800 MB. Lambda's limit is 250 MB (zip) /
#    ~550 MB (unzipped container). onnxruntime==1.20.0 is ~50 MB.
#    We swap torch for onnxruntime and load best_lstm.onnx instead of
#    best_lstm.pt — identical predictions, 16x smaller package.
#
# 3. WORKDIR is /var/task
#    Lambda runs code from /var/task by default. The Lambda base image sets
#    this automatically. All COPY instructions use this as the root.
#
# 4. CMD points to the handler function
#    Format: "module.function" where module is the Python file (without .py)
#    and function is the callable Lambda invokes.
#    Railway uses uvicorn to run a long-lived HTTP server.
#    Lambda uses the handler function for event-driven invocations.

FROM public.ecr.aws/lambda/python:3.12

WORKDIR /var/task

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
# build-essential and gcc are needed to compile any C extensions during pip
# install. Lambda base images are minimal — we add only what's needed.
RUN dnf install -y gcc gcc-c++ && dnf clean all

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
# Install only the packages needed for INFERENCE — no training tools.
# Excluded from Railway: torch, mlflow, optuna, jupyter, streamlit
# New for Lambda:        onnxruntime, mangum
#
# mangum: ASGI adapter that translates API Gateway events → FastAPI
# onnxruntime: runs .onnx models without PyTorch
#
# We pin every version for reproducible Lambda images.
COPY requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
# Copy shared inference code (pipeline.py, schemas.py, app.py)
COPY src/ src/

# Copy the Lambda-specific predict.py — it overwrites the Railway version
# inside the image so app.py imports the ONNX-based ModelRegistry.
COPY aws/lambda/src/inference/predict.py src/inference/predict.py

# Copy the Lambda handler entry point
COPY aws/lambda/handler.py handler.py

# ---------------------------------------------------------------------------
# Model artifacts
# ---------------------------------------------------------------------------
# Bundling models inside the Docker image is the simplest approach.
# The image is stored in ECR and Lambda pulls it at cold start.
# Alternative: store models in S3 and download at cold start — slower
# cold starts but smaller ECR image. For this project, bundling is fine
# since the total model size (~30 MB) is well within Lambda limits.
#
# best_lstm.onnx  — exported by scripts/export_onnx.py (NOT best_lstm.pt)
# best_xgb.pkl    — standard joblib pickle
# best_lgbm.pkl   — standard joblib pickle
# *.pkl + *.txt   — scalers and feature column list for the pipeline
COPY models/best_xgb.pkl            models/best_xgb.pkl
COPY models/best_lgbm.pkl           models/best_lgbm.pkl
COPY models/best_lstm.onnx          models/best_lstm.onnx
COPY models/lstm_config.json        models/lstm_config.json
COPY models/preprocess_scaler.pkl   models/preprocess_scaler.pkl
COPY models/feature_scaler.pkl      models/feature_scaler.pkl
COPY models/feature_cols.txt        models/feature_cols.txt

# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------
# Format: "python_module_name.function_name"
# handler.py → handler module
# lambda_handler → the function Lambda calls for each request
CMD ["handler.lambda_handler"]