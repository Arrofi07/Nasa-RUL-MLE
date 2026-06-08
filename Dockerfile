FROM python:3.12-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# copy dependency definition first (cache layer)
COPY pyproject.toml .

# install package manager
RUN pip install --no-cache-dir --upgrade pip

# install runtime deps
RUN pip install --no-cache-dir \
    numpy==2.4.4 \
    pandas==2.3.3 \
    scikit-learn==1.8.0 \
    xgboost==3.2.0 \
    torch==2.11.0 \
    mlflow==3.11.1 \
    scipy==1.17.1 \
    pyarrow==23.0.1 \
    cloudpickle==3.1.2 \
    psutil==7.2.2 \
    tqdm==4.67.3 \
    fastapi==0.115.12 \
    uvicorn[standard]==0.34.3 \
    pydantic==2.11.4 \
    joblib

# copy source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]