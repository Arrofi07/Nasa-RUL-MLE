FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (Docker layer cache: only re-installs if these change)
COPY pyproject.toml uv.lock ./

# Install only runtime deps (no dev extras like pytest/ruff in production)
RUN uv sync --frozen --no-dev

# Copy source
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]