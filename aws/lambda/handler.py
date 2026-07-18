"""
aws/lambda/handler.py
======================
AWS Lambda entry point.

Architecture
------------
Lambda receives HTTP events from API Gateway (or a Function URL) in a
proprietary JSON format. FastAPI speaks ASGI, not that format. Mangum is an
adapter that translates between the two — it converts the Lambda event dict
into an ASGI-compatible scope and feeds it to the FastAPI app, then converts
the FastAPI response back into the format Lambda/API Gateway expects.

The result: the EXACT same FastAPI app code (app.py, routes, Pydantic schemas)
runs on both Railway (real HTTP server via uvicorn) and Lambda (event-driven
via Mangum) without any changes to the application layer.

Cold start optimisation
-----------------------
The ModelRegistry is loaded OUTSIDE the handler function at module level.
Lambda freezes the execution environment between invocations and thaws it
for the next request. By loading models at module level they survive thaws
and only pay the loading cost once per container lifetime (~2–4 seconds),
not once per request.

If we loaded inside the handler: every request → cold start → 2-4s latency.
If we load at module level:      first request → cold start → 2-4s latency
                                 subsequent   → warm start → <100ms latency

Environment variables
---------------------
Set these in the Lambda console or via the deploy script:

  KMP_DUPLICATE_LIB_OK=TRUE    prevent OpenMP conflict (same as Railway)
  OMP_NUM_THREADS=1            single-threaded BLAS (Lambda is single-vCPU)
  LGBM_NUM_THREADS=1           single-threaded LightGBM
  LOG_LEVEL=info               uvicorn / FastAPI log level
"""

import os

# Set env vars BEFORE any library imports — same reasoning as main.py.
# On Lambda there is no Apple Silicon so the MPS/OpenMP crash can't happen,
# but KMP_DUPLICATE_LIB_OK and thread count vars still matter because Lambda
# shares a CPU with other functions and multi-threading wastes its time slice.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK",    "TRUE")
os.environ.setdefault("OMP_NUM_THREADS",          "1")
os.environ.setdefault("MKL_NUM_THREADS",          "1")
os.environ.setdefault("LGBM_NUM_THREADS",         "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM",   "false")

import traceback  # noqa: E402

from mangum import Mangum  # noqa: E402  (installed in Lambda image only)

# Import the same FastAPI app used on Railway.
# app.py uses a lifespan handler that loads models at startup.
# On Lambda, the lifespan runs when this module is first imported (cold start).
from src.api.app import app  # noqa: E402

# ---------------------------------------------------------------------------
# Mangum adapter
# ---------------------------------------------------------------------------
# api_gateway_base_path: if you mount the Lambda under a stage prefix like
# /prod/v1 in API Gateway, set this to "/prod/v1" so FastAPI routes resolve
# correctly. Leave as "/" for Lambda Function URLs (no stage prefix).
#
# lifespan="auto" tells Mangum to honour FastAPI's lifespan events (startup/
# shutdown). This is what triggers ModelRegistry.from_paths() on cold start.

handler = Mangum(app, lifespan="auto", api_gateway_base_path="/")


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------
# AWS Lambda calls this function for every incoming request.
# `event`   = the raw API Gateway / Function URL event dict
# `context` = Lambda runtime context (timeout, memory, request ID, etc.)
#
# Mangum translates event → ASGI → FastAPI → response → Lambda response dict.

def lambda_handler(event: dict, context) -> dict:
    """
    AWS Lambda handler — translates API Gateway events to FastAPI responses.

    This is the function name you enter in the Lambda console:
        Handler: handler.lambda_handler
    """
    # Log the incoming request for CloudWatch debugging.
    # In production you might want to strip this or log at DEBUG level.
    path   = event.get("rawPath", event.get("path", "unknown"))
    method = event.get("requestContext", {}).get("http", {}).get("method", "unknown")
    print(f"→ {method} {path}")

    try:
        response = handler(event, context)
        print(f"← {response.get('statusCode', 'no-status')}")
        return response

    except Exception as exc:
        # Catch any unhandled exception at the Lambda boundary so it appears
        # in CloudWatch with a full traceback rather than a generic timeout.
        print(f"❌ Unhandled exception in lambda_handler:\n{traceback.format_exc()}")
        return {
            "statusCode": 500,
            "headers":    {"Content-Type": "application/json"},
            "body":       f'{{"detail": "Internal server error: {str(exc)}"}}',
        }