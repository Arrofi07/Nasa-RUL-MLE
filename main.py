"""
Entry point for running the RUL API with Uvicorn.

Development:   python main.py
Production:    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1

Open http://localhost:8000/docs for the Swagger UI.

macOS OpenMP fix
----------------
On Apple Silicon (M1/M2), XGBoost and LightGBM each bundle their own
libomp.dylib. When both are loaded in the same process two OpenMP runtimes
compete for the same thread-state memory, causing a SIGSEGV in
__kmp_suspend_initialize_thread (exactly the crash in the bug report).

The environment variables below MUST be set before any C-extension is
imported (i.e. before torch, xgboost, lightgbm, or sklearn are touched).

  KMP_DUPLICATE_LIB_OK=TRUE
    Tells the Intel OpenMP runtime to tolerate duplicate library loads
    instead of aborting. This is the primary fix for the dual-libomp crash.

  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  LGBM_NUM_THREADS=1
    Cap every BLAS/OpenMP backend to a single thread. Prevents the
    multi-runtime thread pool from ever being created, which is the
    second layer of defence.

  PYTORCH_ENABLE_MPS_FALLBACK=1
    Prevents PyTorch from using the Apple MPS (Metal) backend, which
    segfaults on some macOS + PyTorch 2.x combinations at torch.load() time.

  TOKENIZERS_PARALLELISM=false
    Suppresses a secondary fork-safety warning from HuggingFace tokenizers
    if they happen to be installed in the same venv.
"""

import multiprocessing
import os

# ── MUST be set BEFORE any C-extension / torch / xgboost / lightgbm import ──
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # primary OpenMP crash fix
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LGBM_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import uvicorn  # noqa: E402  (import after env vars are set)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,   # never enable reload — fork+PyTorch segfaults on macOS
        workers=1,      # single worker avoids any inter-process torch issues
        log_level="info",
    )