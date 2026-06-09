"""
Entry point for running the RUL API with Uvicorn.

Development:   python main.py
Production:    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1

Open http://localhost:8000/docs for the Swagger UI.

macOS fix
---------
Two environment variables are set before anything imports torch:

  PYTORCH_ENABLE_MPS_FALLBACK=1
    Prevents PyTorch from trying to use the Apple MPS (Metal) backend,
    which segfaults on some macOS + PyTorch version combinations at
    model-load time.

  OMP_NUM_THREADS=1
    OpenMP spins up threads during torch.load(). On macOS those threads
    conflict with uvicorn's event loop and cause a segfault. Limiting to
    1 thread disables the conflicting parallelism.

  TOKENIZERS_PARALLELISM=false
    Suppresses a secondary fork-safety warning from HuggingFace tokenizers
    if they happen to be installed in the same venv.
"""

import os
import multiprocessing
import uvicorn

# Must be set BEFORE torch is imported anywhere
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    uvicorn.run(...)

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # never enable reload — fork+PyTorch segfaults on macOS
        workers=1,  # single worker avoids any inter-process torch issues
        log_level="info",
    )
