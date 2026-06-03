"""
Entry point for running the RUL API with Uvicorn.

Development
-----------
    python main.py
    # or with auto-reload:
    uvicorn src.api.app:app --reload --port 8000

Production
----------
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 2

Then open http://localhost:8000/docs for the interactive Swagger UI.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,       # set False in production
        log_level="info",
    )