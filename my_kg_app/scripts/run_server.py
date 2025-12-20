"""Launch the `serving.api` FastAPI app using uvicorn (Windows-friendly).

Usage:
    python scripts/run_server.py
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.serving.api.main:app", host="0.0.0.0", port=8000, reload=True)
