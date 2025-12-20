#!/usr/bin/env bash
# Launch the serving.api FastAPI app with uvicorn
python -m uvicorn src.serving.api.main:app --reload --host 0.0.0.0 --port 8000
