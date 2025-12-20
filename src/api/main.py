"""FastAPI app initialization (minimal)."""

from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Dict
from .routers import ingestion_router, query_router, explain_router
from config.settings import Settings  # hypothetical settings loader

def create_app() -> FastAPI:
    app = FastAPI(title="my_kg_app API")
    # attach routers and shared settings
    settings = Settings()
    app.state.settings = settings
    # include routers defined in src/api/routers
    app.include_router(ingestion_router.router)
    app.include_router(query_router.router)
    app.include_router(explain_router.router)
    return app

app = create_app()

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the KG-backed QA service"}

@app.post("/ingest/")
async def ingest(file: UploadFile = File(...)):
    # Example ingest endpoint
    try:
        contents = await file.read()
        # TODO: save to temp file or pass bytes to ingestion
        return {"status": "ingested", "filename": file.filename, "size": len(contents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
def query_endpoint(payload: Dict):
    """
    Example payload:
    {
        "question": "...",
        "session_id": "...",  # optional
    }
    """
    question = payload.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Missing question in payload")
    # TODO: perform retrieval + LLM generation + return response
    return {"answer": "This is a placeholder answer"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
