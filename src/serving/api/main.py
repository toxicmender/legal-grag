"""Minimal FastAPI entrypoint under serving.api."""
from fastapi import FastAPI
from .routes import router

app = FastAPI(title="my_kg_app (serving)")
app.include_router(router)

@app.get("/health")
async def health():
    return {"status": "ok"}
