"""FastAPI app initialization (minimal)."""

from fastapi import FastAPI
from .routers import ingestion_router, query_router, explain_router
from config.settings import Settings  # hypothetical settings loader

def create_app() -> FastAPI:
    app = FastAPI(title="my_kg_app API")
    # you can add routers, middleware, dependencies here
    return app

app = create_app()

app.include_router(ingestion_router.router)
app.include_router(query_router.router)
app.include_router(explain_router.router)


@app.get("/ping")
async def ping():
    return {"status": "ok"}
