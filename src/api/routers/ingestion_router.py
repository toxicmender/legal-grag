from fastapi import APIRouter

router = APIRouter(prefix="/ingest")

@router.get("/status")
async def status():
    return {"ingest": "ready"}
