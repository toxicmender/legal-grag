from fastapi import APIRouter

router = APIRouter(prefix="/explain")

@router.get("/ping")
async def ping():
    return {"explain": "ok"}
