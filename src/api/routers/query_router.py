from fastapi import APIRouter

router = APIRouter(prefix="/query")

@router.get("/health")
async def health():
    return {"query": "ok"}
