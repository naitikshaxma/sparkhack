from __future__ import annotations

from fastapi import FastAPI

from backend.src.routes.api import router as api_router

app = FastAPI(title="Voice OS Backend", version="hardcode-30-only")
app.include_router(api_router)
