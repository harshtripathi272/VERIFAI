"""
VERIFAI FastAPI Application

Main entrypoint for the diagnostic API.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api import router

# Import past mistakes router
try:
    from app.past_mistakes_routes import router as past_mistakes_router
    PAST_MISTAKES_API_AVAILABLE = True
except ImportError:
    PAST_MISTAKES_API_AVAILABLE = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print(f"[VERIFAI] Starting up (ENV={settings.ENV}, MOCK={settings.MOCK_MODELS})")
    if PAST_MISTAKES_API_AVAILABLE:
        print("[VERIFAI] Past Mistakes API enabled")
    yield
    print("[VERIFAI] Shutting down...")


app = FastAPI(
    title="VERIFAI API",
    description="Evidence-first, uncertainty-gated clinical diagnostic AI for chest X-rays",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Include past mistakes router if available
if PAST_MISTAKES_API_AVAILABLE:
    app.include_router(past_mistakes_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
