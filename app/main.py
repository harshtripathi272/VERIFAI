"""
VERIFAI FastAPI Application

Main entrypoint for the diagnostic API.
"""

from contextlib import asynccontextmanager

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.api import router
from agents.validator.agent import initialize_validator_tools

# Import past mistakes router
try:
    from app.past_mistakes_routes import router as past_mistakes_router
    PAST_MISTAKES_API_AVAILABLE = True
except ImportError:
    PAST_MISTAKES_API_AVAILABLE = False

# Import SSE streaming router
from app.streaming import streaming_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print(f"[VERIFAI] Starting up (ENV={settings.ENV}, MOCK={settings.MOCK_MODELS})")
    if PAST_MISTAKES_API_AVAILABLE:
        print("[VERIFAI] Past Mistakes API enabled")
        
    # Initialize validator models asynchronously
    if not getattr(settings, "MOCK_MODELS", False):
        try:
            initialize_validator_tools()
        except Exception as e:
            print(f"[VERIFAI] Failed to initialize validator tools: {e}")
            
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

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

os.makedirs("output/heatmaps", exist_ok=True)
app.mount("/output", StaticFiles(directory="output"), name="output")

# Mobile demo (on-device AI)
_demo_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mobile_demo")
if os.path.isdir(_demo_dir):
    app.mount("/demo", StaticFiles(directory=_demo_dir, html=True), name="mobile_demo")

# Include API routes
app.include_router(router, prefix="/api/v1")
app.include_router(streaming_router, prefix="/api/v1")

# Include past mistakes router if available
if PAST_MISTAKES_API_AVAILABLE:
    app.include_router(past_mistakes_router, prefix="/api/v1")

# Root-level /metrics for Prometheus (standard convention)
from fastapi.responses import PlainTextResponse
from monitoring.metrics import metrics as _metrics, get_metrics_summary

@app.get("/metrics", response_class=PlainTextResponse)
async def root_prometheus_metrics():
    return PlainTextResponse(content=_metrics.to_prometheus_format(), media_type="text/plain; charset=utf-8")

@app.get("/metrics/summary")
async def root_metrics_summary():
    return get_metrics_summary()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
