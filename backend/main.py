"""
EyeNet FastAPI Application Entry Point
"""

import logging
import os
import sys

# Make sure project root is on the path when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.routes.predict import router
from backend.services.inference import load_model
from database.db import close_db

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("eyenet")

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EyeNet – Retinal Disease Detection & Analysis System",
    description=(
        "AI-powered API for classifying retinal fundus images into:\n"
        "Diabetic Retinopathy, Glaucoma, Cataract, or Normal.\n\n"
        "Provides Grad-CAM explainability heatmaps and MongoDB-backed history."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow the React frontend (or any origin during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

# ── Serve frontend at http://localhost:8000/ ──────────────────────────────────
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(os.path.join(_frontend_dir, "index.html"))

    @app.get("/styles.css", include_in_schema=False)
    async def serve_css():
        return FileResponse(os.path.join(_frontend_dir, "styles.css"))

    @app.get("/app.js", include_in_schema=False)
    async def serve_js():
        return FileResponse(os.path.join(_frontend_dir, "app.js"))


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle events
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    weights = os.getenv("MODEL_WEIGHTS_PATH", "./weights/eyenet_ensemble.pth")
    device  = "cuda" if os.getenv("USE_GPU", "0") == "1" else "cpu"
    load_model(weights_path=weights, device_str=device)
    logger.info("EyeNet backend started – docs at /docs")


@app.on_event("shutdown")
async def shutdown_event():
    close_db()
    logger.info("EyeNet backend shut down.")


# ─────────────────────────────────────────────────────────────────────────────
# Dev runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("BACKEND_HOST", "0.0.0.0"),
        port=int(os.getenv("BACKEND_PORT", 8000)),
        reload=True,
    )
