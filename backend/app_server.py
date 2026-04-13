from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import (  # noqa: E402
    API_PREFIX,
    APP_TITLE,
    APP_VERSION,
    DEFAULT_BACKEND_HOST,
    DEFAULT_BACKEND_PORT,
    DEFAULT_WEIGHTS_PATH,
    FRONTEND_DIR,
)
from backend.routes.api_v2 import router  # noqa: E402
from backend.services import prediction_service  # noqa: E402
from database.db import close_db  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("eyenet")

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=API_PREFIX)

if FRONTEND_DIR.is_dir():
    # If dist exists, use it as the root for static files
    dist_dir = FRONTEND_DIR / "dist"
    static_root = dist_dir if dist_dir.is_dir() else FRONTEND_DIR
    
    app.mount("/static", StaticFiles(directory=str(static_root)), name="static")
    
    # Also mount assets directory specifically if it exists (Vite production build)
    assets_dir = static_root / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        index_file = "index.html"
        # Try index_v2.html first if it exists, otherwise fallback to index.html
        if (static_root / "index_v2.html").exists():
            index_file = "index_v2.html"
        return FileResponse(static_root / index_file)

    @app.get("/styles.css", include_in_schema=False)
    async def serve_styles():
        # Fallback for old style naming
        css_file = "styles_v2.css" if (static_root / "styles_v2.css").exists() else "index.css"
        return FileResponse(static_root / css_file)

    @app.get("/app.js", include_in_schema=False)
    async def serve_script():
        # Fallback for old style naming
        js_file = "app_v2.js" if (static_root / "app_v2.js").exists() else "main.js"
        return FileResponse(static_root / js_file)


@app.on_event("startup")
async def startup_event():
    weights_path = os.getenv("MODEL_WEIGHTS_PATH", str(DEFAULT_WEIGHTS_PATH))
    device = "cuda" if os.getenv("USE_GPU", "0") == "1" else "cpu"
    prediction_service.load_model(weights_path=weights_path, device_str=device)
    logger.info("EyeNet backend ready at /docs")


@app.on_event("shutdown")
async def shutdown_event():
    close_db()
    logger.info("EyeNet backend stopped")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app_server:app",
        host=os.getenv("BACKEND_HOST", DEFAULT_BACKEND_HOST),
        port=int(os.getenv("BACKEND_PORT", DEFAULT_BACKEND_PORT)),
        reload=True,
    )
