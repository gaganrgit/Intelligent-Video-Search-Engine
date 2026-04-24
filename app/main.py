"""
Intelligent Video Search Engine — FastAPI Application
Variphi Take-Home Project
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from app.routers import index_router, query_router, results_router
from app.config import settings

# ── Logging ──────────────────────────────────────────────────────────────────
import sys

_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_formatter)
_file_handler = logging.FileHandler(settings.LOG_FILE, mode="a", encoding="utf-8")
_file_handler.setFormatter(_formatter)

logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])
logger = logging.getLogger("video_search")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Video Search Engine starting up")
    Path(settings.THUMBNAIL_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.INDEX_DIR).mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Video Search Engine shutting down")


app = FastAPI(
    title="Intelligent Video Search Engine",
    description=(
        "Search through video archives using natural language queries. "
        "Returns ranked, timestamped results with frame previews."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ──────────────────────────────────────────────────────────────
app.mount("/thumbnails", StaticFiles(directory=settings.THUMBNAIL_DIR, html=False), name="thumbnails")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(index_router.router, prefix="/api/v1/index",  tags=["Indexing"])
app.include_router(query_router.router, prefix="/api/v1/query",  tags=["Query"])
app.include_router(results_router.router, prefix="/api/v1/results", tags=["Results"])


@app.get("/", response_class=HTMLResponse, summary="Web UI")
async def root():
    """Serve the interactive web UI."""
    ui_path = Path("static/index.html")
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Video Search Engine</h1><p>Visit <a href='/docs'>/docs</a> for the API.</p>")


@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "timestamp": time.time()}
