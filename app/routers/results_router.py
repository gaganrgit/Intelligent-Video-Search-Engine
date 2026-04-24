"""
Results endpoints.

GET /api/v1/results/list        — list saved result files
GET /api/v1/results/latest      — fetch the latest results JSON
GET /api/v1/results/{filename}  — download a specific result file
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings
from utils.results_writer import latest_results_json

logger = logging.getLogger("video_search.router.results")
router = APIRouter()


@router.get("/list", summary="List saved result files")
async def list_results():
    results_dir = Path(settings.RESULTS_DIR)
    files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"files": [f.name for f in files]}


@router.get("/latest", summary="Get the most recent query results")
async def latest_results():
    path = latest_results_json()
    if path is None:
        raise HTTPException(404, "No results found yet. Run a query first.")
    import json
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/{filename}", summary="Download a specific results file")
async def download_result(filename: str):
    path = Path(settings.RESULTS_DIR) / filename
    if not path.exists():
        raise HTTPException(404, f"File '{filename}' not found.")
    return FileResponse(str(path), media_type="application/json", filename=filename)
