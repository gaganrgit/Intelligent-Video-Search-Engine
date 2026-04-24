"""
Query endpoints.

POST /api/v1/query/search  — natural language search
GET  /api/v1/query/html    — same search but returns HTML results page
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse

from app.schemas import QueryRequest, QueryResponse
from app.config import settings
from indexer.query_engine import QueryEngine
from utils.html_renderer import render_html_results

logger = logging.getLogger("video_search.router.query")
router = APIRouter()
_engine = QueryEngine()


@router.post("/search", response_model=QueryResponse, summary="Natural language video search")
async def search(req: QueryRequest):
    """
    Search the indexed video archive using a natural language query.

    **Temporal filter examples:**
    - `time_start`: `"18:00:00"`, `time_end`: `"20:00:00"`

    **Query examples:**
    - `"person near the entrance carrying a bag"`
    - `"red vehicle parked in zone 3"`
    - `"anything unusual in the corridor"`
    """
    try:
        response = _engine.search(
            query=req.query,
            top_k=req.top_k,
            time_start=req.time_start,
            time_end=req.time_end,
            video_filter=req.video_filter,
            rerank=req.rerank,
            decompose_query=req.decompose_query,
        )
        return response
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Search error: %s", e)
        raise HTTPException(500, f"Internal error: {e}")


@router.get("/html", response_class=HTMLResponse, summary="Search and get HTML results page")
async def search_html(
    q: str = Query(..., description="Natural language query"),
    top_k: int = Query(10, ge=1, le=100),
    time_start: str = Query(None),
    time_end: str = Query(None),
):
    """
    Same as /search but returns a rendered HTML page with thumbnails.
    Open directly in a browser.
    """
    try:
        req = QueryRequest(
            query=q,
            top_k=top_k,
            time_start=time_start,
            time_end=time_end,
        )
        response = _engine.search(
            query=req.query,
            top_k=req.top_k,
            time_start=req.time_start,
            time_end=req.time_end,
            rerank=req.rerank,
            decompose_query=req.decompose_query,
        )
        html_path = render_html_results(q, response.results, settings.RESULTS_DIR)
        return HTMLResponse(Path(html_path).read_text(encoding="utf-8"))
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("HTML search error: %s", e)
        raise HTTPException(500, str(e))
