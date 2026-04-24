"""
Pydantic request / response schemas.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Indexing ──────────────────────────────────────────────────────────────────

class IndexRequest(BaseModel):
    video_path: str = Field(..., description="Absolute or relative path to a video file or directory.")
    sample_fps: float = Field(1.0, ge=0.1, le=30.0, description="Frames per second to sample.")
    use_scene_detection: bool = Field(True, description="Augment uniform sampling with scene-change detection.")
    force_reindex: bool = Field(False, description="Re-index even if an existing index is found.")


class IndexProgress(BaseModel):
    video: str
    status: str                # "running" | "done" | "error"
    frames_indexed: int
    total_frames_estimated: int
    throughput_fps: float
    elapsed_sec: float
    message: str = ""


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query.")
    top_k: int = Field(10, ge=1, le=100)
    time_start: Optional[str] = Field(None, description="Temporal filter start HH:MM:SS.")
    time_end: Optional[str] = Field(None, description="Temporal filter end HH:MM:SS.")
    video_filter: Optional[str] = Field(None, description="Restrict search to a specific video filename.")
    rerank: bool = Field(True, description="Apply CLIP re-ranking on top-K candidates.")
    decompose_query: bool = Field(True, description="Decompose complex queries into sub-queries.")


class FrameResult(BaseModel):
    rank: int
    video: str
    timestamp_sec: float
    timestamp_hms: str
    score: float
    thumbnail_url: str
    frame_path: str
    query: str
    sub_query: Optional[str] = None   # which sub-query produced this hit


class QueryResponse(BaseModel):
    query: str
    sub_queries: List[str] = []
    results: List[FrameResult]
    latency_ms: float
    reranked: bool
    total_frames_searched: int


# ── Results ───────────────────────────────────────────────────────────────────

class ResultsListResponse(BaseModel):
    files: List[str]
