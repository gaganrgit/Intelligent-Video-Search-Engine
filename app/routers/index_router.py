"""
Indexing endpoints.

POST /api/v1/index/start   — kick off indexing in the background
GET  /api/v1/index/status/{job_id} — poll job progress
GET  /api/v1/index/jobs    — list all jobs
"""

from __future__ import annotations

import logging
import uuid
from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.schemas import IndexRequest, IndexProgress
from indexer.pipeline import IndexingPipeline, IndexJob, get_job, list_jobs, _jobs, _jobs_lock

logger = logging.getLogger("video_search.router.index")
router = APIRouter()
_pipeline = IndexingPipeline()


def _run_job(job: IndexJob, req: IndexRequest):
    _pipeline.run(
        job=job,
        sample_fps=req.sample_fps,
        use_scene_detection=req.use_scene_detection,
        force_reindex=req.force_reindex,
    )


@router.post("/start", summary="Start indexing a video or directory")
async def start_indexing(req: IndexRequest, background_tasks: BackgroundTasks):
    """
    Kick off the offline indexing pipeline asynchronously.
    Returns a `job_id` you can poll via `/status/{job_id}`.
    """
    job_id = str(uuid.uuid4())[:8]
    job = IndexJob(job_id=job_id, video_path=req.video_path)
    with _jobs_lock:
        _jobs[job_id] = job
    background_tasks.add_task(_run_job, job, req)
    logger.info("Index job %s started for '%s'", job_id, req.video_path)
    return {"job_id": job_id, "message": "Indexing started", "video_path": req.video_path}


@router.get("/status/{job_id}", response_model=IndexProgress, summary="Poll indexing progress")
async def job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    return IndexProgress(
        video=job.video_path,
        status=job.status,
        frames_indexed=job.frames_indexed,
        total_frames_estimated=job.total_frames_estimated,
        throughput_fps=round(job.throughput_fps, 2),
        elapsed_sec=round(job.elapsed_sec, 2),
        message=job.message,
    )


@router.get("/jobs", summary="List all indexing jobs")
async def all_jobs():
    jobs = list_jobs()
    return [
        {
            "job_id": j.job_id,
            "status": j.status,
            "video_path": j.video_path,
            "frames_indexed": j.frames_indexed,
            "throughput_fps": round(j.throughput_fps, 2),
            "elapsed_sec": round(j.elapsed_sec, 2),
            "message": j.message,
        }
        for j in jobs
    ]
