"""
Indexing Pipeline
=================
Orchestrates: video discovery → frame sampling → batched CLIP embedding
→ thumbnail saving → FAISS index build → merge.

Designed to be run as a background task via FastAPI BackgroundTasks so
the /index/start endpoint returns immediately with a job ID.
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from app.config import settings
from indexer.frame_sampler import FrameSampler, SampledFrame
from indexer.vector_store import FrameMeta, VideoVectorStore
from models.clip_embedder import get_embedder
from utils.time_utils import sec_to_hms
from utils.thumbnail import save_thumbnail

logger = logging.getLogger("video_search.pipeline")

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".ts", ".m4v", ".wmv", ".mpg", ".mpeg", ".3gp", ""}


@dataclass
class IndexJob:
    job_id: str
    video_path: str
    status: str = "pending"      # pending | running | done | error
    frames_indexed: int = 0
    total_frames_estimated: int = 0
    throughput_fps: float = 0.0
    elapsed_sec: float = 0.0
    message: str = ""
    videos_processed: List[str] = field(default_factory=list)


# Global job registry (in-memory; survives the process lifetime)
_jobs: Dict[str, IndexJob] = {}
_jobs_lock = threading.Lock()


def get_job(job_id: str) -> Optional[IndexJob]:
    return _jobs.get(job_id)


def list_jobs() -> List[IndexJob]:
    return list(_jobs.values())


class IndexingPipeline:
    """
    End-to-end indexing pipeline.

    Usage (background task)::
        pipeline = IndexingPipeline()
        pipeline.run(job)   # blocks until done; call via BackgroundTasks
    """

    def __init__(self):
        self.sampler = FrameSampler()
        self.store = VideoVectorStore()

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self, job: IndexJob, sample_fps: float, use_scene_detection: bool, force_reindex: bool):
        with _jobs_lock:
            job.status = "running"

        self.sampler.sample_fps = sample_fps
        self.sampler.use_scene_detection = use_scene_detection

        t_start = time.perf_counter()
        try:
            video_files = self._discover_videos(job.video_path)
            if not video_files:
                raise ValueError(f"No video files found at: {job.video_path}")

            job.total_frames_estimated = sum(
                self.sampler.estimate_frame_count(v) for v in video_files
            )
            logger.info("Indexing %d video(s), ~%d frames total", len(video_files), job.total_frames_estimated)

            embedder = get_embedder()

            for video_path in video_files:
                stem = Path(video_path).stem
                if not force_reindex and self.store.exists(stem):
                    logger.info("Skipping '%s' (already indexed)", stem)
                    job.videos_processed.append(str(video_path))
                    continue

                self._index_video(video_path, stem, job, embedder)
                job.videos_processed.append(str(video_path))

            # Merge all per-video indices into combined
            self.store.merge_all()

            job.elapsed_sec = time.perf_counter() - t_start
            job.throughput_fps = job.frames_indexed / max(job.elapsed_sec, 0.001)
            job.status = "done"
            job.message = f"Indexed {job.frames_indexed} frames from {len(job.videos_processed)} video(s)"
            logger.info("Indexing complete: %s", job.message)

        except Exception as exc:
            job.status = "error"
            job.message = str(exc)
            job.elapsed_sec = time.perf_counter() - t_start
            logger.exception("Indexing failed: %s", exc)

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _discover_videos(path: str) -> List[Path]:
        p = Path(path)
        if p.is_file():
            if p.suffix.lower() in VIDEO_EXTENSIONS or p.suffix == "":
                return [p]
        if p.is_dir():
            results = []
            for f in p.rglob("*"):
                if f.is_file() and (f.suffix.lower() in VIDEO_EXTENSIONS or f.suffix == ""):
                    results.append(f)
            return sorted(results)
        return []

    def _index_video(self, video_path: Path, stem: str, job: IndexJob, embedder):
        logger.info("Indexing '%s' …", video_path.name)

        frame_buffer: List[SampledFrame] = []
        emb_buffer: List[np.ndarray] = []
        meta_list: List[FrameMeta] = []
        batch_images: List[Image.Image] = []
        batch_frames: List[SampledFrame] = []

        def flush_batch():
            if not batch_images:
                return
            embs = embedder.embed_images(batch_images)
            for emb, sf in zip(embs, batch_frames):
                emb_buffer.append(emb)
                # Save thumbnail
                thumb_rel = save_thumbnail(sf.image, stem, sf.frame_index)
                meta_list.append(FrameMeta(
                    video=str(video_path.name),
                    frame_index=sf.frame_index,
                    timestamp_sec=sf.timestamp_sec,
                    timestamp_hms=sec_to_hms(sf.timestamp_sec),
                    frame_path=thumb_rel,
                    is_scene_change=sf.is_scene_change,
                ))
            job.frames_indexed += len(batch_images)
            batch_images.clear()
            batch_frames.clear()

        for sf in self.sampler.sample(video_path):
            batch_images.append(sf.image)
            batch_frames.append(sf)
            if len(batch_images) >= embedder.batch_size:
                flush_batch()

        flush_batch()  # tail batch

        if not emb_buffer:
            logger.warning("No frames extracted from '%s'", video_path.name)
            return

        embeddings = np.vstack(emb_buffer).astype(np.float32)
        index = self.store.build_index(embeddings)
        self.store.save(stem, index, meta_list)
        self.store.save_embeddings(stem, embeddings)
        logger.info("Done: '%s' → %d frames", video_path.name, len(meta_list))
