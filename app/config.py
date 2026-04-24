"""
Central configuration — all tuneable knobs in one place.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Paths ─────────────────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    THUMBNAIL_DIR: str = "static/thumbnails"
    RESULTS_DIR: str = "static/results"
    INDEX_DIR: str = "index_store"
    LOG_FILE: str = "video_search.log"

    # ── CLIP model ────────────────────────────────────────────────────────────
    CLIP_MODEL: str = "openai/clip-vit-base-patch32"   # swap to ViT-L for higher accuracy
    DEVICE: str = ""                                    # auto-detected at runtime

    # ── Frame sampling ────────────────────────────────────────────────────────
    SAMPLE_FPS: float = 1.0          # baseline: 1 frame/sec
    SCENE_CHANGE_THRESHOLD: float = 30.0   # pixel-diff threshold for scene detection
    USE_SCENE_DETECTION: bool = True  # scene-change + uniform hybrid

    # ── Embedding & indexing ──────────────────────────────────────────────────
    BATCH_SIZE: int = 32             # GPU batch; halved automatically on CPU
    EMBEDDING_DIM: int = 512         # CLIP ViT-B/32
    INDEX_NLIST: int = 100           # FAISS IVFFlat nlist clusters
    INDEX_NPROBE: int = 10           # search probes (speed/recall tradeoff)
    USE_FP16: bool = True            # half-precision inference

    # ── Query ─────────────────────────────────────────────────────────────────
    TOP_K: int = 10
    RERANK_TOP_K: int = 50           # candidates fed to re-ranker
    TEMPORAL_WINDOW_SEC: int = 5     # seconds of context around a hit

    # ── API ───────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
