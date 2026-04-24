"""
FAISS Vector Store
==================

Choice rationale
----------------
FAISS (Facebook AI Similarity Search) was chosen over alternatives because:
  • Pure-Python install (no server process required) — zero-ops.
  • IVFFlat with L2/IP delivers sub-millisecond ANN search for millions of
    vectors on a single machine.
  • Quantisation (PQ, SQ8) available if index grows beyond RAM.
  • Flat index used as fallback when corpus < 4 × nlist (FAISS requirement).

Index layout (persisted to disk)
---------------------------------
  index_store/
    <video_stem>.index   ← FAISS binary index
    <video_stem>.meta.json ← per-frame metadata (timestamp, path, etc.)
  index_store/combined.index / combined.meta.json  ← merged index
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from app.config import settings

logger = logging.getLogger("video_search.vector_store")


@dataclass
class FrameMeta:
    video: str
    frame_index: int
    timestamp_sec: float
    timestamp_hms: str
    frame_path: str      # relative path to saved thumbnail
    is_scene_change: bool = False


class VideoVectorStore:
    """
    Manages one FAISS index per video plus a combined merged index
    for cross-video search.

    Thread-safety: read operations are safe; writes acquire no lock because
    indexing is a single-threaded offline step.
    """

    def __init__(self, index_dir: str = settings.INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.dim = settings.EMBEDDING_DIM

        # In-memory state (loaded lazily)
        self._index: Optional[faiss.Index] = None
        self._meta: List[FrameMeta] = []

    # ── Build / persist ───────────────────────────────────────────────────────

    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build a FAISS IVFFlat index (falls back to Flat for small corpora).
        embeddings: float32 array shape (N, dim), L2-normalised.
        """
        n = embeddings.shape[0]
        nlist = min(settings.INDEX_NLIST, n // 4) if n >= 40 else 1

        if nlist <= 1:
            logger.info("Small corpus (%d frames) → FlatIP index", n)
            index = faiss.IndexFlatIP(self.dim)
        else:
            logger.info("Building IVFFlat index: n=%d, nlist=%d", n, nlist)
            quantiser = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantiser, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)

        index.add(embeddings)
        return index

    def save(self, video_stem: str, index: faiss.Index, meta: List[FrameMeta]):
        idx_path = self.index_dir / f"{video_stem}.index"
        meta_path = self.index_dir / f"{video_stem}.meta.json"
        faiss.write_index(index, str(idx_path))
        with open(meta_path, "w") as f:
            json.dump([asdict(m) for m in meta], f, indent=2)
        logger.info("Saved index '%s' (%d vectors)", video_stem, index.ntotal)

    def load(self, video_stem: str) -> Tuple[faiss.Index, List[FrameMeta]]:
        idx_path = self.index_dir / f"{video_stem}.index"
        meta_path = self.index_dir / f"{video_stem}.meta.json"
        index = faiss.read_index(str(idx_path))
        if hasattr(index, "nprobe"):
            index.nprobe = settings.INDEX_NPROBE
        with open(meta_path) as f:
            meta = [FrameMeta(**m) for m in json.load(f)]
        return index, meta

    def exists(self, video_stem: str) -> bool:
        return (self.index_dir / f"{video_stem}.index").exists()

    # ── Merged index (cross-video search) ─────────────────────────────────────

    def merge_all(self):
        """Merge all per-video indices into combined.index for global search."""
        all_embs: List[np.ndarray] = []
        all_meta: List[FrameMeta] = []

        for idx_file in sorted(self.index_dir.glob("*.index")):
            if idx_file.stem == "combined":
                continue
            stem = idx_file.stem
            try:
                index, meta = self.load(stem)
                # Reconstruct embeddings from the flat portion
                vecs = np.zeros((index.ntotal, self.dim), dtype=np.float32)
                # For IVF we need to iterate; easiest is to read from stored meta
                # and re-embed — instead we store a companion .npy file
                npy_path = self.index_dir / f"{stem}.npy"
                if npy_path.exists():
                    vecs = np.load(str(npy_path))
                    all_embs.append(vecs)
                    all_meta.extend(meta)
            except Exception as e:
                logger.warning("Skipping index '%s': %s", stem, e)

        if not all_embs:
            logger.warning("No embeddings found to merge.")
            return

        combined = np.vstack(all_embs)
        merged_index = self.build_index(combined)
        self.save("combined", merged_index, all_meta)
        # Cache in memory
        self._index = merged_index
        self._meta = all_meta
        logger.info("Merged index: %d total vectors", merged_index.ntotal)

    def save_embeddings(self, video_stem: str, embeddings: np.ndarray):
        """Persist raw embeddings for later merging."""
        npy_path = self.index_dir / f"{video_stem}.npy"
        np.save(str(npy_path), embeddings.astype(np.float32))

    # ── Search ────────────────────────────────────────────────────────────────

    def ensure_loaded(self):
        if self._index is None:
            combined_idx = self.index_dir / "combined.index"
            if combined_idx.exists():
                self._index, self._meta = self.load("combined")
            else:
                # Fall back: load the first available index
                for f in self.index_dir.glob("*.index"):
                    if f.stem != "combined":
                        self._index, self._meta = self.load(f.stem)
                        break
            if self._index is None:
                raise RuntimeError("No index found. Run /api/v1/index/start first.")

    def search(
        self,
        query_emb: np.ndarray,
        top_k: int,
        time_start_sec: Optional[float] = None,
        time_end_sec: Optional[float] = None,
        video_filter: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        """
        ANN search. Returns list of (meta_index, score) sorted by score desc.
        Temporal and video filters applied post-retrieval on expanded top-K.
        """
        self.ensure_loaded()
        t0 = time.perf_counter()

        # Over-fetch to allow for temporal/video filtering
        fetch_k = min(top_k * 10, self._index.ntotal)
        scores, indices = self._index.search(query_emb.astype(np.float32), fetch_k)

        results: List[Tuple[int, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._meta[idx]

            if video_filter and video_filter.lower() not in meta.video.lower():
                continue
            if time_start_sec is not None and meta.timestamp_sec < time_start_sec:
                continue
            if time_end_sec is not None and meta.timestamp_sec > time_end_sec:
                continue

            results.append((int(idx), float(score)))
            if len(results) >= top_k:
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("ANN search: %d results in %.1f ms", len(results), elapsed_ms)
        return results

    def get_meta(self, idx: int) -> FrameMeta:
        meta = self._meta[idx]
        # Normalise backslashes on Windows
        meta.frame_path = meta.frame_path.replace("\\", "/")
        return meta

    @property
    def total_vectors(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal
