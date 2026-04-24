"""
Query Engine
============

Three-stage retrieval:

1. Query decomposition
   Complex queries like "person near entrance carrying a bag after 6 PM" are
   split into atomic sub-queries using simple heuristic rules (extendable with
   an LLM). Sub-query results are merged and de-duplicated by proximity.

2. ANN retrieval (FAISS)
   Each sub-query is encoded by CLIP, and an approximate nearest-neighbour
   search returns `rerank_top_k` candidates per sub-query.

3. CLIP re-ranking
   The original query embedding is dot-producted against each candidate's
   stored embedding (loaded from .npy). Top-K are returned after re-sorting.

Temporal filtering is applied at stage 2 (inside FAISS search).
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from app.config import settings
from app.schemas import FrameResult, QueryResponse
from indexer.vector_store import FrameMeta, VideoVectorStore
from models.clip_embedder import get_embedder
from utils.time_utils import hms_to_sec, sec_to_hms
from utils.results_writer import write_results

logger = logging.getLogger("video_search.query")


class QueryEngine:
    def __init__(self):
        self.store = VideoVectorStore()
        self._all_embeddings: Optional[np.ndarray] = None  # lazy-loaded for re-ranking

    # ── Public ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = settings.TOP_K,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        video_filter: Optional[str] = None,
        rerank: bool = True,
        decompose_query: bool = True,
    ) -> QueryResponse:
        t0 = time.perf_counter()

        # ── 1. Query decomposition ────────────────────────────────────────────
        if decompose_query:
            sub_queries = self._decompose(query)
        else:
            sub_queries = [query]
        logger.info("Sub-queries: %s", sub_queries)

        # ── 2. Temporal filter conversion ─────────────────────────────────────
        ts_start = hms_to_sec(time_start) if time_start else None
        ts_end = hms_to_sec(time_end) if time_end else None

        # ── 3. ANN retrieval for each sub-query ───────────────────────────────
        embedder = get_embedder()
        seen_indices: set = set()
        raw_hits: List[Tuple[int, float, str]] = []   # (meta_idx, score, sub_query)

        rerank_k = settings.RERANK_TOP_K if rerank else top_k

        for sq in sub_queries:
            q_emb = embedder.embed_query(sq)
            hits = self.store.search(q_emb, rerank_k, ts_start, ts_end, video_filter)
            for idx, score in hits:
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    raw_hits.append((idx, score, sq))

        total_searched = self.store.total_vectors

        # ── 4. Re-ranking ─────────────────────────────────────────────────────
        if rerank and len(raw_hits) > top_k:
            raw_hits = self._rerank(query, raw_hits, embedder)

        raw_hits = raw_hits[:top_k]

        # ── 5. Build response ─────────────────────────────────────────────────
        latency_ms = (time.perf_counter() - t0) * 1000
        results: List[FrameResult] = []

        for rank, (idx, score, sq) in enumerate(raw_hits, start=1):
            meta: FrameMeta = self.store.get_meta(idx)
            thumb_url = f"/thumbnails/{meta.frame_path.replace(chr(92), '/')}"
            results.append(FrameResult(
                rank=rank,
                video=meta.video,
                timestamp_sec=meta.timestamp_sec,
                timestamp_hms=meta.timestamp_hms,
                score=round(float(score), 4),
                thumbnail_url=thumb_url,
                frame_path=meta.frame_path,
                query=query,
                sub_query=sq if sq != query else None,
            ))

        # ── 6. Persist results ────────────────────────────────────────────────
        write_results(query, results)

        logger.info(
            "Query '%s': %d results, %.1f ms (reranked=%s)",
            query, len(results), latency_ms, rerank,
        )

        return QueryResponse(
            query=query,
            sub_queries=sub_queries if decompose_query and len(sub_queries) > 1 else [],
            results=results,
            latency_ms=round(latency_ms, 2),
            reranked=rerank,
            total_frames_searched=total_searched,
        )

    # ── Query decomposition ───────────────────────────────────────────────────

    @staticmethod
    def _decompose(query: str) -> List[str]:
        """
        Heuristic decomposition of compound queries into atomic sub-queries.

        Examples
        --------
        "person carrying a bag near the entrance"
          → ["person carrying a bag", "entrance area"]

        "red vehicle parked in zone 3 after 6 PM"
          → ["red vehicle parked", "zone 3"]
          (temporal part stripped — handled by ts_start/ts_end)

        Deliberately simple; replace with an LLM call for production.
        """
        # Strip temporal clauses (handled via filter params)
        cleaned = re.sub(r"(after|before|between|around)\s+\d{1,2}(:\d{2})?\s*(AM|PM|am|pm)?", "", query)
        cleaned = cleaned.strip()

        # Split on spatial prepositions
        spatial_splits = re.split(r"\s+(?:near|next to|beside|in front of|behind|at|in)\s+", cleaned, maxsplit=1)
        if len(spatial_splits) == 2:
            main, location = [s.strip() for s in spatial_splits]
            sub_queries = [q for q in [main, location] if q]
            if len(sub_queries) > 1:
                return sub_queries

        # Split on "and" for compound objects
        and_splits = re.split(r"\s+and\s+", cleaned)
        if len(and_splits) > 1:
            return [s.strip() for s in and_splits if s.strip()]

        return [cleaned if cleaned else query]

    # ── Re-ranking ────────────────────────────────────────────────────────────

    def _rerank(
        self,
        query: str,
        hits: List[Tuple[int, float, str]],
        embedder,
    ) -> List[Tuple[int, float, str]]:
        """
        Re-score candidates with the full query embedding against stored vectors.
        Falls back to original ANN scores if embeddings aren't available.
        """
        store_dir = Path(settings.INDEX_DIR)
        combined_npy = store_dir / "combined.npy"

        if self._all_embeddings is None:
            if combined_npy.exists():
                self._all_embeddings = np.load(str(combined_npy))
            else:
                # Try to build combined.npy from per-video files
                parts = []
                for f in sorted(store_dir.glob("*.npy")):
                    if f.stem != "combined":
                        parts.append(np.load(str(f)))
                if parts:
                    self._all_embeddings = np.vstack(parts)
                    np.save(str(combined_npy), self._all_embeddings)

        if self._all_embeddings is None:
            logger.warning("Re-ranking skipped: no .npy embeddings found.")
            return sorted(hits, key=lambda x: x[1], reverse=True)

        q_emb = embedder.embed_query(query)[0]  # shape (dim,)

        rescored = []
        for idx, _, sq in hits:
            if idx < len(self._all_embeddings):
                score = float(np.dot(q_emb, self._all_embeddings[idx]))
            else:
                score = 0.0
            rescored.append((idx, score, sq))

        return sorted(rescored, key=lambda x: x[1], reverse=True)
