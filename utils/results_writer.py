"""
Results Writer
==============
Persists query results to both JSON and CSV in static/results/.
Each query creates a timestamped file so historical results are preserved.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import List

from app.config import settings
from app.schemas import FrameResult

logger = logging.getLogger("video_search.results")


def write_results(query: str, results: List[FrameResult]) -> dict:
    """
    Write results to JSON and CSV. Returns paths dict.
    """
    ts = int(time.time())
    safe_query = "".join(c if c.isalnum() or c in "-_ " else "_" for c in query)[:50].strip()
    base_name = f"{ts}_{safe_query}"

    results_dir = Path(settings.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / f"{base_name}.json"
    csv_path = results_dir / f"{base_name}.csv"

    # ── JSON ──────────────────────────────────────────────────────────────────
    payload = {
        "query": query,
        "timestamp": ts,
        "count": len(results),
        "results": [r.model_dump() for r in results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # ── CSV ───────────────────────────────────────────────────────────────────
    fieldnames = ["rank", "video", "timestamp_hms", "timestamp_sec", "score", "frame_path", "query", "sub_query"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "rank": r.rank,
                "video": r.video,
                "timestamp_hms": r.timestamp_hms,
                "timestamp_sec": r.timestamp_sec,
                "score": r.score,
                "frame_path": r.frame_path,
                "query": r.query,
                "sub_query": r.sub_query or "",
            })

    logger.info("Results saved: %s", json_path.name)
    return {"json": str(json_path), "csv": str(csv_path)}


def latest_results_json() -> Path | None:
    """Return the most recently written results JSON, or None."""
    results_dir = Path(settings.RESULTS_DIR)
    files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None
