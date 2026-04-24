#!/usr/bin/env python3
"""
CLI — Intelligent Video Search Engine
======================================
Usage examples:

  # Index a video file
  python cli.py index --video /path/to/video.mp4

  # Index a directory of clips
  python cli.py index --video /path/to/clips/ --fps 2.0

  # Search
  python cli.py search --query "person near the entrance carrying a bag"

  # Search with temporal filter
  python cli.py search --query "red vehicle" --start 18:00:00 --end 20:00:00

  # Search and output HTML
  python cli.py search --query "unusual activity" --html

  # Run benchmark
  python cli.py benchmark --video /path/to/video.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings
from indexer.frame_sampler import FrameSampler
from indexer.pipeline import IndexingPipeline, IndexJob
from indexer.query_engine import QueryEngine
from indexer.vector_store import VideoVectorStore
from models.clip_embedder import get_embedder
from utils.html_renderer import render_html_results
from utils.profiler import peak_rss_mb, timer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli")


# ── Index command ─────────────────────────────────────────────────────────────

def cmd_index(args):
    pipeline = IndexingPipeline()
    job = IndexJob(job_id="cli", video_path=args.video)

    print(f"\n📁  Indexing: {args.video}")
    print(f"    FPS={args.fps}  scene_detect={not args.no_scene}  force={args.force}\n")

    t0 = time.perf_counter()
    pipeline.run(
        job=job,
        sample_fps=args.fps,
        use_scene_detection=not args.no_scene,
        force_reindex=args.force,
    )
    elapsed = time.perf_counter() - t0

    if job.status == "done":
        print(f"\n✅  Done!")
        print(f"    Frames indexed : {job.frames_indexed:,}")
        print(f"    Throughput     : {job.throughput_fps:.1f} frames/sec")
        print(f"    Wall time      : {elapsed:.1f} s")
        print(f"    Peak RAM       : {peak_rss_mb():.1f} MB")
    else:
        print(f"\n❌  Error: {job.message}")
        sys.exit(1)


# ── Search command ────────────────────────────────────────────────────────────

def cmd_search(args):
    engine = QueryEngine()

    print(f"\n🔍  Query: \"{args.query}\"")
    if args.start:
        print(f"    Time filter: {args.start} → {args.end or 'end'}")

    t0 = time.perf_counter()
    try:
        response = engine.search(
            query=args.query,
            top_k=args.top_k,
            time_start=args.start,
            time_end=args.end,
            rerank=not args.no_rerank,
            decompose_query=not args.no_decompose,
        )
    except RuntimeError as e:
        print(f"\n❌  {e}")
        sys.exit(1)

    print(f"\n    {len(response.results)} results in {response.latency_ms:.1f} ms"
          f"  (frames searched: {response.total_frames_searched:,})\n")

    if response.sub_queries and len(response.sub_queries) > 1:
        print(f"    Sub-queries: {' | '.join(response.sub_queries)}\n")

    # Print table
    col_w = [5, 10, 8, 10, 40]
    header = f"{'Rank':<{col_w[0]}}  {'Time':<{col_w[1]}}  {'Score':<{col_w[2]}}  {'Video':<{col_w[3]}}  Frame"
    print("    " + header)
    print("    " + "─" * 70)
    for r in response.results:
        vid = r.video[:col_w[3]]
        frame = r.frame_path[:col_w[4]]
        print(f"    {r.rank:<{col_w[0]}}  {r.timestamp_hms:<{col_w[1]}}  {r.score:<{col_w[2]}.4f}  {vid:<{col_w[3]}}  {frame}")

    # HTML output
    if args.html:
        html_path = render_html_results(args.query, response.results, settings.RESULTS_DIR)
        print(f"\n🌐  HTML results: {html_path}")

    # JSON dump
    if args.json:
        out = {
            "query": response.query,
            "latency_ms": response.latency_ms,
            "results": [r.model_dump() for r in response.results],
        }
        print(json.dumps(out, indent=2))


# ── Benchmark command ─────────────────────────────────────────────────────────

def cmd_benchmark(args):
    print("\n🏎  Benchmark mode\n")

    # ── Step 1: Embedding throughput ──────────────────────────────────────────
    from PIL import Image as PILImage
    import numpy as np

    embedder = get_embedder()
    dummy_images = [PILImage.new("RGB", (224, 224)) for _ in range(64)]

    with timer("Image embedding (64 frames)", n_items=64):
        embedder.embed_images(dummy_images)

    dummy_texts = ["person walking near the door"] * 32
    with timer("Text embedding (32 queries)", n_items=32, unit="queries/sec"):
        embedder.embed_texts(dummy_texts)

    # ── Step 2: Indexing throughput ───────────────────────────────────────────
    if args.video:
        pipeline = IndexingPipeline()
        job = IndexJob(job_id="bench", video_path=args.video)
        t0 = time.perf_counter()
        pipeline.run(job=job, sample_fps=args.fps, use_scene_detection=True, force_reindex=True)
        elapsed = time.perf_counter() - t0
        print(f"\n[Indexing] {job.frames_indexed} frames in {elapsed:.2f}s"
              f" = {job.throughput_fps:.1f} frames/sec | peak RAM {peak_rss_mb():.1f} MB")

    # ── Step 3: Query latency ─────────────────────────────────────────────────
    engine = QueryEngine()
    test_queries = [
        "person near the entrance",
        "red vehicle parked",
        "anything unusual in the corridor",
    ]
    latencies = []
    for q in test_queries:
        try:
            t0 = time.perf_counter()
            engine.search(q, top_k=10, rerank=True)
            latencies.append((time.perf_counter() - t0) * 1000)
        except Exception:
            pass

    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"\n[Query latency] avg={avg:.1f} ms  min={min(latencies):.1f} ms  max={max(latencies):.1f} ms")
    print(f"[Peak RAM at benchmark end] {peak_rss_mb():.1f} MB\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Video Search Engine — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_idx = sub.add_parser("index", help="Index a video file or directory")
    p_idx.add_argument("--video", required=True, help="Path to video file or folder")
    p_idx.add_argument("--fps", type=float, default=1.0, help="Sampling FPS (default: 1.0)")
    p_idx.add_argument("--no-scene", action="store_true", help="Disable scene-change detection")
    p_idx.add_argument("--force", action="store_true", help="Force re-indexing")

    # search
    p_srch = sub.add_parser("search", help="Natural language search")
    p_srch.add_argument("--query", "-q", required=True, help="Search query")
    p_srch.add_argument("--top-k", type=int, default=10)
    p_srch.add_argument("--start", help="Temporal filter start HH:MM:SS")
    p_srch.add_argument("--end", help="Temporal filter end HH:MM:SS")
    p_srch.add_argument("--no-rerank", action="store_true")
    p_srch.add_argument("--no-decompose", action="store_true")
    p_srch.add_argument("--html", action="store_true", help="Also save HTML results")
    p_srch.add_argument("--json", action="store_true", help="Print JSON output")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run performance benchmarks")
    p_bench.add_argument("--video", help="Optional video to benchmark indexing")
    p_bench.add_argument("--fps", type=float, default=1.0)

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)


if __name__ == "__main__":
    main()
