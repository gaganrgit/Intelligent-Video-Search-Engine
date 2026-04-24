#!/usr/bin/env python3
"""
Evaluation Protocol
===================
Measures retrieval quality using Precision@K and Mean Reciprocal Rank (MRR).

Ground-truth format (eval_ground_truth.json):
[
  {
    "query": "person near the entrance",
    "relevant_timestamps": [
      {"video": "lobby.mp4", "timestamp_sec": 142.0, "tolerance_sec": 5.0}
    ]
  },
  ...
]

Usage:
  python scripts/evaluate.py --ground-truth eval_ground_truth.json --top-k 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexer.query_engine import QueryEngine
from app.schemas import FrameResult

logging.basicConfig(level=logging.WARNING)


def timestamps_match(result: FrameResult, relevant: Dict, tolerance: float) -> bool:
    """Check if a result is within `tolerance` seconds of a ground-truth timestamp."""
    if relevant.get("video") and relevant["video"].lower() not in result.video.lower():
        return False
    return abs(result.timestamp_sec - relevant["timestamp_sec"]) <= tolerance


def precision_at_k(hits: List[bool]) -> float:
    return sum(hits) / len(hits) if hits else 0.0


def reciprocal_rank(hits: List[bool]) -> float:
    for i, h in enumerate(hits):
        if h:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(ground_truth_path: str, top_k: int = 10, rerank: bool = True):
    with open(ground_truth_path) as f:
        gt_entries = json.load(f)

    engine = QueryEngine()
    results_summary = []
    all_precisions = []
    all_rrs = []

    print(f"\n{'Query':<50}  {'P@K':>6}  {'RR':>6}  Latency")
    print("─" * 75)

    for entry in gt_entries:
        query = entry["query"]
        relevant = entry["relevant_timestamps"]
        tolerance = entry.get("tolerance_sec", 5.0)

        t0 = time.perf_counter()
        try:
            response = engine.search(query, top_k=top_k, rerank=rerank)
        except Exception as e:
            print(f"  ERROR on '{query}': {e}")
            continue
        latency_ms = (time.perf_counter() - t0) * 1000

        hits = []
        for r in response.results:
            is_hit = any(timestamps_match(r, rel, tolerance) for rel in relevant)
            hits.append(is_hit)

        pk = precision_at_k(hits)
        rr = reciprocal_rank(hits)
        all_precisions.append(pk)
        all_rrs.append(rr)

        q_display = query[:48] + ".." if len(query) > 50 else query
        print(f"  {q_display:<50}  {pk:>6.3f}  {rr:>6.3f}  {latency_ms:.1f} ms")

        results_summary.append({
            "query": query,
            "precision_at_k": round(pk, 4),
            "reciprocal_rank": round(rr, 4),
            "latency_ms": round(latency_ms, 1),
            "top_k": top_k,
            "hits": hits,
        })

    if all_precisions:
        mean_pk = sum(all_precisions) / len(all_precisions)
        mrr = sum(all_rrs) / len(all_rrs)
        print(f"\n{'─'*75}")
        print(f"  Mean Precision@{top_k}: {mean_pk:.4f}")
        print(f"  MRR             : {mrr:.4f}")
        print(f"  Queries evaluated: {len(all_precisions)}\n")

        out_path = Path("eval_results.json")
        with open(out_path, "w") as f:
            json.dump({
                "mean_precision_at_k": round(mean_pk, 4),
                "mrr": round(mrr, 4),
                "top_k": top_k,
                "queries": results_summary,
            }, f, indent=2)
        print(f"  Results saved to {out_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth JSON")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()
    evaluate(args.ground_truth, top_k=args.top_k, rerank=not args.no_rerank)


if __name__ == "__main__":
    main()
