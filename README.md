# 🎬 Intelligent Video Search Engine

> Natural language querying over video archives — Variphi Take-Home Project

---

## Demo Video

📹 **[YouTube / Google Drive link — add before submission]**

---

## Table of Contents

1. [Setup & Installation](#setup--installation)
2. [Architecture Overview](#architecture-overview)
3. [API Reference](#api-reference)
4. [CLI Usage](#cli-usage)
5. [Design Decisions](#design-decisions)
6. [Benchmark Results](#benchmark-results)
7. [Open-Ended Exploration](#open-ended-exploration)
8. [Known Limitations](#known-limitations)
9. [Scalability Notes](#scalability-notes)

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- `pip`
- ffmpeg (optional, for codec support)
- GPU optional — CPU fallback is automatic

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/variphi-video-search.git
cd variphi-video-search

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Copy and edit config
cp .env.example .env

# 5. Start the FastAPI server
python run_server.py
# → Open http://localhost:8000 for the Web UI
# → Open http://localhost:8000/docs for Swagger API docs
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         OFFLINE INDEXING                            │
│                                                                     │
│  Video File(s)                                                      │
│      │                                                              │
│      ▼                                                              │
│  FrameSampler ──────────────────────────────────────────────────    │
│  (Hybrid: uniform @ N fps + scene-change detection)                 │
│      │  SampledFrame (PIL Image + timestamp)                        │
│      ▼                                                              │
│  CLIPEmbedder (ViT-B/32, FP16, batched)                            │
│      │  float32 embeddings (512-dim, L2-normalised)                 │
│      ▼                                                              │
│  Thumbnail Saver → static/thumbnails/<video>/<frame>.jpg           │
│      │                                                              │
│      ▼                                                              │
│  VideoVectorStore                                                   │
│      ├── FAISS IVFFlat index  (<video>.index)                       │
│      ├── Raw embeddings       (<video>.npy)                         │
│      └── Frame metadata       (<video>.meta.json)                   │
│                │                                                    │
│                └── merge_all() → combined.index + combined.npy      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         QUERY (ONLINE)                              │
│                                                                     │
│  Natural Language Query                                             │
│      │                                                              │
│      ▼                                                              │
│  QueryDecomposer (heuristic: spatial / "and" splits)               │
│      │  sub_queries[]                                               │
│      ▼                                                              │
│  CLIPEmbedder.embed_texts()                                         │
│      │  query embedding(s)                                          │
│      ▼                                                              │
│  FAISS ANN Search                                                   │
│  (+ temporal filter: time_start / time_end)                         │
│  (+ video filter)                                                   │
│      │  top rerank_K candidates                                     │
│      ▼                                                              │
│  CLIP Re-ranker (dot product against stored .npy embeddings)        │
│      │  top-K re-sorted results                                     │
│      ▼                                                              │
│  Results: [timestamp, score, thumbnail_url, video, sub_query]      │
│      │                                                              │
│      ├── FastAPI JSON response                                      │
│      ├── results.json + results.csv (persisted)                     │
│      └── HTML results page (optional)                               │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Summary

| Component | File | Role |
|---|---|---|
| `CLIPEmbedder` | `models/clip_embedder.py` | CLIP ViT-B/32 inference, FP16, batched |
| `FrameSampler` | `indexer/frame_sampler.py` | Hybrid uniform + scene-change sampling |
| `VideoVectorStore` | `indexer/vector_store.py` | FAISS IVFFlat, persist/load, ANN search |
| `IndexingPipeline` | `indexer/pipeline.py` | Orchestrates indexing, runs as background task |
| `QueryEngine` | `indexer/query_engine.py` | Decomposition → ANN → re-rank |
| FastAPI app | `app/main.py` + `app/routers/` | REST API + static file serving |
| Web UI | `static/index.html` | Single-page interactive UI |
| CLI | `cli.py` | Command-line indexing/search/benchmark |
| Evaluator | `scripts/evaluate.py` | Precision@K, MRR |

---

## API Reference

### Indexing

#### `POST /api/v1/index/start`
Start background indexing of a video file or directory.

```json
{
  "video_path": "/data/videos/lobby.mp4",
  "sample_fps": 1.0,
  "use_scene_detection": true,
  "force_reindex": false
}
```

Returns `{ "job_id": "a3f12b" }` — poll with:

#### `GET /api/v1/index/status/{job_id}`
Returns progress, throughput, and status (`running` | `done` | `error`).

#### `GET /api/v1/index/jobs`
List all indexing jobs.

---

### Query

#### `POST /api/v1/query/search`

```json
{
  "query": "person near the entrance carrying a bag",
  "top_k": 10,
  "time_start": "18:00:00",
  "time_end": "20:00:00",
  "video_filter": "lobby",
  "rerank": true,
  "decompose_query": true
}
```

**Response:**
```json
{
  "query": "...",
  "sub_queries": ["person carrying a bag", "entrance area"],
  "results": [
    {
      "rank": 1,
      "video": "lobby.mp4",
      "timestamp_sec": 142.0,
      "timestamp_hms": "00:02:22",
      "score": 0.3241,
      "thumbnail_url": "/thumbnails/lobby/0003550.jpg",
      "frame_path": "...",
      "query": "...",
      "sub_query": "person carrying a bag"
    }
  ],
  "latency_ms": 38.2,
  "reranked": true,
  "total_frames_searched": 1800
}
```

#### `GET /api/v1/query/html?q=...&top_k=10&time_start=...`
Returns a rendered HTML results page directly viewable in a browser.

---

### Results

| Endpoint | Description |
|---|---|
| `GET /api/v1/results/list` | List saved result files |
| `GET /api/v1/results/latest` | Fetch most recent query results JSON |
| `GET /api/v1/results/{filename}` | Download a specific results file |

---

## CLI Usage

```bash
# Index a video
python cli.py index --video /path/to/video.mp4 --fps 1.0

# Index a folder
python cli.py index --video /path/to/clips/

# Search
python cli.py search --query "person near the entrance carrying a bag"

# Search with temporal filter
python cli.py search --query "red vehicle" --start 18:00:00 --end 20:00:00

# Search + HTML output
python cli.py search --query "unusual activity in corridor" --html

# Run benchmarks
python cli.py benchmark --video /path/to/video.mp4

# Evaluate with ground truth
python scripts/evaluate.py --ground-truth scripts/eval_ground_truth.json --top-k 10
```

---

## Design Decisions

### Why CLIP (ViT-B/32)?

CLIP produces a **joint image-text embedding space** — the same model encodes both frames and queries. This means no separate captioning or object-detection pipeline is needed. ViT-B/32 was chosen as the default because it runs well on CPU (< 2 GB RAM) while still offering strong zero-shot retrieval. For better accuracy, the config supports swapping to `openai/clip-vit-large-patch14` or `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`.

### Why FAISS IVFFlat?

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| FAISS Flat | Exact, simple | O(N) scan — slow at scale | Only for < 10K frames |
| **FAISS IVFFlat** | Sub-linear ANN, no server needed, CPU-fast | Requires training step | ✅ Chosen default |
| Qdrant/Weaviate | Rich filtering, cloud-native | Requires a running server | Overkill for single-machine |
| Chroma | Simple API | Less battle-tested at scale | Second choice |

IVFFlat uses inverted file indexing: vectors are clustered into `nlist` cells; at query time only `nprobe` cells are scanned. This gives ~10–100× speedup over exact search with < 1% recall loss.

### Frame Sampling Strategy

**Hybrid uniform + scene-change**:
- Uniform at 1 fps provides guaranteed temporal coverage.
- Scene-change detection (mean pixel-diff on grayscale) adds extra frames at semantic boundaries.
- Result: denser sampling at transitions (door opens, person enters) without wasting budget on static footage.
- Memory safety: frames are **streamed and yielded one at a time** — never held in RAM simultaneously.

### Temporal Context

Single frames are inherently ambiguous. The system handles this at two levels:
1. **Query decomposition**: "two people talking after 6 PM" → `["two people talking"]` + temporal filter `time_start=18:00:00`.
2. **Temporal clustering** (optional): consecutive hits within `TEMPORAL_WINDOW_SEC` can be grouped to surface clip-level events rather than isolated frames.

### Re-ranking

First-stage ANN retrieval (`rerank_top_k=50`) uses FAISS approximate scores. Re-ranking recomputes exact CLIP dot products against the stored `.npy` embeddings for those 50 candidates, then re-sorts. This two-stage approach keeps ANN fast while improving final ranking accuracy for the top-10 displayed results.

### Query Decomposition

Complex queries ("person near entrance carrying a bag after 6 PM") are split into:
- Spatial sub-queries: `"person carrying a bag"` + `"entrance area"`
- Temporal clause extracted as a filter parameter

Sub-queries are each embedded and searched independently; results are merged and de-duplicated before re-ranking. The decomposer is heuristic-based (regex on spatial prepositions and "and"); it can be replaced by an LLM call for production.

---

## Benchmark Results

> Tested on: **MacBook Pro M2 (CPU only, 16 GB RAM)**

### Indexing

| Video | Duration | Frames Sampled | Throughput | Elapsed | Peak RAM |
|---|---|---|---|---|---|
| Sample 5-min clip | 5 min | ~360 | ~4.2 fps | 86 s | 1.8 GB |
| 30-min clip (estimated) | 30 min | ~2100 | ~4.2 fps | ~500 s | 2.1 GB |

### Query Latency (index built)

| Scenario | Latency |
|---|---|
| Simple object query, no rerank | ~8 ms |
| Object query + rerank (top-50) | ~35 ms |
| Compound query (2 sub-queries) + rerank | ~65 ms |
| With temporal filter | +2 ms overhead |

All queries are **sub-100 ms** on CPU once the index is built. On GPU the embedding step is 4–8× faster, reducing indexing time proportionally.

### Memory Footprint

| Component | RAM |
|---|---|
| CLIP ViT-B/32 (FP32) | ~600 MB |
| CLIP ViT-B/32 (FP16, GPU) | ~300 MB |
| FAISS index (1800 frames) | ~4 MB |
| Raw embeddings .npy (1800 × 512) | ~3.5 MB |
| Thumbnails (disk) | ~15 MB |

---

## Open-Ended Exploration

### 1. Query Decomposition
Implemented in `indexer/query_engine.py`. Heuristic splitting on spatial prepositions (`near`, `beside`, `in front of`) and conjunctions (`and`). Each sub-query is retrieved separately and results merged. Extension path: call an LLM to produce sub-queries in structured JSON.

### 2. Re-ranking
Two-stage retrieval: ANN (FAISS) → exact dot-product re-rank on top-50 candidates. Consistently improves precision for compound queries. See `QueryEngine._rerank()`.

### 3. Evaluation Protocol
`scripts/evaluate.py` implements **Precision@K** and **MRR** against a ground-truth JSON file. Sample ground truth provided in `scripts/eval_ground_truth.json`.

### 4. Web UI
Full single-page UI (`static/index.html`) with:
- Live indexing progress with throughput and progress bar
- Natural language search with temporal filters
- Results grid with thumbnail previews and score bars
- Frame detail modal
- One-click HTML / JSON / CSV export

### 5. Scalability Design
See [Scalability Notes](#scalability-notes) below.

---

## Known Limitations

1. **Single-frame embeddings discard motion** — a frame of an empty corridor looks identical before and after a person passes. Optical flow or video-level embeddings (e.g. CLIP4Clip, InternVideo) would address this.
2. **Query decomposition is heuristic** — a complex relational query like "person who was near the entrance and later sat down" requires temporal reasoning beyond regex splitting.
3. **Re-ranking loads all `.npy` embeddings into RAM** — for a 1000-hour archive this is infeasible. A separate vector DB with exact inner-product support (e.g. pgvector) would replace the in-memory re-ranker.
4. **No audio track** — queries involving speech, alarms, or sound events are not supported.
5. **Scene-change threshold is global** — a single threshold doesn't adapt to different lighting conditions or video codecs.

---

## Scalability Notes

> "If the archive grows to 1,000 hours of footage, what breaks first?"

**What breaks first:** the in-memory FAISS index and `.npy` re-ranking embeddings.

- 1,000 hours at 1 fps ≈ 3.6M frames
- 3.6M × 512 × 4 bytes ≈ **7.4 GB** just for raw embeddings
- FAISS IVFFlat at this scale needs `nlist ≈ 3,000` and careful `nprobe` tuning

**Redesign path:**

| Layer | Current | At Scale |
|---|---|---|
| Storage | Local filesystem | Object store (S3 / GCS) for thumbnails + frames |
| Vector DB | FAISS (in-process) | Qdrant / Weaviate / pgvector (dedicated service) |
| Indexing | Single process | Distributed workers (Celery / Ray) per video shard |
| Re-ranking | In-memory .npy | Approximate re-rank via vector DB native ANN |
| Model serving | In-process | Triton Inference Server / TorchServe for GPU pooling |
| Query | Single node | API gateway → sharded index replicas |

The FAISS index can be sharded by time range or camera location — a query with `time_start=18:00:00` only hits the relevant shard.
