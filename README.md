# Intelligent Video Search Engine

Natural language querying over video archives — type a plain-English description, get back ranked timestamps and frame previews.

| | |
|---|---|
| **Language** | Python 3.10+ |
| **Framework** | FastAPI + Uvicorn |
| **Embedding Model** | OpenAI CLIP ViT-B/32 |
| **Vector Store** | FAISS IVFFlat |

---

## Table of Contents

1. [Setup & Installation](#setup--installation)
2. [Overview](#overview)
3. [Indexing Pipeline](#indexing-pipeline-offline)
4. [Query Pipeline](#query-pipeline-online--400-ms)
5. [API Reference](#api-reference)
6. [CLI Usage](#cli-usage)
7. [Web UI](#web-ui)
8. [Project Structure](#project-structure)
9. [Design Decisions](#design-decisions)
10. [Benchmark](#benchmark)
11. [Scalability](#scalability)
12. [Known Limitations](#known-limitations)
13. [Evaluation](#evaluation)

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- `pip`
- `ffmpeg` (optional, for broader codec support)
- GPU optional — CPU fallback is automatic

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/video-search.git
cd video-search

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Copy and edit config
cp .env.example .env

# 5. Start the FastAPI server
python run_server.py
# → http://localhost:8000        Web UI
# → http://localhost:8000/docs   Swagger API docs
```

> **Note:** The CLIP ViT-B/32 model (~605 MB) is downloaded from HuggingFace on first run. Subsequent starts use the local cache (~7.5 s load time).

---

## Overview

The system is split into two phases:

1. **Indexing** — a one-time offline step that processes video and stores it in a searchable format.
2. **Querying** — a fast online step that matches text queries against the stored index in under a second.

**Key insight:** CLIP understands both images and text in the same mathematical space, so a text sentence can be directly compared to a video frame — no manual tagging required.

---

## Architecture Diagrams

```
┌─────────────────────────────────────────────────────────────────────┐
│                         OFFLINE INDEXING                            │
│                                                                     │
│  Video File(s)                                                      │
│      │                                                              │
│      ▼                                                              │
│  FrameSampler                                                       │
│  (Hybrid: uniform @ N fps + scene-change detection)                 │
│      │  SampledFrame (PIL Image + timestamp)                        │
│      ▼                                                              │
│  CLIPEmbedder (ViT-B/32, FP16, batched)                             │
│      │  float32 embeddings (512-dim, L2-normalised)                 │
│      ▼                                                              │
│  Thumbnail Saver → static/thumbnails/<video>/<frame>.jpg            │
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
│  QueryDecomposer (heuristic: spatial / "and" splits)                │
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

---

## Indexing Pipeline (Offline)

### Step 1 — Video Input

Accepts a path to a single video file or a folder of clips. Supported extensions: `.mp4`, `.mkv`, `.avi`, `.mov`, and others. Each video is processed one at a time.

### Step 2 — Frame Sampling

A hybrid strategy selects only the most useful frames:

- **Uniform sampling** — one frame per second of video, guaranteeing full temporal coverage.
- **Scene-change detection** — grayscale frame differencing captures abrupt visual transitions (e.g. a door opening) that fall between uniform samples.
- **Memory safety** — frames are yielded one at a time and released after processing; the system handles long videos on limited RAM without issue.

### Step 3 — CLIP Embedding

Each sampled frame is passed through CLIP, which converts it into a **512-dimensional float32 vector** (embedding). Images that look similar — and text that describes an image — end up at nearby coordinates in this space.

- **Batched processing** — 32 frames per batch on both CPU and GPU.
- **L2 normalisation** — all embeddings are normalised to unit length so similarity is measured by direction, not magnitude.
- **FP16 / CPU fallback** — GPU runs in half-precision; CPU falls back to full-precision automatically.

### Step 4 — Thumbnail Saving

Each sampled frame is saved as a 320×180 JPEG to `static/thumbnails/<video>/<frame>.jpg`, served as static files by FastAPI.

### Step 5 — FAISS Index

All frame embeddings are stored in a FAISS index:

| Corpus size | Index type | Behaviour |
|---|---|---|
| < 400 frames | `FlatIP` | Exact search against every frame |
| ≥ 400 frames | `IVFFlat` | Clusters vectors; searches only the most relevant clusters |

### Step 6 — Persisted Files

Three files are written to `index_store/` per video:

| File | Contents |
|---|---|
| `<name>.index` | FAISS binary index |
| `<name>.meta.json` | Per-vector metadata: video name, timestamp (seconds + HH:MM:SS), thumbnail path |
| `<name>.npy` | Raw float32 embeddings for re-ranking |

After all videos are indexed, a single `combined.index` is merged for cross-video search.

---

## Query Pipeline (Online, < 400 ms)

### Stage 1 — Query Decomposition

Complex queries are split into simpler atomic sub-queries before searching:

- Spatial prepositions (`near`, `beside`, `at`, etc.) trigger a split.
- Compound objects are split on `and`.
- Temporal phrases (`after 6 PM`, `before 8 AM`) are stripped and converted to time-filter parameters.

**Example:**
```
Input:  "person near the entrance carrying a bag after 6 PM"
Output: sub-query 1 → "person carrying a bag"
        sub-query 2 → "entrance area"
        time_start  → 18:00:00
```

Results from all sub-queries are merged and de-duplicated; a frame matching multiple sub-queries scores higher.

### Stage 2 — Text Embedding + ANN Search

Each sub-query is embedded by CLIP into a 512-d vector (same space as frame embeddings). The `combined.index` is loaded from disk into RAM on the **first query only**; all subsequent queries reuse the in-memory index with no disk reads. FAISS returns the top-50 most similar frames using inner product (equivalent to cosine similarity after L2 normalisation).

Temporal and video-name filters are applied post-retrieval; the system over-fetches up to 10× the requested count to ensure enough candidates survive filtering.

### Stage 3 — Re-Ranking

The top-50 FAISS candidates are re-scored using **exact dot-product** against the raw `.npy` embeddings, then re-sorted. This corrects ordering errors introduced by approximate search cheaply — exact comparison against 50 vectors instead of thousands.

### Stage 4 — Result Assembly

Each result includes:

- Rank, timestamp (seconds + HH:MM:SS), similarity score (0–1)
- Thumbnail URL, video filename, originating sub-query

Results are saved synchronously to a timestamped JSON and CSV in `static/results/`, and a rolling `results.csv` is maintained.

---

## API Reference

All request bodies are validated against Pydantic schemas before any ML code is touched. Invalid payloads (empty query, `top_k` outside 1–100, malformed time strings) return a `422` immediately. Indexing runs as a background thread; poll `/index/status/{job_id}` to track progress.

### Indexing

#### `POST /api/v1/index/start`
```json
{
  "video_path": "/data/videos/lobby.mp4",
  "sample_fps": 1.0,
  "use_scene_detection": true,
  "force_reindex": false
}
```
Returns `{ "job_id": "a3f12b" }`.

#### `GET /api/v1/index/status/{job_id}`
Returns frames done, throughput, and status (`running` | `done` | `error`).

#### `GET /api/v1/index/jobs`
List all indexing jobs started in this server session.

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

### Results & Health

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/results/list` | List saved result files |
| `GET` | `/api/v1/results/latest` | Fetch most recent results JSON |
| `GET` | `/api/v1/results/{filename}` | Download a specific results file |
| `GET` | `/health` | Server health check |
| `GET` | `/` | Single-page web UI |

---

## CLI Usage

```bash
# Index a single video
python cli.py index --video /path/to/video.mp4 --fps 1.0

# Index a folder of clips
python cli.py index --video /path/to/clips/

# Search
python cli.py search --query "person near the entrance carrying a bag"

# Search with temporal filter
python cli.py search --query "red vehicle" --start 18:00:00 --end 20:00:00

# Search and export to HTML
python cli.py search --query "unusual activity in corridor" --html

# Run benchmark
python cli.py benchmark --video /path/to/video.mp4

# Evaluate against ground truth
python scripts/evaluate.py --ground-truth scripts/eval_ground_truth.json --top-k 10
```

---

## Web UI

Served at `http://localhost:8000`. Single self-contained HTML file — no React, no Node.js, no build step.

- **Index panel** — enter video path, set FPS and scene-detection options, start indexing.
- **Live progress** — polls status every 1.5 s; shows progress bar, frame count, throughput, elapsed time.
- **Search panel** — query box, time-range filters, top-K selector, video filter, re-ranking and decomposition toggles.
- **Results grid** — thumbnail cards with timestamp, score bar, video name, and sub-query label.
- **Frame modal** — click any card for a full-size preview.
- **Export** — one-click JSON/CSV download and standalone HTML results page.

---

## Project Structure

```
app/
  main.py               FastAPI app setup, CORS, static file serving, lifespan
  config.py             All configuration (model, paths, batch size, FPS, etc.)
  schemas.py            Pydantic models for all request/response shapes
  routers/
    index_router.py     Indexing job endpoints
    query_router.py     Search endpoints (JSON + HTML)
    results_router.py   Result listing and download endpoints

models/
  clip_embedder.py      CLIP loader, batched image embedding, text embedding

indexer/
  frame_sampler.py      Hybrid uniform + scene-change frame extraction
  vector_store.py       FAISS build, save, load, merge, ANN search
  pipeline.py           Full indexing orchestration and background job management
  query_engine.py       Decomposition, ANN search, re-ranking, result assembly

utils/
  thumbnail.py          Frame resizing and JPEG saving
  results_writer.py     JSON and CSV result persistence
  html_renderer.py      Standalone HTML results page generation
  time_utils.py         Seconds ↔ HH:MM:SS conversion
  profiler.py           Wall-clock timing, throughput, peak RAM measurement

static/
  index.html            Single-page web UI

cli.py                  CLI for indexing, searching, and benchmarking
run_server.py           Uvicorn server entry point
scripts/evaluate.py     Precision@K and MRR evaluation against ground truth
```

---

## Benchmark (CPU, No GPU)

### Indexing

| Metric | Value |
|---|---|
| Frames sampled (8 s clip) | 8 |
| Indexing throughput | 0.73 fps (including model load) |
| Total elapsed time | 11.03 s |
| Model load time (cached) | ~7.5 s |
| CLIP model size | 605 MB |

### Query Latency

| Query type | Latency |
|---|---|
| Simple + re-rank | ~289 ms |
| Simple, no re-rank | ~50 ms |
| Compound + re-rank | ~350 ms |
| With temporal filter | +5 ms overhead |

### Memory (Windows CPU)

| Component | RAM |
|---|---|
| CLIP ViT-B/32 (CPU FP32) | ~650 MB |
| FastAPI + Uvicorn | ~50 MB |
| FAISS index (8 frames) | < 1 MB |

### Extended Benchmark (Apple M2, CPU only, 16 GB RAM)

| Video | Duration | Frames Sampled | Throughput | Elapsed |
|---|---|---|---|---|
| Sample clip | 5 min | ~360 | ~4.2 fps | 86 s |
| Long clip (est.) | 30 min | ~2,100 | ~4.2 fps | ~500 s |

| Query type | Latency |
|---|---|
| Simple, no re-rank | ~8 ms |
| Simple + re-rank (top-50) | ~35 ms |
| Compound + re-rank | ~65 ms |
| With temporal filter | +2 ms overhead |

| Component | RAM |
|---|---|
| CLIP ViT-B/32 (FP32) | ~600 MB |
| CLIP ViT-B/32 (FP16, GPU) | ~300 MB |
| FAISS index (1,800 frames) | ~4 MB |
| Raw embeddings (1,800 × 512) | ~3.5 MB |
| Thumbnails on disk | ~15 MB |

---

## Design Decisions

### CLIP ViT-B/32
Produces a joint image-text embedding space — the same model encodes both frames and queries, eliminating the need for a separate captioning or object-detection pipeline. ViT-B/32 runs well on CPU (< 2 GB RAM) with strong zero-shot accuracy. The config supports swapping to `openai/clip-vit-large-patch14` for better accuracy at higher resource cost.

### FAISS Index Selection

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| `FlatIP` | Exact, simple | O(N) scan — slow at scale | Auto-selected for < 400 frames |
| **`IVFFlat`** | Sub-linear ANN, no server, CPU-fast | Requires a training step | Default for larger corpora |
| Qdrant / Weaviate | Rich filtering, cloud-native | Requires a running server | Recommended at 1,000 h+ scale |

IVFFlat clusters vectors into `nlist` cells; at query time only `nprobe` cells are scanned — ~10–100× faster than exact search with < 1% recall loss.

### Hybrid Frame Sampling
Uniform sampling at 1 fps guarantees full temporal coverage. Scene-change detection (mean pixel-diff on grayscale) adds extra samples at semantic boundaries (door opens, person enters) without wasting budget on static footage. Frames are streamed and yielded one at a time — never all held in RAM simultaneously.

### Two-Stage Retrieval
ANN retrieval via FAISS fetches the top-50 candidates fast but approximately. Re-ranking recomputes exact dot products against the stored `.npy` embeddings for only those 50 candidates, then re-sorts. This gives the accuracy of exact search at the speed of approximate search.

### Temporal Handling
Temporal context is handled at two levels: query decomposition strips phrases like "after 6 PM" into `time_start` filter parameters, and post-retrieval filtering discards candidates outside the requested window (with 10× over-fetching to ensure enough results survive).

### Other Choices

| Decision | Rationale |
|---|---|
| FastAPI + Uvicorn | Async-native, automatic Pydantic validation, auto-generated docs |
| Background indexing | Keeps API responsive during multi-minute indexing jobs |
| Single-file HTML UI | Zero build step; no Node.js; works immediately on any machine |

---

## Scalability

At **1,000 hours of footage** (≈ 3.6 M frames, 7.4 GB of raw embeddings), the current in-process design hits RAM limits. Recommended migration path:

| Layer | Current | At Scale |
|---|---|---|
| Vector storage | FAISS in-process | Qdrant or pgvector as a dedicated service |
| Raw embeddings | `.npy` files in RAM | Stored in vector DB, fetched on demand |
| Thumbnails | Local disk | Object storage (S3 / GCS) |
| Indexing | Single background thread | Distributed workers (Celery + Redis) |
| Model serving | Loaded once in FastAPI | Triton Inference Server with GPU pooling |
| Search | Single FastAPI node | Load-balanced replicas with sharded indices |

---

## Known Limitations

- **No temporal reasoning** — CLIP embeds single frames; it cannot model motion or changes over time. Video-level models (CLIP4Clip, InternVideo) would address this.
- **Heuristic decomposer** — regex-based splitting fails on complex relational queries; an LLM-based decomposer would handle arbitrary language.
- **Re-ranking RAM ceiling** — loading all `.npy` files into memory is infeasible beyond a few hundred hours; a vector DB with built-in exact re-ranking is needed at scale.
- **No audio** — purely visual; queries about sounds, speech, or alarms are not supported.
- **Global scene-change threshold** — a single pixel-difference threshold suits neither bright outdoor nor dark indoor footage equally; per-video adaptive calibration would improve frame selection.
- **First-run model download** — the 605 MB CLIP model is fetched from HuggingFace on first run; air-gapped deployments must pre-cache it.

---

## Evaluation

`scripts/evaluate.py` measures retrieval quality against a ground-truth JSON file:

- **Precision@K** — fraction of top-K results that are genuinely relevant.
- **MRR (Mean Reciprocal Rank)** — average inverse rank of the first relevant result.
