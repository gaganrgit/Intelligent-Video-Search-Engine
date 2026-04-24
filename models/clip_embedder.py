"""
CLIP-based visual-semantic embedding model.

• Uses openai/clip-vit-base-patch32 (swappable via config).
• Supports FP16 inference on GPU, graceful CPU fallback.
• Thread-safe singleton so the model loads exactly once.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.config import settings

logger = logging.getLogger("video_search.model")

_lock = threading.Lock()
_instance: "CLIPEmbedder | None" = None


def get_embedder() -> "CLIPEmbedder":
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = CLIPEmbedder()
    return _instance


class CLIPEmbedder:
    """Thread-safe CLIP wrapper with batched inference and FP16 support."""

    def __init__(self):
        self.device = self._resolve_device()
        logger.info("Loading CLIP model '%s' on %s …", settings.CLIP_MODEL, self.device)
        t0 = time.time()
        self.processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL)
        self.model = CLIPModel.from_pretrained(settings.CLIP_MODEL)
        self.model.eval()

        # FP16 on CUDA only
        if self.device == "cuda" and settings.USE_FP16:
            self.model = self.model.half()
            logger.info("FP16 enabled")

        self.model.to(self.device)
        logger.info("Model loaded in %.1f s", time.time() - t0)

        self.embedding_dim = getattr(self.model.config, 'projection_dim', 512)

        # Effective batch size: halve on CPU to avoid OOM
        self.batch_size = settings.BATCH_SIZE if self.device == "cuda" else max(8, settings.BATCH_SIZE // 4)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device() -> str:
        if settings.DEVICE:
            return settings.DEVICE
        return "cuda" if torch.cuda.is_available() else "cpu"

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        """
        Encode a list of PIL images → L2-normalised float32 embeddings.
        Returns shape (N, embedding_dim).
        """
        all_embs: List[np.ndarray] = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self.model.get_image_features(**inputs)
            if hasattr(feats, 'pooler_output'):
                feats = feats.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2-normalise
            all_embs.append(feats.cpu().float().numpy())
        return np.vstack(all_embs)

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of text strings → L2-normalised float32 embeddings.
        Returns shape (N, embedding_dim).
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_text_features(**inputs)
        if hasattr(feats, 'pooler_output'):
            feats = feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    def embed_query(self, query: str) -> np.ndarray:
        """Single-query convenience wrapper. Returns shape (1, dim)."""
        return self.embed_texts([query])
