"""
Frame Sampler
=============
Hybrid strategy: uniform sampling at `sample_fps` augmented by
scene-change detection (pixel-diff between consecutive decoded frames).

Rationale
---------
• Pure uniform: simple but misses burst-scene changes and wastes budget
  on static footage.
• Scene-change only: can oversample rapid-motion segments.
• Hybrid: uniform grid ensures temporal coverage; scene changes add
  semantically richer samples without proportional cost increase.

Memory safety
-------------
Frames are decoded and yielded one at a time; the PIL Image is released
after embedding. A 30-minute 1080p video at 1 fps ≈ 1 800 frames ×
~6 MB raw = ~10 GB if held in RAM simultaneously — we never do that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List

import cv2
import numpy as np
from PIL import Image

from app.config import settings

logger = logging.getLogger("video_search.sampler")


@dataclass
class SampledFrame:
    frame_index: int       # raw frame number in the video
    timestamp_sec: float
    image: Image.Image
    is_scene_change: bool = False


class FrameSampler:
    """
    Yields SampledFrame objects from a video file.

    Parameters
    ----------
    sample_fps : float
        Target uniform sampling rate (frames/second of video).
    scene_threshold : float
        Mean pixel-diff to flag a scene change (0–255 scale).
    use_scene_detection : bool
        Whether to include scene-change frames on top of uniform.
    """

    def __init__(
        self,
        sample_fps: float = settings.SAMPLE_FPS,
        scene_threshold: float = settings.SCENE_CHANGE_THRESHOLD,
        use_scene_detection: bool = settings.USE_SCENE_DETECTION,
    ):
        self.sample_fps = sample_fps
        self.scene_threshold = scene_threshold
        self.use_scene_detection = use_scene_detection

    # ── Public ────────────────────────────────────────────────────────────────

    def sample(self, video_path: str | Path) -> Generator[SampledFrame, None, None]:
        """
        Yield sampled frames one-at-a-time (memory-safe streaming).
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        native_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(native_fps / self.sample_fps)))

        logger.info(
            "Sampling '%s': native_fps=%.2f, step=%d (~%.2f fps), scene_detect=%s",
            Path(video_path).name,
            native_fps,
            step,
            native_fps / step,
            self.use_scene_detection,
        )

        frame_idx = 0
        prev_gray: np.ndarray | None = None
        sampled_indices: set[int] = set()

        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            timestamp = frame_idx / native_fps
            is_scene_change = False

            # ── Scene-change detection ────────────────────────────────────────
            if self.use_scene_detection and prev_gray is not None:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                diff = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
                if diff > self.scene_threshold:
                    is_scene_change = True

            if self.use_scene_detection:
                prev_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            # ── Decide whether to yield this frame ────────────────────────────
            is_uniform = (frame_idx % step == 0)
            if is_uniform or (is_scene_change and frame_idx not in sampled_indices):
                sampled_indices.add(frame_idx)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                yield SampledFrame(
                    frame_index=frame_idx,
                    timestamp_sec=timestamp,
                    image=pil_img,
                    is_scene_change=is_scene_change and not is_uniform,
                )

            frame_idx += 1

        cap.release()
        logger.info("Sampling complete: %d frames yielded from '%s'", len(sampled_indices), Path(video_path).name)

    def estimate_frame_count(self, video_path: str | Path) -> int:
        """Cheap estimate of how many frames will be sampled (for progress bars)."""
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        step = max(1, int(round(fps / self.sample_fps)))
        uniform_count = max(1, total // step)
        scene_bonus = int(uniform_count * 0.15) if self.use_scene_detection else 0
        return uniform_count + scene_bonus
