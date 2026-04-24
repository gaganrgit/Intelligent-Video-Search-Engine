"""
Thumbnail utility — saves resized JPEG thumbnails for each indexed frame.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from app.config import settings

logger = logging.getLogger("video_search.thumbnail")

THUMB_SIZE = (320, 180)


def save_thumbnail(image: Image.Image, video_stem: str, frame_index: int) -> str:
    out_dir = Path(settings.THUMBNAIL_DIR) / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{frame_index:07d}.jpg"

    if not out_path.exists():
        thumb = image.copy()
        thumb.thumbnail(THUMB_SIZE, Image.LANCZOS)
        thumb.save(str(out_path), "JPEG", quality=75, optimize=True)

    # Always return a forward-slash URL-safe relative path
    return f"{video_stem}/{frame_index:07d}.jpg"
