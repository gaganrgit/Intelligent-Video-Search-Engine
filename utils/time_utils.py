"""
Time utility helpers.
"""

from __future__ import annotations
from typing import Optional


def sec_to_hms(seconds: float) -> str:
    """Convert fractional seconds to HH:MM:SS string."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def hms_to_sec(hms: Optional[str]) -> Optional[float]:
    """
    Convert HH:MM:SS (or MM:SS) string to total seconds.
    Returns None if input is None or empty.
    """
    if not hms:
        return None
    parts = hms.strip().split(":")
    try:
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = int(parts[0]), float(parts[1])
            return m * 60 + s
        else:
            return float(parts[0])
    except ValueError:
        return None
