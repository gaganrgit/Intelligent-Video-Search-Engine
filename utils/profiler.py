"""
Profiler — lightweight benchmarking helpers.
Tracks peak RSS memory, wall-clock time, and throughput.
"""

from __future__ import annotations

import resource
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional


def peak_rss_mb() -> float:
    """Return peak resident set size in MB (Linux/macOS)."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux: ru_maxrss is in kilobytes; macOS: bytes
        import platform
        if platform.system() == "Darwin":
            return usage.ru_maxrss / 1024 / 1024
        return usage.ru_maxrss / 1024
    except Exception:
        return 0.0


@dataclass
class BenchmarkResult:
    label: str
    elapsed_sec: float
    peak_rss_mb: float
    throughput: Optional[float] = None    # items/sec
    throughput_unit: str = "items/sec"
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        tp = f" | {self.throughput:.1f} {self.throughput_unit}" if self.throughput else ""
        return (
            f"[{self.label}] {self.elapsed_sec:.2f}s"
            f"{tp} | peak RAM: {self.peak_rss_mb:.1f} MB"
        )


@contextmanager
def timer(label: str, n_items: int = 0, unit: str = "frames/sec"):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    tp = n_items / elapsed if (n_items and elapsed > 0) else None
    result = BenchmarkResult(
        label=label,
        elapsed_sec=round(elapsed, 3),
        peak_rss_mb=round(peak_rss_mb(), 1),
        throughput=round(tp, 2) if tp else None,
        throughput_unit=unit,
    )
    print(result)
