"""
HTML Results Renderer
=====================
Generates a self-contained HTML page displaying thumbnails, timestamps,
and scores for a given set of query results. Saved to static/results/.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

from app.schemas import FrameResult


_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Search Results: {query}</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0; margin: 0; padding: 20px; }}
  h1 {{ font-size: 1.4rem; color: #7eb8f7; margin-bottom: 4px; }}
  .meta {{ color: #888; font-size: 0.85rem; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
  .card {{ background: #1a1a2e; border-radius: 10px; overflow: hidden; border: 1px solid #2a2a4a; }}
  .card img {{ width: 100%; display: block; object-fit: cover; height: 160px; background: #111; }}
  .card .info {{ padding: 10px 12px; }}
  .card .ts {{ font-size: 1.1rem; font-weight: 600; color: #7eb8f7; }}
  .card .score {{ font-size: 0.8rem; color: #aaa; margin-top: 2px; }}
  .card .vid {{ font-size: 0.75rem; color: #666; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .card .sq {{ font-size: 0.75rem; color: #9b7; margin-top: 2px; font-style: italic; }}
  .rank {{ float: right; background: #7eb8f7; color: #000; font-weight: 700; font-size: 0.75rem;
           border-radius: 4px; padding: 1px 6px; }}
</style>
</head>
<body>
<h1>🔍 &ldquo;{query}&rdquo;</h1>
<p class="meta">{count} results &nbsp;·&nbsp; generated {ts}</p>
<div class="grid">
{cards}
</div>
</body>
</html>
"""

_CARD = """\
  <div class="card">
    <img src="{thumb_url}" alt="frame at {ts_hms}" loading="lazy">
    <div class="info">
      <span class="rank">#{rank}</span>
      <div class="ts">⏱ {ts_hms}</div>
      <div class="score">Score: {score}</div>
      <div class="vid">📹 {video}</div>
      {sq_line}
    </div>
  </div>"""


def render_html_results(query: str, results: List[FrameResult], out_dir: str) -> str:
    """
    Render an HTML results page and save it. Returns the file path.
    """
    cards_html = []
    for r in results:
        sq_line = f'<div class="sq">sub-query: {r.sub_query}</div>' if r.sub_query else ""
        cards_html.append(_CARD.format(
            thumb_url=r.thumbnail_url,
            ts_hms=r.timestamp_hms,
            rank=r.rank,
            score=r.score,
            video=r.video,
            sq_line=sq_line,
        ))

    html = _TEMPLATE.format(
        query=query,
        count=len(results),
        ts=time.strftime("%Y-%m-%d %H:%M:%S"),
        cards="\n".join(cards_html),
    )

    ts = int(time.time())
    safe = "".join(c if c.isalnum() else "_" for c in query)[:40]
    out_path = Path(out_dir) / f"{ts}_{safe}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)
