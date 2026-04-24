#!/usr/bin/env python3
"""
Server entrypoint — run with:
    python run_server.py
or:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info",
        access_log=True,
    )
