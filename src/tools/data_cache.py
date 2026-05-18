"""Load pre-fetched tool data from a JSON cache file.

Network access (yfinance, RSS, search) is unreliable on some GPU
servers. scripts/prepare_data.py fetches everything once on a machine
with good network and writes data/cache/<ticker>.json. Agents then
read from that file instead of calling the tools live.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Keys every cached bundle must contain.
REQUIRED_KEYS = ["fundamental", "news", "social", "world_news", "indicators"]


def cache_path(ticker: str) -> Path:
    """Return the cache file path for a ticker."""
    return CACHE_DIR / f"{ticker.upper()}.json"


def load_data_bundle(ticker: str) -> dict[str, Any]:
    """Load the cached tool data for a ticker.

    Raises a clear error if the cache is missing, telling the caller
    how to generate it.
    """
    path = cache_path(ticker)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached data for {ticker} at {path}.\n"
            f"Generate it on a machine with network access:\n"
            f"    python scripts/prepare_data.py --ticker {ticker}"
        )

    with open(path, encoding="utf-8") as f:
        bundle = json.load(f)

    missing = [k for k in REQUIRED_KEYS if k not in bundle]
    if missing:
        raise ValueError(f"Cache file {path} is missing keys: {missing}")

    return bundle
