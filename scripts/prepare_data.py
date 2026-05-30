"""Fetch all tool data for a ticker and cache it as JSON.

Run this on a machine with working network access (yfinance, RSS,
search). The agents then read data/cache/<ticker>.json, so a GPU
server with a blocked or rate-limited IP never needs to call the
network itself.

Usage:
    python scripts/prepare_data.py --ticker NVDA
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.finance_news import format_news_for_agent  # noqa: E402
from src.tools.fundamental_data import format_fundamental_for_agent  # noqa: E402
from src.tools.indicators import format_indicators_for_agent  # noqa: E402
from src.tools.market_data import get_price_history  # noqa: E402
from src.tools.social_media import format_social_for_agent  # noqa: E402
from src.tools.world_news import format_world_news_for_agent  # noqa: E402


def fetch_price_history(ticker: str) -> list[dict]:
    """Cache the last 30 close prices for chart rendering on the frontend."""
    df = get_price_history(ticker, period="1mo", interval="1d")
    date_col = "Date" if "Date" in df.columns else "Datetime"
    rows = []
    for _, r in df.tail(30).iterrows():
        rows.append({
            "date": str(r[date_col])[:10],
            "close": float(r["Close"]),
        })
    return rows

CACHE_DIR = PROJECT_ROOT / "data" / "cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and cache tool data.")
    parser.add_argument("--ticker", default="NVDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper()

    print(f"Fetching data for {ticker} (this calls the network)...")

    bundle = {"ticker": ticker}

    # Each fetch is wrapped so one failing source does not lose the rest.
    steps = [
        ("fundamental", lambda: format_fundamental_for_agent(ticker)),
        ("news", lambda: format_news_for_agent(ticker)),
        ("social", lambda: format_social_for_agent(ticker)),
        ("world_news", lambda: format_world_news_for_agent(
            f"{ticker} stock industry macro news")),
        ("indicators", lambda: format_indicators_for_agent(ticker)),
        ("price_history", lambda: fetch_price_history(ticker)),
    ]

    for key, fn in steps:
        try:
            bundle[key] = fn()
            print(f"  [OK]   {key}")
        except Exception as exc:
            bundle[key] = f"{key} unavailable: {exc}"
            print(f"  [FAIL] {key}: {exc}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{ticker}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)

    print(f"\nSaved cache -> {path}")


if __name__ == "__main__":
    main()
