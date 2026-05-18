"""Quick manual test for free finance news retrieval tools."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.finance_news import format_news_for_agent, get_yahoo_rss_news  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test finance news retrieval tools.")
    parser.add_argument("--ticker", default="NVDA", help="Stock ticker to query.")
    parser.add_argument("--limit", type=int, default=5, help="Number of articles.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Ticker: {args.ticker.upper()}")
    print("\nStructured articles")
    for article in get_yahoo_rss_news(args.ticker, limit=args.limit):
        print(article)

    print("\nFormatted analyst input")
    print(format_news_for_agent(args.ticker, limit=args.limit))


if __name__ == "__main__":
    main()
