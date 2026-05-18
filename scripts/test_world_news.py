"""Quick manual test for world news and macro context search tools."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.world_news import (  # noqa: E402
    format_world_news_results,
    search_world_news,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test world news search tools.")
    parser.add_argument("query", help="Search query.")
    parser.add_argument("--limit", type=int, default=5, help="Number of results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Query: {args.query}")
    results = search_world_news(args.query, limit=args.limit)

    print("\nStructured results")
    for result in results:
        print(result)

    print("\nFormatted analyst input")
    print(format_world_news_results(args.query, results))


if __name__ == "__main__":
    main()
