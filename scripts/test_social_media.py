"""Quick manual test for social media data tools."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.social_media import get_social_media_context  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test social media data tools.")
    parser.add_argument("--ticker", default="NVDA", help="Stock ticker to query.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = get_social_media_context(args.ticker)

    print(f"Ticker: {context['ticker']}")
    print("\nStockTwits")
    print(context["stocktwits"])

    print("\nReddit")
    print(context["reddit"])


if __name__ == "__main__":
    main()
