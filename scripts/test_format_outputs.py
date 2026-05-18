"""Test the format_*_for_agent() functions for all tools."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.finance_news import format_news_for_agent
from src.tools.fundamental_data import format_fundamental_for_agent
from src.tools.indicators import format_indicators_for_agent
from src.tools.social_media import format_social_for_agent
from src.tools.world_news import format_world_news_for_agent


DIVIDER = "\n" + "=" * 60 + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test all format_*_for_agent functions.")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--query", default="Nvidia AI chip export controls US China")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ticker = args.ticker
    query = args.query

    sections = [
        ("FUNDAMENTAL (Analyst)", lambda: format_fundamental_for_agent(ticker)),
        ("FINANCE NEWS (Analyst)", lambda: format_news_for_agent(ticker)),
        ("SOCIAL MEDIA (Analyst)", lambda: format_social_for_agent(ticker)),
        ("WORLD NEWS (Analyst)", lambda: format_world_news_for_agent(query)),
        ("INDICATORS (Risk Manager)", lambda: format_indicators_for_agent(ticker)),
    ]

    for title, fn in sections:
        print(DIVIDER + f"[{title}]")
        try:
            print(fn())
        except Exception as exc:
            print(f"ERROR: {exc}")

    print(DIVIDER)


if __name__ == "__main__":
    main()
