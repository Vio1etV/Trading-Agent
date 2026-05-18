"""Quick manual test for fundamental data tools."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.fundamental_data import get_fundamental_context  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test fundamental data tools.")
    parser.add_argument("--ticker", default="NVDA", help="Stock ticker to query.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = get_fundamental_context(args.ticker)

    print(f"Ticker: {context['ticker']}")
    print("\nCompany profile")
    print(context["company_profile"])

    print("\nRecent SEC filings")
    print(context["recent_sec_filings"])


if __name__ == "__main__":
    main()
