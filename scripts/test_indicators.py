"""Quick manual test for short-horizon technical indicators."""

from __future__ import annotations

import argparse
from pathlib import Path
import pprint
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.indicators import compute_short_horizon_indicators  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test technical indicators.")
    parser.add_argument("--ticker", default="NVDA", help="Stock ticker to query.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compute_short_horizon_indicators(args.ticker)
    pprint.pp(result, sort_dicts=False)


if __name__ == "__main__":
    main()
