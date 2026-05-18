"""Quick manual test for yfinance-backed market data tools."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.market_data import (  # noqa: E402
    get_company_info,
    get_latest_price,
    get_price_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test market data tools.")
    parser.add_argument("--ticker", default="NVDA", help="Stock ticker to query.")
    parser.add_argument("--period", default="1mo", help="History period.")
    parser.add_argument("--interval", default="1d", help="History interval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Ticker: {args.ticker.upper()}")
    print("\nLatest price")
    print(get_latest_price(args.ticker))

    print("\nCompany info")
    print(get_company_info(args.ticker))

    print("\nRecent price history")
    history = get_price_history(
        args.ticker,
        period=args.period,
        interval=args.interval,
    )
    print(history.tail().to_string(index=False))


if __name__ == "__main__":
    main()
