"""Collect one full data package for a ticker and user question."""

from __future__ import annotations

import argparse
from pathlib import Path
import pprint
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.finance_news import get_yahoo_rss_news  # noqa: E402
from src.tools.fundamental_data import get_fundamental_context  # noqa: E402
from src.tools.indicators import compute_short_horizon_indicators  # noqa: E402
from src.tools.market_data import get_latest_price, get_price_history  # noqa: E402
from src.tools.social_media import get_social_media_context  # noqa: E402
from src.tools.world_news import search_world_news  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test all data sources.")
    parser.add_argument("--ticker", default="NVDA", help="Stock ticker to query.")
    parser.add_argument(
        "--query",
        default="US China AI chip export controls Nvidia",
        help="World news / macro context query.",
    )
    parser.add_argument("--limit", type=int, default=5, help="News/search limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    price_history = get_price_history(args.ticker, period="1mo", interval="1d")
    data_package = {
        "ticker": args.ticker.upper(),
        "user_query_for_context": args.query,
        "risk_inputs": {
            "latest_price": get_latest_price(args.ticker),
            "price_history_tail": price_history.tail().to_dict(orient="records"),
            "indicators": compute_short_horizon_indicators(args.ticker),
        },
        "analyst_inputs": {
            "fundamental_data": get_fundamental_context(args.ticker),
            "finance_news": get_yahoo_rss_news(args.ticker, limit=args.limit),
            "social_media": get_social_media_context(args.ticker),
            "world_news": search_world_news(args.query, limit=args.limit),
        },
    }

    pprint.pp(data_package, sort_dicts=False)


if __name__ == "__main__":
    main()
