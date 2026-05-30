"""Batch-fetch data for a list of tickers across sectors.

Usage:
    python scripts/batch_prepare_data.py
"""

from __future__ import annotations

import argparse
import time

# Diverse tickers across sectors for training data variety
TICKERS = [
    # Tech / Semiconductors
    "NVDA", "AMD", "INTC", "TSM", "AVGO", "QCOM",
    # Big Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
    # Finance
    "JPM", "GS", "BAC", "V",
    # Healthcare
    "JNJ", "PFE", "UNH", "LLY",
    # Energy
    "XOM", "CVX", "NEE",
    # Consumer
    "WMT", "COST", "NKE", "SBUX",
    # Industrial / Defense
    "BA", "LMT", "CAT",
]


def main():
    # Import here so the script can show help without needing all deps
    from src.tools.fundamental_data import format_fundamental_for_agent
    from src.tools.finance_news import format_news_for_agent
    from src.tools.social_media import format_social_for_agent
    from src.tools.world_news import format_world_news_for_agent
    from src.tools.indicators import format_indicators_for_agent
    from src.tools.data_cache import cache_path, CACHE_DIR

    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers", nargs="*", default=None,
        help="Override ticker list (default: built-in 30 tickers)",
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds to wait between tickers to avoid rate limits",
    )
    args = parser.parse_args()

    tickers = args.tickers or TICKERS
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    succeeded = []
    failed = []

    for i, ticker in enumerate(tickers, 1):
        path = cache_path(ticker)
        if path.exists():
            print(f"[{i}/{len(tickers)}] {ticker}: already cached, skipping")
            succeeded.append(ticker)
            continue

        print(f"[{i}/{len(tickers)}] {ticker}: fetching...", end=" ", flush=True)
        try:
            bundle = {
                "fundamental": format_fundamental_for_agent(ticker),
                "news": format_news_for_agent(ticker),
                "social": format_social_for_agent(ticker),
                "world_news": format_world_news_for_agent(
                    f"{ticker} stock market news outlook"
                ),
                "indicators": format_indicators_for_agent(ticker),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(bundle, f, indent=2)
            print("OK")
            succeeded.append(ticker)
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append((ticker, str(e)))

        if i < len(tickers):
            time.sleep(args.delay)

    print(f"\nDone: {len(succeeded)} succeeded, {len(failed)} failed")
    if failed:
        print("Failed tickers:")
        for t, err in failed:
            print(f"  {t}: {err}")


if __name__ == "__main__":
    main()
