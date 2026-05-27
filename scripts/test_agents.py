"""End-to-end test: run all three agents for a ticker and print reports."""

from __future__ import annotations

import argparse

from src.graph.workflow import build_workflow


def main():
    parser = argparse.ArgumentParser(description="Test the trading agent pipeline")
    parser.add_argument("--ticker", required=True, help="Stock ticker, e.g. NVDA")
    parser.add_argument(
        "--question",
        default=None,
        help="User question (default: 'Should I buy <ticker> right now?')",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    question = args.question or f"Should I buy {ticker} right now?"

    print("Building workflow (loading models, this takes a while)...")
    graph = build_workflow()

    print(f"\nRunning agents for {ticker}...\n")
    result = graph.invoke(
        {
            "ticker": ticker,
            "user_question": question,
            "analyst_report": None,
            "risk_report": None,
            "trader_recommendation": None,
        }
    )

    sections = [
        ("ANALYST REPORT", "analyst_report"),
        ("RISK MANAGER REPORT", "risk_report"),
        ("TRADER RECOMMENDATION", "trader_recommendation"),
    ]

    for title, key in sections:
        print("=" * 60)
        print(f"[{title}]")
        print("=" * 60)
        print(result.get(key) or "(empty)")
        print("-" * 60)

    print("\nDone.")


if __name__ == "__main__":
    main()