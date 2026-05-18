"""Run the analyst + risk manager agents end-to-end and check their output.

This loads both local models, so it MUST run on a GPU node.

Usage:
    python scripts/test_agents.py --ticker NVDA --question "Should I buy NVDA?"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.workflow import build_workflow


# Headers each report should contain, per our agreed schema.
ANALYST_REQUIRED = [
    "Analyst Report",
    "Sentiment:",
    "Confidence:",
    "Fundamentals:",
    "News:",
    "Social Sentiment:",
    "Macro Context:",
    "Key Risks:",
]

RISK_REQUIRED = [
    "Risk Manager Report",
    "Current Price:",
    "Risk Level:",
    "Trend:",
    "Technical Signals:",
    "RSI:",
    "MACD:",
    "Bollinger:",
    "Support:",
    "Resistance:",
    "Entry Zone:",
    "Stop Loss:",
    "Risk Assessment:",
]


def check_report(name: str, report: str | None, required: list[str]) -> None:
    """Print the report, then list which required headers are missing."""
    print("\n" + "=" * 60)
    print(f"[{name}]")
    print("=" * 60)
    if not report:
        print("ERROR: report is empty / None")
        return
    print(report)
    missing = [h for h in required if h not in report]
    print("-" * 60)
    if missing:
        print(f"WARNING: missing {len(missing)} expected sections: {missing}")
    else:
        print("OK: all expected sections present.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--question", default="Should I buy NVDA right now?")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Building workflow (loading models, this takes a while)...")
    app = build_workflow()

    initial_state = {
        "ticker": args.ticker,
        "user_question": args.question,
        "analyst_report": None,
        "risk_report": None,
        "trader_recommendation": None,
    }

    print(f"\nRunning agents for {args.ticker}...")
    result = app.invoke(initial_state)

    check_report("ANALYST REPORT", result.get("analyst_report"), ANALYST_REQUIRED)
    check_report("RISK MANAGER REPORT", result.get("risk_report"), RISK_REQUIRED)

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
