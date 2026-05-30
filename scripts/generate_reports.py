"""Generate analyst + risk manager reports for all cached tickers.

Both agents now use Qwen2.5-7B, so we load the model ONCE and reuse it
for both phases. The model is loaded in 4-bit to fit in 8GB VRAM.

Usage:
    python scripts/generate_reports.py
    python scripts/generate_reports.py --tickers NVDA AAPL MSFT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import PROJECT_ROOT
from src.tools.data_cache import CACHE_DIR

REPORTS_DIR = PROJECT_ROOT / "data" / "reports"


def get_cached_tickers() -> list[str]:
    """Return all tickers that have cached data."""
    return sorted(p.stem.upper() for p in CACHE_DIR.glob("*.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers", nargs="*", default=None,
        help="Tickers to process (default: all cached tickers)",
    )
    args = parser.parse_args()

    tickers = args.tickers or get_cached_tickers()
    if not tickers:
        print("No cached data found. Run batch_prepare_data.py first.")
        sys.exit(1)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check which tickers already have complete reports
    todo = []
    for t in tickers:
        report_path = REPORTS_DIR / f"{t}.json"
        if report_path.exists():
            with open(report_path, encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("analyst_report") and existing.get("risk_report"):
                print(f"[SKIP] {t}: complete report already exists")
                continue
        todo.append(t)

    if not todo:
        print("All reports already generated.")
        return

    print(f"\nGenerating reports for {len(todo)} tickers...")

    # Both agents use Qwen2.5-7B, so we load it once and share.
    # We load the analyst model — same weights as risk manager.
    print("Loading Qwen2.5-7B (4-bit) — shared by both agents...")

    from src.agents.analyst import load_analyst_model, make_analyst_node
    from src.agents.risk_manager import make_risk_node

    model, tokenizer = load_analyst_model()
    analyst_node = make_analyst_node(model, tokenizer)
    risk_node = make_risk_node(model, tokenizer)

    question = "Should I buy {ticker} right now?"

    for i, ticker in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {ticker}...", flush=True)
        try:
            state = {
                "ticker": ticker,
                "user_question": question.format(ticker=ticker),
                "analyst_report": None,
                "risk_report": None,
                "trader_recommendation": None,
            }

            # Run analyst
            analyst_out = analyst_node(state)
            state.update(analyst_out)
            print(f"  Analyst: done ({len(analyst_out['analyst_report'])} chars)")

            # Run risk manager (same model, different prompt)
            risk_out = risk_node(state)
            state.update(risk_out)
            print(f"  Risk Manager: done ({len(risk_out['risk_report'])} chars)")

            # Save combined report
            report = {
                "ticker": ticker,
                "user_question": state["user_question"],
                "analyst_report": state["analyst_report"],
                "risk_report": state["risk_report"],
            }
            report_path = REPORTS_DIR / f"{ticker}.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"  Saved: {report_path}")

        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\nDone. Reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()