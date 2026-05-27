"""Generate gold Trader recommendations using GPT-4 and format as SFT data.

Reads analyst + risk reports from data/reports/, sends them to GPT-4
with varied user questions, and saves the training data as JSONL.

Usage:
    $env:OPENAI_API_KEY = "sk-..."
    python scripts/generate_sft_data.py
    python scripts/generate_sft_data.py --questions-per-ticker 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from src.config import PROJECT_ROOT

REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
SFT_DIR = PROJECT_ROOT / "data" / "sft"

# Varied user questions to create diverse training examples.
# {ticker} will be replaced with the actual ticker symbol.
USER_QUESTIONS = [
    "Should I buy {ticker} right now?",
    "Is {ticker} a good short-term trade this week?",
    "What's the risk/reward on {ticker} at current levels?",
    "I'm thinking of selling my {ticker} position. What do you think?",
    "Is now a good entry point for {ticker}?",
    "Should I hold or sell {ticker}?",
    "What's your trading recommendation for {ticker}?",
]

# The same system prompt and template used by the Trader agent,
# so the GPT-4 outputs match the format the Trader will learn.
TRADER_INSTRUCTIONS = """You are a professional stock trader.
You receive two reports from your team:
1. An Analyst Report covering fundamentals, news, sentiment, and macro context.
2. A Risk Manager Report covering technical indicators, support/resistance, and risk assessment.

Your job is to synthesize both reports and the user's question into a
single, actionable trading recommendation.

Rules:
- Base every number on the reports. Do not invent prices or levels.
- If the analyst is bullish but technicals are bearish, weigh the risk carefully.
- The Entry Zone should be consistent with the Risk Manager's support/resistance.
- The Stop Loss should be at or below the Risk Manager's support level.
- The Target Price should be realistic given the current price and resistance.
- Follow the report format exactly. Replace every <...> placeholder
  with real content and keep the section headers unchanged."""

TRADER_REPORT_TEMPLATE = """## Trading Recommendation: {ticker}

Action: <Buy / Hold / Sell>
Confidence: <High / Medium / Low>

Entry Zone: <$ low - $ high>
Stop Loss: <$ value>
Target Price: <$ value>

Reasoning: <2-3 sentences explaining the decision, referencing both the analyst and risk reports>

Key Risks:
- <risk 1>
- <risk 2>"""


def build_user_prompt(
    ticker: str,
    user_question: str,
    analyst_report: str,
    risk_report: str,
) -> str:
    report_format = TRADER_REPORT_TEMPLATE.format(ticker=ticker)
    return f"""User question: {user_question}

=== ANALYST REPORT ===
{analyst_report}

=== RISK MANAGER REPORT ===
{risk_report}

=== YOUR OUTPUT FORMAT (follow exactly) ===
{report_format}
"""


def call_gpt4(system_prompt: str, user_prompt: str, api_key: str) -> str:
    """Call GPT-4 via the OpenAI API."""
    import openai

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def validate_recommendation(text: str) -> bool:
    """Basic check that the output has the expected sections."""
    required = ["Action:", "Entry Zone:", "Stop Loss:", "Target Price:", "Reasoning:"]
    return all(section in text for section in required)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions-per-ticker", type=int, default=5,
        help="Number of question variants per ticker (default: 5, max: 7)",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds between API calls to avoid rate limits",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY first:")
        print('  $env:OPENAI_API_KEY = "sk-..."')
        sys.exit(1)

    # Find all report files
    report_files = sorted(REPORTS_DIR.glob("*.json"))
    if not report_files:
        print("No reports found. Run generate_reports.py first.")
        sys.exit(1)

    questions_per = min(args.questions_per_ticker, len(USER_QUESTIONS))
    total = len(report_files) * questions_per
    print(f"Generating SFT data: {len(report_files)} tickers × "
          f"{questions_per} questions = {total} examples")

    SFT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SFT_DIR / "trader_sft.jsonl"

    # Load existing examples to support resuming
    existing = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                # Key by ticker + question to avoid duplicates
                user_msg = ex["messages"][1]["content"]
                existing.add(user_msg[:200])
        print(f"  Resuming: {len(existing)} examples already generated")

    succeeded = 0
    failed = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for report_file in report_files:
            with open(report_file, encoding="utf-8") as f:
                report = json.load(f)

            ticker = report["ticker"]
            analyst = report["analyst_report"]
            risk = report["risk_report"]

            for q_idx in range(questions_per):
                question = USER_QUESTIONS[q_idx].format(ticker=ticker)
                user_prompt = build_user_prompt(
                    ticker, question, analyst, risk
                )

                # Skip if already generated
                if user_prompt[:200] in existing:
                    print(f"  [{ticker}] Q{q_idx+1}: already exists, skipping")
                    succeeded += 1
                    continue

                print(f"  [{ticker}] Q{q_idx+1}: {question[:50]}...", end=" ", flush=True)

                try:
                    recommendation = call_gpt4(
                        TRADER_INSTRUCTIONS, user_prompt, api_key
                    )

                    if not validate_recommendation(recommendation):
                        print("WARN: missing sections, saving anyway")

                    # Format as SFT training example (chat format)
                    example = {
                        "messages": [
                            {"role": "system", "content": TRADER_INSTRUCTIONS},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": recommendation},
                        ]
                    }

                    out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    out_f.flush()
                    succeeded += 1
                    print("OK")

                except Exception as e:
                    failed += 1
                    print(f"FAILED: {e}")

                time.sleep(args.delay)

    print(f"\nDone: {succeeded} succeeded, {failed} failed")
    print(f"SFT data saved to: {output_path}")


if __name__ == "__main__":
    main()
