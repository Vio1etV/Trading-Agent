"""Router Agent: classifies the user's question and decides what to do.

Uses the same Qwen2.5-7B model as the analyst/risk agents (already
loaded). One extra LLM call (~1s) that prevents the full pipeline
from running for every question.

Outputs JSON-like dict with: in_scope, intent, ticker, direct_reply.
"""

from __future__ import annotations

import json
import re

from src.config import TEMPERATURE
from src.graph.state import TradingState
from src.tools.data_cache import CACHE_DIR


def _available_tickers() -> list[str]:
    """List of tickers with cached data, so the router can pick a sensible one."""
    if not CACHE_DIR.exists():
        return []
    return sorted(p.stem.upper() for p in CACHE_DIR.glob("*.json"))


ROUTER_INSTRUCTIONS = """You are the router for a multi-agent stock trading assistant.
Classify the user's question and decide what the system should do.

You MUST output ONLY a JSON object. No prose before or after.

JSON schema:
{
  "in_scope": true/false,
  "intent": one of [trade, impact, technical, info, broker, vague, meta, out_of_scope],
  "ticker": "<UPPERCASE_SYMBOL>" or null,
  "direct_reply": "<short answer to show the user>" or null
}

Intent definitions:
- trade        : user explicitly asks for a buy/sell/hold decision on a specific stock.
                 Examples: "Should I buy NVDA?", "买不买 TSLA", "is NVDA a buy here?"
- impact       : how an event/factor/news affects a specific stock.
                 Examples: "How does the Iran situation affect NVDA?", "Fed rate impact on AAPL?"
- technical    : asks about technical indicators / chart setup / support / resistance.
                 Examples: "NVDA technical setup?", "TSLA chart looks?"
- info         : asks what a company does, its sector, basic profile.
                 Examples: "What is NVDA?", "What does UNH do?"
- broker       : asks where/how to buy stocks generally.
                 Examples: "Where do I buy NVDA?", "How do I start trading?"
- vague        : trading/stock related but no specific ticker or unclear ask.
                 Examples: "What should I buy?", "I have never invested."
- meta         : asks about the assistant itself.
                 Examples: "Who are you?", "What can you do?"
- out_of_scope : not stock/trading/finance related.
                 Examples: "How do I play Apex?", "What is Steam?", "How to recoil control?"

Rules:
- For intents {trade, impact, technical, info}: direct_reply = null. The system will dispatch to the right agents.
- For intents {broker, meta, out_of_scope}: write a short direct_reply (1-3 sentences). No agents will run.
- For intent "vague": direct_reply MUST be a clarifying QUESTION that asks the user for the missing context.
  Ask about capital amount, time horizon (intraday / weeks / years), risk tolerance
  (conservative / moderate / aggressive), and which sectors they care about.
  Pick the 2-3 most relevant. Example:
    "Before I can suggest a stock, tell me:
     1) how much capital you're working with,
     2) your time horizon (days, months, years), and
     3) any sector you're drawn to (AI, semiconductors, healthcare, etc.)."
  Do NOT give a generic 'it depends on your goals' answer.
- For ticker: extract from question if mentioned (e.g. "NVDA", "Nvidia" -> "NVDA"). Otherwise null.
- If the user mentions a company name ("Nvidia", "Apple"), convert to ticker ("NVDA", "AAPL").

Output ONLY the JSON. No markdown fences, no explanation."""


def _fallback_decision(question: str) -> dict:
    """Cheap keyword-based fallback when JSON parsing fails."""
    q = question.lower()
    if any(w in q for w in ["who are you", "what can you do", "你是谁", "能干啥"]):
        return {
            "in_scope": False,
            "intent": "meta",
            "ticker": None,
            "direct_reply": (
                "I'm a multi-agent trading assistant. I can analyze stocks, give you "
                "market context, technical setups, and trading recommendations. "
                "Try asking 'Should I buy NVDA?' or 'How does X affect TSLA?'"
            ),
        }
    return {
        "in_scope": False,
        "intent": "out_of_scope",
        "ticker": None,
        "direct_reply": (
            "Sorry, I'm a trading assistant — I can only help with stocks and markets. "
            "Try asking about a ticker like 'Should I buy NVDA?'"
        ),
    }


def make_router_node(model, tokenizer):
    """Return the LangGraph node function for the router."""

    available = _available_tickers()
    ticker_hint = ""
    if available:
        ticker_hint = f"\n\nTickers we have cached data for: {', '.join(available)}."

    def router_node(state: TradingState) -> dict:
        question = state["user_question"]
        instructions = ROUTER_INSTRUCTIONS + ticker_hint

        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": question},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=True,
        )

        prompt_len = inputs["input_ids"].shape[-1]
        raw = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

        # Strip code fences and find the first JSON object.
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            decision = _fallback_decision(question)
        else:
            try:
                decision = json.loads(m.group(0))
            except json.JSONDecodeError:
                decision = _fallback_decision(question)

        # Normalize ticker; fall back to UI selection if present.
        ticker = decision.get("ticker")
        if ticker:
            ticker = str(ticker).upper().strip()
        if not ticker and state.get("ticker"):
            ticker = state["ticker"]

        update = {
            "intent": decision.get("intent") or "out_of_scope",
            "direct_reply": decision.get("direct_reply"),
            "ticker": ticker,
        }
        return update

    return router_node
