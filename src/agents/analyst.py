"""Analyst Agent: reads fundamentals, news, social, macro -> writes analyst_report.

Backed by a local Gemma-2-9B-it model. Three pieces:
  load_analyst_model()  - load the model once at startup
  _build_prompt()       - gather tool data + assemble the prompt
  make_analyst_node()   - return the LangGraph node function
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import ANALYST_MODEL_PATH, MAX_NEW_TOKENS, TEMPERATURE
from src.graph.state import TradingState
from src.tools.data_cache import load_data_bundle


# --- The exact report format the model must follow ---
ANALYST_REPORT_TEMPLATE = """## Analyst Report: {ticker}

Sentiment: <Bullish / Neutral / Bearish>
Confidence: <High / Medium / Low>

Fundamentals: <2-3 sentences on financial health and valuation>
News: <2-3 sentences on recent events>
Social Sentiment: <1-2 sentences on retail mood>
Macro Context: <1-2 sentences on industry/policy backdrop>

Key Risks:
- <risk 1>
- <risk 2>"""


# --- The role instruction given to the model ---
ANALYST_INSTRUCTIONS = """You are a professional equity analyst.
Read the data below and write a concise analyst report.
Base every statement on the data provided. Do not invent numbers.
Follow the report format exactly. Replace every <...> placeholder
with real content and keep the section headers unchanged."""


def load_analyst_model():
    """Load the Gemma model and tokenizer. Call this once at startup."""
    tokenizer = AutoTokenizer.from_pretrained(ANALYST_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        ANALYST_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


def _build_prompt(ticker: str, user_question: str) -> str:
    """Load cached tool data and assemble the full prompt text."""
    data = load_data_bundle(ticker)
    fundamentals = data["fundamental"]
    news = data["news"]
    social = data["social"]
    macro = data["world_news"]

    report_format = ANALYST_REPORT_TEMPLATE.format(ticker=ticker)

    return f"""{ANALYST_INSTRUCTIONS}

User question: {user_question}

=== FUNDAMENTAL DATA ===
{fundamentals}

=== FINANCE NEWS ===
{news}

=== SOCIAL MEDIA ===
{social}

=== MACRO / WORLD NEWS ===
{macro}

=== REPORT FORMAT (follow exactly) ===
{report_format}
"""


def make_analyst_node(model, tokenizer):
    """Return the LangGraph node function, with the model bound in.

    LangGraph nodes must take only `state`, so we use a factory:
    the model/tokenizer are captured in the closure.
    """

    def analyst_node(state: TradingState) -> dict:
        ticker = state["ticker"]
        question = state["user_question"]

        prompt = _build_prompt(ticker, question)

        # Gemma-2 has no system role, so everything goes in one user turn.
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
        )

        # Decode only the newly generated tokens, not the prompt.
        prompt_len = inputs["input_ids"].shape[-1]
        generated = outputs[0][prompt_len:]
        report = tokenizer.decode(generated, skip_special_tokens=True)

        return {"analyst_report": report.strip()}

    return analyst_node
