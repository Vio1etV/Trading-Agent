"""Analyst Agent: reads fundamentals, news, social, macro -> writes analyst_report.

Backed by a local Gemma-2-9B-it model. Three pieces:
  load_analyst_model()  - load the model once at startup
  _build_prompt()       - gather tool data + assemble the prompt
  make_analyst_node()   - return the LangGraph node function
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import ANALYST_MODEL_PATH, MAX_NEW_TOKENS, TEMPERATURE, USE_4BIT
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
    """Load the analyst model and tokenizer. Call this once at startup."""
    tokenizer = AutoTokenizer.from_pretrained(ANALYST_MODEL_PATH)

    load_kwargs = {"device_map": "cuda:0"}

    if USE_4BIT:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        ANALYST_MODEL_PATH, **load_kwargs
    )
    return model, tokenizer


def _build_prompt(ticker: str, user_question: str, intent: str) -> str:
    """Load cached tool data and assemble a prompt that adapts to intent.

    - intent == "trade"  -> write the structured analyst report (template).
    - other intents      -> answer the question conversationally using the data.
    """
    data = load_data_bundle(ticker)
    fundamentals = data["fundamental"]
    news = data["news"]
    social = data["social"]
    macro = data["world_news"]

    if intent == "trade":
        report_format = ANALYST_REPORT_TEMPLATE.format(ticker=ticker)
        instruction = (
            "Write a structured analyst report following the format below. "
            "Base every statement on the data. Do not invent numbers."
        )
        format_section = f"=== REPORT FORMAT (follow exactly) ===\n{report_format}"
    else:
        instruction = (
            "Answer the user's question directly using only the data provided. "
            "Keep it to 2-4 short paragraphs. Do NOT use the report template. "
            "Do not invent numbers."
        )
        format_section = ""

    return f"""{instruction}

User question: {user_question}

=== FUNDAMENTAL DATA ===
{fundamentals}

=== FINANCE NEWS ===
{news}

=== SOCIAL MEDIA ===
{social}

=== MACRO / WORLD NEWS ===
{macro}

{format_section}
"""


def make_analyst_node(model, tokenizer):
    """Return the LangGraph node function, with the model bound in."""

    def analyst_node(state: TradingState) -> dict:
        ticker = state["ticker"]
        question = state["user_question"]
        intent = state.get("intent") or "trade"

        prompt = _build_prompt(ticker, question, intent)

        # Qwen2.5 supports a system role.
        messages = [
            {"role": "system", "content": ANALYST_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ]
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