"""Risk Manager Agent: reads technical indicators -> writes risk_report.

Backed by a local Qwen2.5-7B-Instruct model. Same three-piece shape as
the analyst: load model, build prompt, make node.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import RISK_MODEL_PATH, MAX_NEW_TOKENS, TEMPERATURE, USE_4BIT
from src.graph.state import TradingState
from src.tools.data_cache import load_data_bundle


# --- The exact report format the model must follow ---
RISK_REPORT_TEMPLATE = """## Risk Manager Report: {ticker}

Current Price: <$ value>
Risk Level: <High / Medium / Low>
Trend: <Uptrend / Sideways / Downtrend>

Technical Signals:
- RSI: <value + overbought/oversold/neutral>
- MACD: <bullish / bearish>
- Bollinger: <where price sits in the band>

Support: <$ value>
Resistance: <$ value>
Entry Zone: <$ low - $ high>
Stop Loss: <$ value>

Risk Assessment: <1-2 sentences summarizing the risk>"""


# --- The role instruction given to the model ---
RISK_INSTRUCTIONS = """You are a professional risk manager.
Read the technical indicators below and write a concise risk report.
Base every statement on the data provided. Do not invent numbers.
For Entry Zone and Stop Loss, reason from the support, resistance,
and current price. Follow the report format exactly. Replace every
<...> placeholder with real content and keep the headers unchanged."""


def load_risk_model():
    """Load the Qwen2.5 model and tokenizer. Call this once at startup."""
    tokenizer = AutoTokenizer.from_pretrained(RISK_MODEL_PATH)

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
        RISK_MODEL_PATH, **load_kwargs
    )
    return model, tokenizer


def _build_prompt(ticker: str, user_question: str) -> str:
    """Load cached indicator data and assemble the user-turn prompt text."""
    data = load_data_bundle(ticker)
    indicators = data["indicators"]
    report_format = RISK_REPORT_TEMPLATE.format(ticker=ticker)

    return f"""User question: {user_question}

=== TECHNICAL INDICATORS ===
{indicators}

=== REPORT FORMAT (follow exactly) ===
{report_format}
"""


def make_risk_node(model, tokenizer):
    """Return the LangGraph node function, with the model bound in."""

    def risk_node(state: TradingState) -> dict:
        ticker = state["ticker"]
        question = state["user_question"]

        prompt = _build_prompt(ticker, question)

        messages = [
            {"role": "system", "content": RISK_INSTRUCTIONS},
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

        prompt_len = inputs["input_ids"].shape[-1]
        generated = outputs[0][prompt_len:]
        report = tokenizer.decode(generated, skip_special_tokens=True)

        return {"risk_report": report.strip()}

    return risk_node