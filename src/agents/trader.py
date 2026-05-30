"""Trader Agent: reads analyst_report + risk_report -> writes final recommendation.

Backed by a local Qwen3-8B model (will be SFT fine-tuned later).
Same three-piece shape as the other agents: load model, build prompt, make node.
"""

from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import TRADER_MODEL_PATH, TEMPERATURE, USE_4BIT, MODELS_DIR

# Trader gets a larger token budget because Qwen3 uses some tokens for
# internal <think> reasoning before producing the visible answer.
TRADER_MAX_TOKENS = 2048

# Path to the LoRA adapter (created by sft_train.py)
LORA_ADAPTER_PATH = MODELS_DIR / "trader-sft-lora"
from src.graph.state import TradingState


# --- The exact output format the model must follow ---
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


# --- The role instruction given to the model ---
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


def load_trader_model():
    """Load the Qwen3-8B model and tokenizer.

    If a LoRA adapter exists at models/trader-sft-lora/, it is loaded
    on top of the base model automatically.
    """
    tokenizer = AutoTokenizer.from_pretrained(TRADER_MODEL_PATH)

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
        TRADER_MODEL_PATH, **load_kwargs
    )

    # Load LoRA adapter if it exists (after SFT training)
    if LORA_ADAPTER_PATH.exists() and (LORA_ADAPTER_PATH / "adapter_config.json").exists():
        from peft import PeftModel
        print(f"  Loading LoRA adapter from {LORA_ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(model, str(LORA_ADAPTER_PATH))
        model = model.merge_and_unload()  # merge for faster inference
        print("  LoRA adapter merged.")
    else:
        print("  No LoRA adapter found — using base Qwen3-8B.")

    return model, tokenizer


def _build_prompt(
    ticker: str,
    user_question: str,
    analyst_report: str,
    risk_report: str,
) -> str:
    """Assemble the prompt from both upstream reports."""
    report_format = TRADER_REPORT_TEMPLATE.format(ticker=ticker)

    return f"""User question: {user_question}

=== ANALYST REPORT ===
{analyst_report}

=== RISK MANAGER REPORT ===
{risk_report}

=== YOUR OUTPUT FORMAT (follow exactly) ===
{report_format}
"""


def make_trader_node(model, tokenizer):
    """Return the LangGraph node function, with the model bound in."""

    def trader_node(state: TradingState) -> dict:
        ticker = state["ticker"]
        question = state["user_question"]
        analyst_report = state["analyst_report"]
        risk_report = state["risk_report"]

        prompt = _build_prompt(ticker, question, analyst_report, risk_report)

        # Qwen3 supports a system role, like Qwen2.5.
        messages = [
            {"role": "system", "content": TRADER_INSTRUCTIONS},
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
            max_new_tokens=TRADER_MAX_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
        )

        prompt_len = inputs["input_ids"].shape[-1]
        generated = outputs[0][prompt_len:]
        recommendation = tokenizer.decode(generated, skip_special_tokens=True)

        # Qwen3 wraps internal reasoning in <think>...</think> tags.
        # Strip it so the output is just the clean recommendation.
        recommendation = re.sub(
            r"<think>.*?</think>", "", recommendation, flags=re.DOTALL
        ).strip()

        return {"trader_recommendation": recommendation}

    return trader_node