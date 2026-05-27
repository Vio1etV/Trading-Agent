"""SFT fine-tune Qwen3-8B on the trader recommendation dataset.

Uses LoRA (QLoRA) to fine-tune a 4-bit quantized model so it fits
in 8GB VRAM. Produces an adapter that can be loaded on top of the
base Qwen3-8B model.

Usage:
    python scripts/sft_train.py
    python scripts/sft_train.py --epochs 3 --lr 2e-4 --batch-size 2
    python scripts/sft_train.py --resume  # resume from last checkpoint
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from src.config import PROJECT_ROOT, TRADER_MODEL_PATH

SFT_DATA_PATH = PROJECT_ROOT / "data" / "sft" / "trader_sft.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "models" / "trader-sft-lora"


def load_dataset_from_jsonl(path: Path) -> Dataset:
    """Load the SFT JSONL file into a HuggingFace Dataset."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} training examples from {path}")
    return Dataset.from_list(examples)


def format_and_tokenize(example: dict, tokenizer, max_length: int) -> dict:
    """Apply the chat template and tokenize a training example."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():
    parser = argparse.ArgumentParser(description="SFT fine-tune the Trader agent")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    if not SFT_DATA_PATH.exists():
        print(f"ERROR: SFT data not found at {SFT_DATA_PATH}")
        print("Run generate_sft_data.py first.")
        sys.exit(1)

    # --- 1. Load tokenizer ---
    print(f"Loading tokenizer from {TRADER_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(TRADER_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Load dataset ---
    dataset = load_dataset_from_jsonl(SFT_DATA_PATH)

    # Apply chat template + tokenize each example
    dataset = dataset.map(
        lambda ex: format_and_tokenize(ex, tokenizer, args.max_seq_len),
        remove_columns=dataset.column_names,
    )

    # Train/eval split (90/10)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)} examples, Eval: {len(eval_dataset)} examples")

    # --- 3. Load model in 4-bit ---
    print("Loading Qwen3-8B in 4-bit for LoRA training...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # extra memory savings
    )

    model = AutoModelForCausalLM.from_pretrained(
        TRADER_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )

    # Prepare for k-bit training (freeze base, enable gradient checkpointing)
    model = prepare_model_for_kbit_training(model)

    # --- 4. Configure LoRA ---
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Target the attention and MLP projection layers
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 5. Training arguments ---
    effective_batch = args.batch_size * args.grad_accum
    total_steps = (len(train_dataset) * args.epochs) // effective_batch
    print(f"\nTraining config:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x {args.grad_accum} grad accum = {effective_batch} effective")
    print(f"  Total steps: ~{total_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"  Output: {OUTPUT_DIR}\n")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",  # no wandb needed for a class project
    )

    # --- 6. Train ---
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting training...\n")
    trainer.train(resume_from_checkpoint=args.resume if args.resume else None)

    # --- 7. Save the LoRA adapter ---
    print(f"\nSaving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("\nTraining complete!")
    print(f"Adapter saved to: {OUTPUT_DIR}")
    print(f"\nTo use the fine-tuned model, load the base model + adapter:")
    print(f"  from peft import PeftModel")
    print(f"  base_model = AutoModelForCausalLM.from_pretrained('{TRADER_MODEL_PATH}', ...)")
    print(f"  model = PeftModel.from_pretrained(base_model, '{OUTPUT_DIR}')")


if __name__ == "__main__":
    main()