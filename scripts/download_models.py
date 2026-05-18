"""Download model weights from HuggingFace into models/.

Usage:
    python scripts/download_models.py --model all
    python scripts/download_models.py --model qwen3
    python scripts/download_models.py --model gemma
    python scripts/download_models.py --model llama

Gemma and Llama are gated models — you must:
1. Accept the license on huggingface.co for each model
2. Set HF_TOKEN in your environment:
       export HF_TOKEN=hf_your_token_here
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

MODELS = {
    "qwen3": {
        "repo_id": "Qwen/Qwen3-8B",
        "local_dir": MODELS_DIR / "qwen3-8b",
        "gated": False,
        "used_by": "Trader (SFT base)",
    },
    "gemma": {
        "repo_id": "google/gemma-2-9b-it",
        "local_dir": MODELS_DIR / "gemma-2-9b-it",
        "gated": True,
        "used_by": "Analyst Agent",
    },
    "llama": {
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "local_dir": MODELS_DIR / "llama-3.1-8b-instruct",
        "gated": True,
        "used_by": "Risk Manager Agent",
    },
    "qwen25": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "local_dir": MODELS_DIR / "qwen2.5-7b-instruct",
        "gated": False,
        "used_by": "Risk Manager Agent",
    },
}

# Models the project actually uses today. `--model all` downloads these.
# Llama is excluded: it is still pending access review and is not wired
# into config.py. Download it explicitly with `--model llama` once granted.
DEFAULT_MODELS = ["qwen3", "gemma", "qwen25"]


def download(name: str, config: dict, token: str | None) -> None:
    if config["gated"] and not token:
        print(f"[SKIP] {name}: gated model — set HF_TOKEN first.")
        print(f"       Accept license at: https://huggingface.co/{config['repo_id']}")
        return

    local_dir = config["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{name.upper()}] Downloading {config['repo_id']} → {local_dir}")
    print(f"  Used by: {config['used_by']}")

    snapshot_download(
        repo_id=config["repo_id"],
        local_dir=str(local_dir),
        token=token,
        ignore_patterns=["*.gguf", "original/*"],
    )
    print(f"[{name.upper()}] Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["all", "qwen3", "gemma", "llama", "qwen25"],
        default="all",
        help="Which model to download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")

    if not token:
        print("Warning: HF_TOKEN not set. Gated models (Gemma, Llama) will be skipped.")
        print("Set it with: export HF_TOKEN=hf_your_token_here\n")

    if args.model == "all":
        targets = {name: MODELS[name] for name in DEFAULT_MODELS}
    else:
        targets = {args.model: MODELS[args.model]}

    for name, config in targets.items():
        download(name, config, token)

    print("\nAll done. Models stored in:", MODELS_DIR)


if __name__ == "__main__":
    main()
