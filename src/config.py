"""Central configuration: model paths and generation settings.

Keeping paths in one place means agents never hard-code a directory.
If a model moves, you change it here only.
"""

from __future__ import annotations

from pathlib import Path

# Project root = two levels up from this file (src/config.py -> trading-agent/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

# --- Model paths (local HuggingFace snapshots) ---
ANALYST_MODEL_PATH = str(MODELS_DIR / "gemma-2-9b-it")
RISK_MODEL_PATH = str(MODELS_DIR / "qwen2.5-7b-instruct")
TRADER_MODEL_PATH = str(MODELS_DIR / "qwen3-8b")

# --- Generation settings (shared defaults) ---
MAX_NEW_TOKENS = 768
TEMPERATURE = 0.7
