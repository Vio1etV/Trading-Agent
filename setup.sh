#!/usr/bin/env bash
# One-time setup after cloning the repo.
#
# Installs Python dependencies and downloads model weights (~47GB).
# Run it from the project root, inside your activated conda env.
#
# The gated Gemma model needs a HuggingFace token. Export it first:
#     export HF_TOKEN=hf_your_token_here
#     bash setup.sh

set -e  # stop on first error

echo "=== Step 1/2: Installing Python dependencies ==="
pip install -r requirements.txt

echo ""
echo "=== Step 2/2: Downloading model weights (~47GB) ==="
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN is not set — the gated Gemma model will be skipped."
    echo "         Export it and re-run this script:  export HF_TOKEN=hf_..."
fi
python scripts/download_models.py --model all

echo ""
echo "=== Setup complete ==="
echo "Next, on a GPU node, run the agent test:"
echo "    python scripts/test_agents.py --ticker NVDA"
