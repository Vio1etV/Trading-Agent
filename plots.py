import json
import matplotlib.pyplot as plt
from pathlib import Path

# Find the trainer state file
candidates = sorted(Path("models/trader-sft-lora").glob("checkpoint-*/trainer_state.json"))
if not candidates:
    print("No trainer_state.json found in checkpoints!")
    exit(1)

state_path = candidates[-1]  # use the latest checkpoint
print(f"Reading {state_path}")

with open(state_path, encoding="utf-8") as f:
    data = json.load(f)

steps = [x["step"] for x in data["log_history"] if "loss" in x]
losses = [x["loss"] for x in data["log_history"] if "loss" in x]
eval_steps = [x["step"] for x in data["log_history"] if "eval_loss" in x]
eval_losses = [x["eval_loss"] for x in data["log_history"] if "eval_loss" in x]

plt.figure(figsize=(8, 5))
plt.plot(steps, losses, "o-", label="Train Loss")
plt.plot(eval_steps, eval_losses, "s-", label="Eval Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("SFT Training Loss Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("training_curve.png", dpi=150, bbox_inches="tight")
print("Saved training_curve.png")