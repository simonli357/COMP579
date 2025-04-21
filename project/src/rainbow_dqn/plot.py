import matplotlib.pyplot as plt
import numpy as np
import pathlib
import json

def smooth(data, window=20):
    if len(data) < window:
        return data  # Skip smoothing if too short
    return np.convolve(data, np.ones(window) / window, mode='valid')

current_dir = pathlib.Path(__file__).parent
exp_root = current_dir / "experiments"

all_rewards = []
labels = []

# First, load and smooth all rewards
for exp_dir in exp_root.iterdir():
    rewards = np.load(exp_dir / "rewards.npy")
    cfg = json.loads((exp_dir / "config.json").read_text())
    smoothed_rewards = smooth(rewards, window=5)
    all_rewards.append(smoothed_rewards)
    labels.append(cfg["name"])

# Find shortest length
min_len = min(len(r) for r in all_rewards)

# Truncate all to the same length
truncated_rewards = [r[:min_len] for r in all_rewards]

# Plot
plt.figure(figsize=(10, 4))
for r, label in zip(truncated_rewards, labels):
    plt.plot(r, label=label)

plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Rainbow ablation on Ms Pac‑Man")
plt.legend()
plt.tight_layout()
plt.savefig(current_dir / "combined_learning_curves_truncated.png")
plt.show()
