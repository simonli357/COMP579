import matplotlib.pyplot as plt, numpy as np, pathlib, json

current_dir = pathlib.Path(__file__).parent
exp_root = current_dir / "experiments"
plt.figure(figsize=(10, 4))
for exp_dir in exp_root.iterdir():
    rewards = np.load(exp_dir / "rewards.npy")
    cfg = json.loads((exp_dir / "config.json").read_text())
    plt.plot(rewards, label=cfg["name"])
plt.xlabel("Episode"); plt.ylabel("Score")
plt.title("Rainbow ablation on Ms Pac‑Man")
plt.legend(); plt.tight_layout()
plt.savefig(current_dir / "combined_learning_curves.png")
plt.show()