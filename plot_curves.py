import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

experiments = [
    ("PPO Baseline",    "logs/ppo/evaluations.npz"),
    ("SAC",             "logs/sac/evaluations.npz"),
    ("PPO + Smooth",    "logs/ppo_smooth/evaluations.npz"),
    ("PPO + Sparse",    "logs/ppo_sparse/evaluations.npz"),
]

for name, path in experiments:
    try:
        data = np.load(path)
        steps = data["timesteps"]
        means = data["results"].mean(axis=1)
        stds  = data["results"].std(axis=1)
        ax.plot(steps, means, label=name)
        ax.fill_between(steps, means - stds, means + stds, alpha=0.2)
    except FileNotFoundError:
        print(f"Skipping {name} — no log found at {path}")

ax.set_xlabel("Training Steps")
ax.set_ylabel("Mean Eval Reward")
ax.set_title("Reacher-v5: Learning Curves Comparison")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)
plt.show()
print("Saved to learning_curves.png")