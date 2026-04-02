"""
Generate the final comparison plot: Isaac Gym GPU experiments.
Plots training curves for baseline MLP, bigger network, and LSTM.
"""
import csv
import matplotlib.pyplot as plt
import os

def load_curve(csv_path):
    """Load training curve CSV (iteration, mean_reward)."""
    iterations = []
    rewards = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                it = int(row.get('iteration') or row.get('step') or row.get('Step'))
                rew = float(row.get('mean_reward') or row.get('rewards/step') or row.get('reward'))
                iterations.append(it)
                rewards.append(rew)
            except (ValueError, TypeError):
                continue
    return iterations, rewards

# Try to find the CSV files
script_dir = os.path.dirname(os.path.abspath(__file__))
isaac_dir = os.path.join(script_dir, 'isaac_gym')

experiments = [
    ("PPO Baseline [256,128,64]", os.path.join(isaac_dir, "training_curve.csv")),
    ("PPO Bigger Net [512,256,128]", os.path.join(isaac_dir, "bignet_training_curve.csv")),
    ("PPO + LSTM [256 hidden]", os.path.join(isaac_dir, "lstm_training_curve.csv")),
]

fig, ax = plt.subplots(figsize=(12, 7))

colors = ['#2196F3', '#FF9800', '#4CAF50']

for (name, path), color in zip(experiments, colors):
    if os.path.exists(path):
        iterations, rewards = load_curve(path)
        if iterations:
            ax.plot(iterations, rewards, label=name, linewidth=2, color=color, alpha=0.8)
            print(f"{name}: {len(iterations)} points, final reward = {rewards[-1]:.1f}, peak = {max(rewards):.1f}")
    else:
        print(f"Skipping {name} — no CSV at {path}")

ax.set_xlabel("Training Iteration", fontsize=13)
ax.set_ylabel("Mean Episode Reward", fontsize=13)
ax.set_title("Humanoid-v5 on A100: What Beats Scale?", fontsize=15, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()

output_path = os.path.join(script_dir, "isaac_gym_comparison.png")
plt.savefig(output_path, dpi=150)
print(f"\nSaved to {output_path}")
plt.show()
