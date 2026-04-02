#!/usr/bin/env python3
"""Parse IsaacGymEnvs training logs into CSV files using TensorBoard events."""

import csv
import os
import glob
import sys

def find_summaries_dir(run_dir):
    """Find the summaries directory with TensorBoard events."""
    summaries = os.path.join(run_dir, "summaries")
    if os.path.isdir(summaries):
        return summaries
    # Check for events directly in run_dir
    events = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if events:
        return run_dir
    return None


def parse_tensorboard(summaries_dir, csv_path, label):
    """Parse TensorBoard events into CSV."""
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(summaries_dir)
    ea.Reload()

    tags = ea.Tags().get('scalars', [])
    if not tags:
        print(f"  No scalar data found in {summaries_dir}")
        return False

    # Key metrics to extract
    desired_tags = {
        'rewards/iter': 'reward',
        'rewards/step': 'reward_per_step',
        'losses/a_loss': 'actor_loss',
        'losses/c_loss': 'critic_loss',
        'losses/entropy': 'entropy',
        'info/last_lr': 'learning_rate',
        'info/kl': 'kl_divergence',
        'performance/step_inference_rl_update_fps': 'fps_total',
    }

    # Collect data by step
    step_data = {}
    for tag in tags:
        col_name = desired_tags.get(tag, tag.replace('/', '_'))
        if tag not in desired_tags:
            continue
        for event in ea.Scalars(tag):
            step = event.step
            if step not in step_data:
                step_data[step] = {'step': step}
            step_data[step][col_name] = event.value

    if not step_data:
        print(f"  No data extracted")
        return False

    # Determine columns
    all_cols = set()
    for d in step_data.values():
        all_cols.update(d.keys())

    col_order = ['step', 'reward', 'reward_per_step', 'actor_loss', 'critic_loss',
                 'entropy', 'learning_rate', 'kl_divergence', 'fps_total']
    columns = [c for c in col_order if c in all_cols]
    columns += sorted(all_cols - set(columns))

    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for step in sorted(step_data.keys()):
            writer.writerow(step_data[step])

    n_rows = len(step_data)
    rewards = [d['reward'] for d in step_data.values() if 'reward' in d]

    print(f"  Wrote {n_rows} rows to {csv_path}")
    if rewards:
        print(f"  Reward: {rewards[0]:.1f} -> {rewards[-1]:.1f} (max: {max(rewards):.1f})")

    return True


def parse_from_log(log_path, csv_path, label):
    """Fallback: parse training log text if TensorBoard not available."""
    import re

    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        print(f"  Log file empty or missing: {log_path}")
        return False

    with open(log_path, 'r') as f:
        content = f.read()

    rows = []
    # Parse fps/epoch lines
    epoch_pattern = re.compile(
        r'fps step:\s*([\d.]+)\s+fps step and policy inference:\s*([\d.]+)\s+'
        r'fps total:\s*([\d.]+)\s+epoch:\s*(\d+)/(\d+)\s+frames:\s*(\d+)'
    )

    for m in epoch_pattern.finditer(content):
        rows.append({
            'epoch': int(m.group(4)),
            'frames': int(m.group(6)),
            'fps_step': float(m.group(1)),
            'fps_total': float(m.group(3)),
        })

    # Parse best rewards from checkpoints
    reward_pattern = re.compile(r'saving next best rewards:\s+\[([\d.]+)\]')
    best_rewards = [float(m.group(1)) for m in reward_pattern.finditer(content)]

    # Parse rewards from checkpoint filenames
    ckpt_pattern = re.compile(r'last_\w+_ep_(\d+)_rew_([\d.]+)\.pth')
    ckpt_rewards = {int(m.group(1)): float(m.group(2)) for m in ckpt_pattern.finditer(content)}

    # Add checkpoint rewards to matching epochs
    for row in rows:
        if row['epoch'] in ckpt_rewards:
            row['reward'] = ckpt_rewards[row['epoch']]

    if not rows:
        print(f"  No training data found in {log_path}")
        return False

    columns = ['epoch', 'frames', 'fps_step', 'fps_total']
    if any('reward' in r for r in rows):
        columns.append('reward')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"  Wrote {len(rows)} rows to {csv_path} (from log text)")
    if best_rewards:
        print(f"  Best rewards seen: {best_rewards[0]:.1f} -> {best_rewards[-1]:.1f}")
    return True


def main():
    results_dir = "/workspace/humanoid_results"
    runs_dir = "/workspace/IsaacGymEnvs/isaacgymenvs/runs"

    experiments = [
        ("bignet_training_log.txt", "bignet_training_curve.csv", "BIGGER NETWORK", "Humanoid"),
        ("amp_training_log.txt", "amp_training_curve.csv", "AMP", "HumanoidAMP"),
        ("lstm_training_log.txt", "lstm_training_curve.csv", "LSTM", "Humanoid"),
    ]

    for log_name, csv_name, label, task_prefix in experiments:
        log_path = os.path.join(results_dir, log_name)
        csv_path = os.path.join(results_dir, csv_name)
        print(f"\n=== {label} ===")

        # Try to find the run directory from the log
        run_path = None
        if os.path.exists(log_path):
            import re
            with open(log_path, 'r') as f:
                for line in f:
                    m = re.search(r"runs/(\S+)/nn/", line)
                    if m:
                        candidate = os.path.join(runs_dir, m.group(1))
                        if os.path.isdir(candidate):
                            run_path = candidate
                            break

        # Try TensorBoard first
        parsed = False
        if run_path:
            sdir = find_summaries_dir(run_path)
            if sdir:
                print(f"  Using TensorBoard events from: {run_path}")
                try:
                    parsed = parse_tensorboard(sdir, csv_path, label)
                except Exception as e:
                    print(f"  TensorBoard parsing failed: {e}")

        # Fallback to log parsing
        if not parsed:
            print(f"  Falling back to log text parsing")
            parse_from_log(log_path, csv_path, label)


if __name__ == "__main__":
    main()
