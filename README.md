# Teaching Simulated Robots to Move

A hands-on exploration of reinforcement learning for simulated robotics — from a 2-joint arm touching a target to a bipedal humanoid walking on a cloud GPU. 

## Results Summary

### Part 1: Learning the Fundamentals (MacBook)

| Environment | Algorithm | Reward | Key Lesson |
|---|---|---|---|
| Reacher-v5 | PPO (500k steps) | -5.2 | Baseline — arm learns to reach |
| Reacher-v5 | SAC (500k steps) | -3.4 | Off-policy beats on-policy on small tasks |
| Reacher-v5 | PPO + smooth | -5.8 | Jerkiness penalty trades accuracy for grace |
| Reacher-v5 | PPO + sparse | ~0.3 | Sparse reward barely learns — no gradient signal |
| HalfCheetah-v5 | SAC (1M steps) | 1,985 | Cheetah ran on its back — reward hacking |
| Ant-v5 | PPO (500k steps) | ~800 | 3D locomotion is qualitatively harder |
| Humanoid-v5 | PPO naive (2M) | ~500 | Barely learned — falls over immediately |
| Humanoid-v5 | SAC naive (2M) | ~5,500 | Much better — actually walks |
| Humanoid-v5 | PPO engineered (2M) | ~800 | VecNormalize + bigger net, still worse than SAC |

### Part 2: The Bitter Lesson (A100 GPU, Isaac Gym)

| Experiment | Architecture | Iterations | Final Reward | Peak Reward |
|---|---|---|---|---|
| Baseline | MLP [256,128,64] | 5,000 | 6,334 | 6,919 |
| Bigger Network | MLP [512,256,128] | 5,000 | 9,536 | 9,780 |
| LSTM | MLP + LSTM [256 hidden] | 5,000 | 9,590 | 9,620 |

All three GPU experiments used **PPO with 4,096 parallel environments** on an NVIDIA A100 80GB. Same algorithm, same task — the only variables were network capacity and architecture.

## Findings

### 1. Reward Functions Are the Real Problem
The HalfCheetah learned to run on its back because forward velocity was rewarded with no orientation penalty. The sparse Reacher got zero signal until accidentally touching the target. On the learning curves plot, the sparse agent *appeared* to score highest — but only because it was measuring a completely different reward. **Reward magnitude is only meaningful within the same reward function.**

### 2. Agents Exploit 
The HalfCheetah's reward went from 1,200 to 2,000 (quantitative improvement) while running on its back the entire time (no qualitative change). The agent found a "good enough" exploit early and refined it rather than discovering something better.

### 3. Bitter Lesson 
PPO on a MacBook with 8 parallel environments: reward ~500. PPO on an A100 with 4,096 environments: reward 6,334. Same algorithm, 500× more compute. **Scale alone turned a failing agent into a walking one.**

But scale had limits. The baseline MLP plateaued around 6,900. A bigger network ([512,256,128]) reached 9,780, and adding LSTM memory reached 9,620. **Network capacity and architecture mattered once scale alone wasn't enough.**

### 4. Memory Helps Locomotion
The LSTM agent matched the bigger network despite having fewer total parameters. Walking is rhythmic — knowing where your leg *was* helps predict where it should *go*. Recurrent networks capture this temporal structure that feedforward MLPs miss.

### 5. Algorithm Choice Is Task-Dependent
On Reacher (easy), SAC crushed PPO. On Humanoid with massive parallelism (hard), PPO crushed SAC. SAC's replay buffer advantage disappears when you can generate 130k fresh on-policy samples per update. **The best algorithm depends on your compute budget.**

## Project Structure

```
reacher-project/
│
├── Part 1: Fundamentals (MacBook, Gymnasium + SB3)
│   ├── explore.py                      # Random actions on Reacher — the "before"
│   ├── train_ppo.py                    # PPO baseline on Reacher
│   ├── train_sac.py                    # SAC comparison
│   ├── train_ppo_smooth.py             # Jerkiness penalty (reward shaping)
│   ├── train_ppo_sparse.py             # Sparse binary reward
│   ├── enjoy.py                        # Watch trained agents
│   ├── fair_compare.py                 # Evaluate all agents on same reward
│   ├── enjoy_cheetah_progression.py    # Watch HalfCheetah checkpoints
│   ├── watch_humanoid.py               # Watch Humanoid checkpoints
│   │
│   ├── Humanoid MacBook Experiments
│   │   ├── level1_ppo_naive.py         # PPO, default everything
│   │   ├── level2_sac_naive.py         # SAC, default everything
│   │   ├── level3_ppo_engineered.py    # PPO + VecNormalize + SubprocVecEnv + [256,256]
│   │   ├── level4_sac_engineered.py    # SAC + VecNormalize + [256,256]
│   │   └── record_and_compare.py       # Plot MacBook learning curves
│   │
│   ├── learning_curves.png             # Reacher comparison plot
│   └── models/                         # Saved checkpoints (all environments)
│
├── Part 2: GPU Experiments (A100, Isaac Gym)
│   ├── isaac_gym/
│   │   ├── training_curve.csv          # Baseline MLP training curve
│   │   ├── bignet_training_curve.csv   # Bigger network training curve
│   │   ├── lstm_training_curve.csv     # LSTM training curve
│   │   ├── training_log.txt            # Baseline raw log
│   │   ├── bignet_training_log.txt     # Bigger network raw log
│   │   ├── lstm_training_log.txt       # LSTM raw log
│   │   ├── amp_training_log.txt        # AMP crash log (failed)
│   │   ├── parse_training_curves.py    # TensorBoard → CSV parser
│   │   ├── run_experiments.sh          # Experiment chain script
│   │   ├── checkpoints/                # Baseline milestone checkpoints
│   │   │   ├── humanoid_iter{1000-5000}.pth
│   │   │   ├── bignet/                 # Bigger network final checkpoint
│   │   │   └── lstm/                   # LSTM final checkpoint
│   │   └── final_videos/               # Walking videos (tracking camera)
│   │       ├── humanoid_baseline_final.mp4
│   │       ├── humanoid_bignet_final.mp4
│   │       └── humanoid_lstm_final.mp4
│   │
│   └── plot_isaac_comparison.py        # Generate GPU comparison plot
│
└── README.md
```

## Setup

### Part 1: MacBook Experiments

```bash
python -m venv reacher-env
source reacher-env/bin/activate
pip install "gymnasium[mujoco]" stable-baselines3 matplotlib
```

Run scripts in order: `explore.py` → `train_ppo.py` → `train_sac.py` → etc.

### Part 2: GPU Experiments

Requires an NVIDIA GPU (A100 recommended) with Isaac Gym Preview 4. We used RunPod ($1.49/hr for A100 SXM).

```bash
# Install Miniconda + Python 3.8 (Isaac Gym requires 3.8)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
conda create -n isaacgym python=3.8 -y
conda activate isaacgym

# Install PyTorch + Isaac Gym
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# Download Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
tar xzf IsaacGym_Preview_4_Package.tar.gz
pip install -e isaacgym/python

# Install IsaacGymEnvs
git clone https://github.com/isaac-sim/IsaacGymEnvs.git
pip install -e IsaacGymEnvs

# Train Humanoid (baseline)
cd IsaacGymEnvs/isaacgymenvs
python train.py task=Humanoid num_envs=4096 headless=True max_iterations=5000

# Bigger network
python train.py task=Humanoid num_envs=4096 headless=True max_iterations=5000 \
    "train.params.network.mlp.units=[512,256,128]"
```

## Tools & Stack

- **Gymnasium** — RL environment interface
- **MuJoCo** — Physics engine (Part 1)
- **Isaac Gym / PhysX** — GPU-accelerated physics (Part 2)
- **Stable-Baselines3** — PPO and SAC implementations (Part 1)
- **rl_games** — PPO implementation for Isaac Gym (Part 2)
- **RunPod** — Cloud GPU rental (A100 SXM, $1.49/hr)
- **Claude Code** — AI coding assistant for GPU setup and experiment management

## What I'd Do Next

- **AMP (Adversarial Motion Priors)** — train with human motion capture data. Tests whether human demonstrations beat pure scale for producing natural-looking walking. AMP is built into IsaacGymEnvs but crashed due to a `seq_len` compatibility bug — fixable with a code patch.
- **Rough terrain** — IsaacGymEnvs supports terrain randomization. Does the flat-ground policy generalize?
- **Train longer** — the LSTM reward was still climbing at 5,000 iterations. Where does it plateau at 20,000 or 50,000?
- **Sim-to-real** — deploy a trained policy on a physical robot (Unitree R1 at $4,900 or a Koch robot arm at $250).

## Cost

Total GPU spend for all experiments: approximately **$12–15** on RunPod. All MacBook experiments ran on an Apple Silicon laptop at no cost.

## Acknowledgments

This project was built interactively with Claude (Anthropic), with Claude Code handling GPU environment setup, experiment orchestration, and video recording on the remote A100 instance.
