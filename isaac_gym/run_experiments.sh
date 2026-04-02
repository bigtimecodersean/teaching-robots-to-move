#!/bin/bash
export PATH="/opt/conda/bin:$PATH"
source /opt/conda/etc/profile.d/conda.sh
conda activate isaacgym
export LD_LIBRARY_PATH=/opt/conda/envs/isaacgym/lib:$LD_LIBRARY_PATH
cd /workspace/IsaacGymEnvs/isaacgymenvs

echo "=== EXPERIMENT 1: BIGGER NETWORK ==="
echo "Started at: $(date)"
python train.py task=Humanoid num_envs=4096 headless=True max_iterations=5000 \
  "train.params.network.mlp.units=[512,256,128]" \
  2>&1 | tee /workspace/humanoid_results/bignet_training_log.txt
echo "Experiment 1 finished at: $(date)"

echo "=== EXPERIMENT 2: AMP ==="
echo "Started at: $(date)"
python train.py task=HumanoidAMP num_envs=4096 headless=True max_iterations=5000 \
  2>&1 | tee /workspace/humanoid_results/amp_training_log.txt
echo "Experiment 2 finished at: $(date)"

echo "=== EXPERIMENT 3: LSTM ==="
echo "Started at: $(date)"
python train.py task=Humanoid num_envs=4096 headless=True max_iterations=5000 \
  "train.params.network.name=a2c_lstm" \
  2>&1 | tee /workspace/humanoid_results/lstm_training_log.txt
echo "Experiment 3 finished at: $(date)"

echo "=== ALL EXPERIMENTS COMPLETE ==="
