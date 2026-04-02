import gymnasium as gym

env = gym.make("Reacher-v5", render_mode="human")
obs, info = env.reset()
print("Observation:", obs)
print("  - obs[0:4]  → joint angles as cos/sin pairs")
print("  - obs[4:6]  → target (x, y) position")
print("  - obs[6:8]  → joint angular velocities")
print("  - obs[8:10] → vector from fingertip to target")

# Run 200 steps of random actions so you can WATCH the arm flail
for _ in range(200):
    action = env.action_space.sample()  # random torques
    obs, reward, terminated, truncated, info = env.step(action)

env.close()