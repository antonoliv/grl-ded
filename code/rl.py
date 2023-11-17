import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)


for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)
   print("Action " + str(action) + " / Observation " + str(observation) + " / Reward " + str(reward))

   if terminated or truncated:
    env.close()

