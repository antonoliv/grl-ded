from create_env import create_environment
from stable_baselines3 import SAC

SEED = 1234

# Model Path
model_path = "/home/treeman/school/dissertation/src/grl/models/sac/"

# Environment Path
env_name = "/home/treeman/school/dissertation/src/grl/data_grid2op/l2rpn_case14_sandbox_test/"

# Create the environment
test = "/home/treeman/school/dissertation/src/grl/data_grid2op/l2rpn_case14_sandbox_train/"
test = create_environment(test, SEED)
envs = create_environment(env_name, SEED)
env = envs[0]
grid_env = envs[1]

# Load the model
model = SAC.load(model_path + "model", env=env, seed=SEED)
env = model.get_env()

# total number of episode
total_episode = len(grid_env.chronics_handler.subpaths)

# Initialize variables
# agent = RandomAgent(env.action_space)
episode_count = 1  # i want to make lots of episode
reward = 0
total_reward = 0
update_interval = 0.1  # Update interval

# and now the loop starts
for i in range(episode_count):

    ###################################
    if i % total_episode == 0:
        # I shuffle each time i need to
        grid_env.chronics_handler.shuffle()
    ###################################

    obs = env.reset()
    # now play the episode as usual
    while True:
        # fig = env.render()  # render the environment
        # plt.pause(update_interval)  # Show the plot after each iteration
        action, _states = model.predict(obs, deterministic=1)
        obs, reward, terminated, info = env.step(action)
        print("Action: " + str(obs[0]))
        # print("Load power: " + str(obs["load_p"]))
        # print("Reward: " + str(reward))

        total_reward += reward
        if terminated:
            # in this case the episode is over
            break

print("Total reward: " + str(total_reward))
