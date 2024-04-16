from create_env import env_multi
from observation_space import get_obs_keys
from stable_baselines3 import SAC

SEED = 1234

# Model Path
model_path = "/home/treeman/school/dissertation/src/grl/models/exp1/m_sac/"


env_name = "/home/treeman/school/dissertation/src/grl/data_grid2op/l2rpn_case14_sandbox_train/"
# Create the environment
default_attr = get_obs_keys(False, False, False, False, False)
env = env_multi(env_name, default_attr, SEED)
# Environment Path
env_name = "/home/treeman/school/dissertation/src/grl/data_grid2op/l2rpn_case14_sandbox_val/"
env = env_multi(env_name, default_attr, SEED)



# Load the model
model = SAC.load(model_path + "model", env=env, seed=SEED)
d_env = model.get_env()

# total number of episode
total_episode = len(env.init_env.chronics_handler.subpaths)

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
        env.init_env.chronics_handler.shuffle()
    ###################################

    obs = d_env.reset()
    # now play the episode as usual
    while True:
        # fig = env.render()  # render the environment
        # plt.pause(update_interval)  # Show the plot after each iteration
        action, _states = model.predict(obs, deterministic=1)
        obs, reward, terminated, info = d_env.step(action)
        # print("Action: " + str(obs["action"]))
        # print("Load power: " + str(obs["load_p"]))
        # print("Reward: " + str(reward))

        total_reward += reward
        if terminated:
            # in this case the episode is over
            break

print("Total reward: " + str(total_reward))
