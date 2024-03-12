import matplotlib
import matplotlib.pyplot as plt
import grid2op
from grid2op.Agent import RandomAgent
from lightsim2grid import LightSimBackend
from grid2op.gym_compat import GymnasiumEnv, BoxGymnasiumActSpace
from grid2op.Reward import EconomicReward
from grid2op.Opponent import BaseOpponent, NeverAttackBudget
from grid2op.Action import DontAct
from stable_baselines3 import SAC

# Create the environment
env_name = "/home/treeman/school/dissertation/src/grl/data_grid2op/l2rpn_case14_sandbox_test/"
backend_class = LightSimBackend
grid_env = grid2op.make(env_name, 
                        backend=backend_class(), 
                        reward_class=EconomicReward, 
                        opponent_attack_cooldown=999999,
                        opponent_attack_duration=0,
                        opponent_budget_per_ts=0,
                        opponent_init_budget=0,
                        opponent_action_class=DontAct,
                        opponent_class=BaseOpponent,
                        opponent_budget_class=NeverAttackBudget)

env = GymnasiumEnv(grid_env, render_mode="rgb_array")
env.action_space = BoxGymnasiumActSpace(grid_env.action_space)

SEED = 0
grid_env.seed(SEED)  # for reproducible experiments
env.action_space.seed(SEED)  # for reproducible experiments

from data_collect import CollectCallback
eval_callback = CollectCallback(save_path=path + "data/train/")
model = SAC.load("sac_ded", env=env)

env = model.get_env()
env.render_mode = "rgb_array"



# total number of episode
total_episode = len(grid_env.chronics_handler.subpaths)

# Initialize variables
# agent = RandomAgent(env.action_space)
episode_count = 1  # i want to make lots of episode
reward = 0
total_reward = 0
update_interval = 0.1                        # Update interval

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

        print("Load power: " + str(obs["load_p"]))
        print("Reward: " + str(reward))

        total_reward += reward
        if terminated:
            # in this case the episode is over
            break

print("Total reward: " + str(total_reward))
