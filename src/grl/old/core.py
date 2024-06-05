import os
import time

import numpy as np
import pandas as pd

from callbacks.data_collect import CollectCallback
from callbacks.train_episode import EpisodeCallback
from stable_baselines3.common.callbacks import CallbackList


def train(path: str, model, episodes: int):
    # Path to save the model

    max_episodes = episodes
    avg_episode_length = 1500
    total_timesteps = max_episodes * avg_episode_length

    # Create a callbacks
    eval_callback = CollectCallback(save_path=path + "data/train/", model=model, verbose=1,
                                    max_episodes=episodes)  # Saves training data
    episode_callback = EpisodeCallback(max_episodes=max_episodes, verbose=1)  # Stops training after x episodes
    callback_lst = CallbackList([eval_callback, episode_callback])

    # Train the agent and display a progress bar
    model.learn(total_timesteps=total_timesteps, progress_bar=False, callback=eval_callback)

    model.save(path + "model")


def validate(path, model, grid_env, eval_episodes: int):
    path += "data/val/"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.

    env = model.get_env()

    # Episode Metrics
    cost = 0
    res_waste = 0
    avg_cost = []
    avg_res_waste = []

    n_res = 0
    for i in range(0, grid_env.n_gen):
        if grid_env.gen_renewable[i] == 1:
            n_res += 1

    start_time = time.time()
    total_steps = 0
    acc_reward = 0
    length = 0
    acc_rewards = []
    lengths = []
    total_episode = len(grid_env.chronics_handler.subpaths)

    # Initialize variables
    # agent = RandomAgent(env.action_space)
    episode_count = eval_episodes  # i want to make lots of episode
    update_interval = 0.1  # Update interval

    # and now the loop starts
    for i in range(episode_count):

        obs = env.reset()

        # now play the episode as usual
        while True:
            # fig = env.render()  # render the environment
            # plt.pause(update_interval)  # Show the plot after each iteration
            action, _states = model.predict(obs, deterministic=1)
            obs, reward, terminated, info = env.step(action)

            acc_reward += reward[0]
            length += 1
            total_steps += 1

            cost += (obs['gen_p'] * grid_env.gen_cost_per_MW).sum() * grid_env.delta_time_seconds / 3600.0

            for j in range(0, grid_env.n_gen):
                if obs['gen_p_before_curtail'][0][j] != 0:
                    res_waste += (obs['gen_p_before_curtail'][0][j] - obs['gen_p'][0][j]).sum()

            if terminated:
                # in this case the episode is over
                acc_rewards.append(acc_reward)
                avg_cost.append(cost * 288 / length)
                avg_res_waste.append(res_waste * 288 / length)
                lengths.append(length)
                acc_reward = 0
                length = 0
                cost = 0
                res_waste = 0
                break

    env.close()

    data = {
        'Time Elapsed': [time.time() - start_time],
        'Num Steps': [total_steps],
        'Num Episodes': [eval_episodes],
        'Avg Steps per Episode': [np.mean(lengths) if eval_episodes > 0 else 0],
    }

    episode = {
        'Episode': range(1, eval_episodes + 1),
        'Accumulative Reward': acc_rewards,
        'Length': lengths,
        'Avg Cost': avg_cost,
        'Avg Renewables Wasted': avg_res_waste,
    }

    df_info = pd.DataFrame(data)
    episode_zip = list(zip(episode["Episode"], episode["Accumulative Reward"], episode["Length"], episode["Avg Cost"],
                           episode["Avg Renewables Wasted"]))
    df_episode = pd.DataFrame(episode_zip, columns=list(episode.keys()))
    # Save to CSV
    df_info.to_csv(path + "info.csv", index=False)
    df_episode.to_csv(path + "episode.csv", index=False)

# from stable_baselines3 import SAC
# from grid_obs import get_obs_keys
# from create_env import env_test_val
#
# SEED = 123456789
# env_name = "l2rpn_case14_sandbox"
#
# path = "/home/treeman/school/dissertation/src/grl/experiments/exp1/m_sac/"
#
# train_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_train/"
# val_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_val/"
# default_attr = get_obs_keys(False, False, False, False)
#
# m_train_env, val_env = env_test_val(train_name, val_name, default_attr, SEED, False)
#
# train(path, sac_model, 1000, SEED)
# sac_model = SAC.load(path + "model", env=val_env, seed=SEED)
#
#
# validate(sac_model, val_env.init_env, 1)
