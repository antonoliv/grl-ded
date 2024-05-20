from create_env import env_multi, env_box, env_test_val, split_dataset
from observation_space import get_obs_keys
from baselines import sac, gcn_sac
import time
from reward.default_reward import EconomicReward
from reward.res_reward import DynamicEconomicReward

import torch

from stable_baselines3.common.vec_env import SubprocVecEnv
from create_env import env_graph

def seed(path):
    # SEED = 123456789
    SEED = int(time.time())

    # Save the seed
    with open(path + 'seed.txt', 'w') as file:
        file.write(f"Seed: {SEED}\n")
    
    return SEED


def make_env(rank, env_name, default_attr, reward, seed):
    def _init():
        env = env_graph(env_name, default_attr, reward, seed)
        env.reset(seed=seed + rank)
        return env

    return _init

def main():
    # Path to save the models
    path = "/home/treeman/school/dissertation/src/grl/models/"

    # Environment Name
    env_name = "l2rpn_case14_sandbox"

    # Environment paths
    train_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_train/"
    val_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_val/"

    train_ep = 500  # Number of Episodes
    eval_ep = 25  # Number of Evaluations

    test_path = path + "grl1/"

    # Paths for the models
    sac_path = "sac/"
    gcn_sac_path = "gcn_sac/"

    # Split the dataset
    # split_dataset(env_name, SEED)

    # Get the observation attributes
    default_attr = get_obs_keys(False, False, False, False, False)

    reward = DynamicEconomicReward(res_penalty=0.4)

    SEED = 123456789
    # SEED = seed(test_path)

    train_env, val_env = env_test_val(train_name, val_name, default_attr, reward, SEED, False)
    g_train_env, g_val_env = env_test_val(train_name, val_name, default_attr, reward, SEED, True)




    # env = SubprocVecEnv([make_env(i, train_name, default_attr, reward, SEED) for i in range(6)])

    gcn_sac(test_path + gcn_sac_path, train_env, g_val_env, train_ep, eval_ep, SEED)
    sac(test_path + sac_path, train_env, val_env, train_ep, eval_ep, SEED)


if __name__ == '__main__':
    main()


# Train and validate the models
