import time

from baselines import gcn_sac
from environment.create_env import env_graph
from environment.create_env import env_test_val
from environment.observation_space import get_obs_keys
from environment.reward.res_reward import DynamicEconomicReward


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
    # env_name = "l2rpn_idf_2023"

    # Environment paths
    train_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_train/"
    val_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_val/"

    train_ep = 1000  # Number of Episodes
    eval_ep = 100  # Number of Evaluations

    test_path = path + "grl1/"

    # Paths for the models
    sac_path = "sac/"
    gcn_sac_path = "gcn_sac/"

    # Split the dataset
    # split_dataset(env_name, SEED)

    # Get the observation attributes
    default_attr = get_obs_keys(False, False, False, False, False)

    reward = DynamicEconomicReward(res_penalty=0.4)

    SEED = 234523455
    # SEED = seed(test_path)

    g_train_env, g_val_env = env_test_val(train_name, val_name, default_attr, reward, SEED, True)

    gcn_sac(test_path + gcn_sac_path, g_train_env, g_val_env, train_ep, eval_ep, SEED)


if __name__ == '__main__':
    main()