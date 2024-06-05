import time

from models.sac import SAC, GCN_SAC
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
    # Path to save the experiments
    path = "/experiments/"

    # Environment Name
    env_name = "l2rpn_case14_sandbox"
    # env_name = "l2rpn_idf_2023"

    # Environment paths
    train_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_train/"
    val_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_val/"

    train_ep = 1000  # Number of Episodes
    eval_ep = 100  # Number of Evaluations

    test_path = path + "test/"

    # Paths for the experiments
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

    # train_env = SubprocVecEnv([make_env(i, train_name, default_attr, reward, SEED) for i in range(6)])

    gcn_sac = GCN_SAC(test_path + sac_path, SEED)
    gcn_sac.train_and_validate(g_train_env, g_val_env, train_ep, eval_ep)

    # sac(test_path + sac_path, train_env, val_env, train_ep, eval_ep, SEED)


if __name__ == '__main__':
    main()

# Train and validate the experiments
