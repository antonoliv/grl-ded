import time

from models.ppo import PPO, SAGE_PPO
from models.sac import SAGE_SAC
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


# Path to save the experiments
path = "/home/treeman/school/dissertation/src/grl/experiments/"

# Environment Name
env_name = "l2rpn_case14_sandbox"
# env_name = "l2rpn_idf_2023"

# Environment paths
train_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_train/"
val_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_val/"

train_ep = 1000  # Number of Episodes
eval_ep = 10  # Number of Evaluations

test_path = path + "grl3/"

path = "gcn_sac/"

SEED = 234523455

gcn_a2c = SAGE_SAC(test_path + path, SEED)
gcn_a2c.train_and_validate(train_name, val_name, train_ep, eval_ep)
