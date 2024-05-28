from baselines import sac
from environment.create_env import env_test_val
from environment.observation_space import get_obs_keys
from environment.reward.default_reward import EconomicReward
from environment.reward.res_reward import DynamicEconomicReward


def seed(path):
    SEED = 123456789
    # SEED = int(time.time())

    # Save the seed
    with open(path + 'seed.txt', 'w') as file:
        file.write(f"Seed: {SEED}\n")

    return SEED


# Path to save the models
path = "/home/treeman/school/dissertation/src/grl/models/"

# Environment Name
env_name = "l2rpn_case14_sandbox"

# Environment paths
train_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_train/"
val_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_val/"

train_ep = 10  # Number of Episodes
eval_ep = 100  # Number of Evaluations

test_path = path + "test/"

# Paths for the models
b_sac_path = "b_sac/"
m_sac_path = "m_sac/"
mt_sac_path = "mt_sac/"
mdt_sac_path = "mdt_sac/"

# Split the dataset
# split_dataset(env_name, SEED)

# Get the observation attributes
default_attr = get_obs_keys(False, False, False, False, False)
delta_time_attr = get_obs_keys(False, False, True, False, False)
time_attr = get_obs_keys(False, False, True, True, False)

default_reward = EconomicReward
res_reward = DynamicEconomicReward

# SEED = 123456789
SEED = seed(test_path)

# MultInput Policy + Default Observation Space
m_train_env, m_val_env = env_test_val(train_name, val_name, default_attr, default_reward, SEED, False)

# MLP Policy + Delta Time Observation Space
mdt_train_env, mdt_val_env = env_test_val(train_name, val_name, delta_time_attr, default_reward, SEED, False)

# MultiInput Policy + Time Observation Space
mt_train_env, mt_val_env = env_test_val(train_name, val_name, time_attr, default_reward, SEED, False)

# MLP Policy + Default Observation Space
b_train_env, b_val_env = env_test_val(train_name, val_name, default_attr, default_reward, SEED, True)

# Train and validate the models
sac(test_path + m_sac_path, m_train_env, m_val_env, train_ep, eval_ep, SEED)
sac(test_path + b_sac_path, b_train_env, b_val_env, train_ep, eval_ep, SEED)
sac(test_path + mt_sac_path, mt_train_env, mt_val_env, train_ep, eval_ep, SEED)
sac(test_path + mdt_sac_path, mdt_train_env, mdt_val_env, train_ep, eval_ep, SEED)
