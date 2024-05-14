from create_env import env_multi, env_box, env_test_val, split_dataset
from observation_space import get_obs_keys
from baselines import sac, ppo, ddpg
import time
from reward.default_reward import EconomicReward
from reward.res_reward import DynamicEconomicReward

def seed(path):
    # SEED = 123456789
    SEED = int(time.time())

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


train_ep = 1500                     # Number of Episodes
eval_ep = 100                   # Number of Evaluations


test_path = path + "exp_reward/"

# Paths for the models
r_sac_path = "r_sac/"
rd1_sac_path = "rd1_sac/"
rd2_sac_path = "rd2_sac/"
rd3_sac_path = "rd3_sac/"
rd4_sac_path = "rd4_sac/"

# Split the dataset
# split_dataset(env_name, SEED)

# Get the observation attributes
default_attr = get_obs_keys(False, False, False, False, False)
delta_time_attr = get_obs_keys(False, False, True, False, False)
time_attr = get_obs_keys(False, False, True, True, False)


default_reward = EconomicReward
res1_reward = DynamicEconomicReward(res_penalty=0.1)
res2_reward = DynamicEconomicReward(res_penalty=0.4)
res3_reward = DynamicEconomicReward(res_penalty=0.7)

# SEED = 123456789
SEED = seed(test_path)

# Default Economic Reward
r_train_env, r_val_env = env_test_val(train_name, val_name, default_attr, default_reward, SEED, False)
sac(test_path + r_sac_path, r_train_env, r_val_env, train_ep, eval_ep, "MultiInputPolicy", SEED)

# Dynamic Economic Reward
rd1_train_env, rd1_val_env = env_test_val(train_name, val_name, default_attr, res1_reward, SEED, False)
sac(test_path + rd1_sac_path, rd1_train_env, rd1_val_env, train_ep, eval_ep, "MultiInputPolicy", SEED)

rd2_train_env, rd2_val_env = env_test_val(train_name, val_name, default_attr, res2_reward, SEED, False)
sac(test_path + rd2_sac_path, rd2_train_env, rd2_val_env, train_ep, eval_ep, "MultiInputPolicy", SEED)

rd3_train_env, rd3_val_env = env_test_val(train_name, val_name, default_attr, res3_reward, SEED, False)
sac(test_path + rd3_sac_path, rd3_train_env, rd3_val_env, train_ep, eval_ep, "MultiInputPolicy", SEED)


# Train and validate the models