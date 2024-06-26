import time

from environment.reward.res_penalty_reward import RESPenaltyReward
from models.sac import GCN_SAC


def seed(path):
    # SEED = 123456789
    SEED = int(time.time())

    # Save the seed
    with open(path + "seed.txt", "w") as file:
        file.write(f"Seed: {SEED}\n")

    return SEED


# Path to save the experiments
path = "/home/treeman/school/dissertation/src/grl/experiments/"

# Environment Name
# env_path = "l2rpn_case14_sandbox"
env_name = "l2rpn_icaps_2021_small"
# env_path = "l2rpn_idf_2023"


test_path = path + "gcn_sac/"

SEED = 142312345

import torch as th

config = {
    # General Parameters
    "env_path": env_name,
    "train_episodes": 2,
    "eval_episodes": 2,
    "seed": SEED,
    # Environment Parameters
    "obs_step": True,
    "obs_scaled_p": False,
    "reward": RESPenaltyReward,
    "res_term": 0.4,
    # DRL Parameters
    "learning_rate": 1e-4,
    "gamma": 0.85,
    "ent_coef": "auto",
    "gradient_steps": 1,
    "buffer_size": int(1e6),
    "batch_size": 256,
    "tau": 0.001,
    "target_update_interval": 1,
    "learning_starts": int(1e3),
    "train_freq": 1,
    "target_entropy": "auto",
    "use_sde": False,
    "sde_sample_freq": -1,
    "use_sde_at_warmup": False,
    "action_noise": None,
    "optimizer": th.optim.Adam,  # Optimizer
    "activation_fn": th.nn.ReLU,  # Activation Function
    "num_units_layer": 128,
    "num_hidden_layers": 6,
    # Graph Neural Network Parameters
    "gnn_in_channels": 6,
    "gnn_hidden_channels": 36,
    "gnn_out_channels": 36,
    "gnn_num_layers": 3,
    "gnn_dropout": 0.0,
}

# config['path'] = test_path + "nocurtail/"
# config['act_no_curtail'] = True
# config['climit_type'] = None
# config['climit_low'] = 0.0
# config['climit_end'] = 0
# config['climit_factor'] = 0.0

# gcn_sac = GCN_SAC(config['path'], config['seed'])
# gcn_sac.train_and_validate(config)

# config2 = config.copy()
# config2['path'] = test_path + "fixed_limit/"
# config2['act_no_curtail'] = False
# config2['climit_type'] = "fixed"
# config2['climit_low'] = 0.5
# config2['climit_end'] = 0
# config2['climit_factor'] = 0.0
#
# gcn_sac = GCN_SAC(config2['path'], config2['seed'])
# gcn_sac.train_and_validate(config2)

config3 = config.copy()
config3["path"] = test_path + "sqrt_limit/"
config3["act_no_curtail"] = False
config3["climit_type"] = "sqrt"
config3["climit_low"] = 0.4
config3["climit_end"] = 1800
config3["climit_factor"] = 3

gcn_sac = GCN_SAC(config3["path"], config3["seed"])
gcn_sac.train_and_validate(config3)

config3 = config.copy()
config3["path"] = test_path + "nolimit/"
config3["act_no_curtail"] = False
config3["climit_type"] = "fixed"
config3["climit_low"] = 0.0
config3["climit_end"] = 0
config3["climit_factor"] = 0.0

gcn_sac = GCN_SAC(config3["path"], config3["seed"])
gcn_sac.train_and_validate(config3)

# ray.init()
#
# # Define the scheduler
# scheduler = ASHAScheduler(
#     metric="mean_reward",
#     mode="max",
#     max_t=15,
#     grace_period=5,
#     reduction_factor=2
# )
#
# from models.experiment import Experiment
# from ray import train, tune
#
# tuner = tune.Tuner(
#     tune.with_resources(Experiment, resources={"cpu": 1, "gpu": 1}),
#     param_space=config,
#     run_config=train.RunConfig(stop={"training_iterations": 1}),
#     tune_config=tune.TuneConfig(
#         num_samples=20,
#         max_concurrent_trials=4,
#         scheduler=scheduler
#     ),
# )
#
# results = tuner.fit()
#
# print(results.get_dataframe())
# results.get_dataframe().to_csv(test_path + "parameters.csv", index=False)
