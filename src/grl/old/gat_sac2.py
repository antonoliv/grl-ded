import time

from models.sac import GAT_SAC


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


test_path = path + "grl6/"

path = "gcn_sac/"

SEED = 123213123

import torch as th

config = {
    "env_path": env_name,
    "res_bonus": 0.4,
    "train_episodes": 5000,
    "eval_episodes": 250,
    "learning_rate": 1e-4,
    "gamma": 0.95,
    "use_sde": False,
    "ent_coef": "auto",
    "optimizer": th.optim.AdamW,  # Optimizer
    "activation_fn": th.nn.ReLU,  # Activation Function
    "num_units_layer": 128,
    "num_hidden_layers": 6,
    "sde_sample_freq": -1,
    "gradient_steps": 1,
    "buffer_size": int(1e6),
    "batch_size": 128,
    "tau": 0.001,
    "target_update_interval": 10,
    "learning_starts": int(1e3),
    "train_freq": 100,
    "gnn_in_channels": 6,
    "gnn_hidden_channels": 36,
    "gnn_out_channels": 36,
    "gnn_num_layers": 3,
    "gnn_dropout": 0.0,
    "gnn_heads": 6,
    "gnn_gatv2": False,
    "gnn_concat": True,
    # action_noise=None,
    # replay_buffer_class=None,
    # replay_buffer_kwargs=None,
    # optimize_memory_usage=False,
    # target_entropy="auto",
    # use_sde_at_warmup=False,
}


gcn_sac = GAT_SAC(test_path + path, SEED)
gcn_sac.train_and_validate(config)
