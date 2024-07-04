import stable_baselines3
import torch as th
import torch_geometric

from environment.reward.res_penalty_reward import RESPenaltyReward
from grid2op.Reward import EconomicReward
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


sac_params = {
    "class": stable_baselines3.SAC,
    "policy": "MultiInputPolicy",
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
}

gnn_params = {
    "class": torch_geometric.nn.models.GCN,
    "in_channels": 6,
    "hidden_channels": 36,
    "num_layers": 3,
    "out_channels": 36,
    "dropout": 0.2,
    "act": "relu",
    "act_first": False,
    "act_kwargs": None,
    "norm": None,
    "norm_kwargs": None,
    "jk": None,
    "aggr": "sum",
    "aggr_kwargs": None,
    "flow": "source_to_target",
    "node_dim": -2,
    "decomposed_layers": 1,
    "improved": False,
    "cached": False,
    "add_self_loops": None,
    "normalize": True,
    "bias": True,
}

env_params = {
    "env_path": "l2rpn_icaps_2021_small",
    "reward": {"class": RESPenaltyReward, "res_penalty": 0.4},
    "obs_scaled": False,
    "obs_step": True,
    "act_no_curtail": True,
    "climit_type": None,
    "climit_end": 0,
    "climit_low": 0,
    "climit_factor": 0,
}

train_ep = 5000
eval_ep = 500

seed = 123433334
from models.sac import SAC, GCN_SAC


m = SAC(
    seed, "sac", 1, train_ep, eval_ep, sac_params, env_params
)
m.train_and_validate()


m = GCN_SAC(
    seed, "gcn_sac/no_curtail", 1, train_ep, eval_ep, sac_params, env_params, gnn_params
)
m.train_and_validate()

env_params = env_params.copy()
env_params["act_no_curtail"] = False
env_params["climit_type"] = None

m = GCN_SAC(seed, "gcn_sac/no_limit", 1, train_ep, eval_ep, sac_params, env_params, gnn_params)
m.train_and_validate()

env_params = env_params.copy()
env_params["act_no_curtail"] = False
env_params["climit_type"] = "fixed"
env_params["climit_low"] = 0.4

m = GCN_SAC(
    seed, "gcn_sac/fixed_curtail", 1, train_ep, eval_ep, sac_params, env_params, gnn_params
)
m.train_and_validate()

env_params = env_params.copy()
env_params["act_no_curtail"] = False
env_params["climit_type"] = "sqrt"
env_params["climit_low"] = 0.4
env_params["climit_end"] = 4200
env_params["climit_factor"] = 3

m = GCN_SAC(
    seed,
    "gcn_sac/sqrt_curtail",
    1,
    sac_params,
    env_params,
    gnn_params,
)

m.train_and_validate(train_ep, eval_ep)
